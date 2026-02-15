"""Single-fold A/B evaluation: Self-supervised encoder representations + XGBoost.

Runs on fold 0 only to compare:
  A) Baseline XGBoost (hand-crafted features only)
  B) Encoder representations only
  C) Combined (hand-crafted + encoder representations)

Usage:
    python prediction/short_term_outcome_prediction/evaluate_ss_xgb_single_fold.py
"""

import os
import sys
import json
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch as ch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitSelfSupervisedModel
from prediction.outcome_prediction.Transformer.utils.callbacks import MyEarlyStopping
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.short_term_outcome_prediction.timeseries_decomposition import (
    BucketBatchSampler,
    prepare_subsequence_dataset,
    prepare_aggregate_dataset,
)
from prediction.short_term_outcome_prediction.gridsearch_aggregate_xgb import focal_loss_objective
from prediction.utils.utils import ensure_dir

ch.set_float32_matmul_precision('high')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
OUTPUT_DIR = '/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/ss_xgb_eval'

# ── Self-supervised encoder config ─────────────────────────────────────────
SS_CONFIG = {
    "batch_size": 416,
    "num_layers": 4,
    "model_dim": 256,
    "train_noise": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "num_head": 8,
    "lr": 5e-4,
    "grad_clip_value": 0.5,
    "early_stopping_step_limit": 15,
    "pos_encode_factor": 0.1,
    "max_epochs": 100,
}

# ── XGBoost config (from best hyperopt) ────────────────────────────────────
XGB_CONFIG = {
    "max_depth": 4,
    "n_estimators": 1000,
    "learning_rate": 0.044,
    "reg_lambda": 100,
    "reg_alpha": 10,
    "early_stopping_rounds": 100,
    "scale_pos_weight": 10,
    "min_child_weight": 2,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 1.0,
    "booster": "dart",
    "grow_policy": "lossguide",
    "gamma": 0.75,
}

FOLD_IDX = 0
USE_GPU = ch.cuda.is_available()


def train_self_supervised_fold0(splits, output_dir, use_gpu=False):
    """Train self-supervised encoder on fold 0."""
    print("\n" + "=" * 70)
    print("STEP 1: Training self-supervised encoder on fold 0")
    print("=" * 70)

    scenario = splits[FOLD_IDX]
    train_dataset, val_dataset = prepare_subsequence_dataset(
        scenario, use_gpu=use_gpu, target_interval=True, restrict_to_first_event=False
    )

    input_dim = train_dataset[0][0].shape[-1]
    print(f"  Input dim: {input_dim}")
    print(f"  Train subsequences: {len(train_dataset)}")
    print(f"  Val subsequences: {len(val_dataset)}")

    model = OPSUMTransformer(
        input_dim=input_dim,
        num_layers=SS_CONFIG['num_layers'],
        model_dim=SS_CONFIG['model_dim'],
        dropout=SS_CONFIG['dropout'],
        ff_dim=2 * SS_CONFIG['model_dim'],
        num_heads=SS_CONFIG['num_head'],
        num_classes=input_dim,  # projector predicts feature vectors
        max_dim=500,
        pos_encode_factor=SS_CONFIG['pos_encode_factor'],
        causal=True,
    )

    train_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, SS_CONFIG['batch_size'])
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)

    val_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    checkpoint_dir = os.path.join(output_dir, 'ss_checkpoints')
    ensure_dir(checkpoint_dir)

    logger = DictLogger(0)
    checkpoint_cb = ModelCheckpoint(
        save_top_k=1, monitor="val_loss", mode="min",
        dirpath=checkpoint_dir,
        filename="ss_encoder_{epoch:02d}_{val_loss:.6f}",
    )

    module = LitSelfSupervisedModel(
        model, lr=SS_CONFIG['lr'], wd=SS_CONFIG['weight_decay'],
        train_noise=SS_CONFIG['train_noise'],
    )

    accelerator = 'gpu' if use_gpu else 'cpu'
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1,
        max_epochs=SS_CONFIG['max_epochs'],
        logger=[logger],
        log_every_n_steps=10,
        enable_checkpointing=True,
        callbacks=[
            MyEarlyStopping(step_limit=SS_CONFIG['early_stopping_step_limit'],
                            metric='val_loss', direction='min'),
            checkpoint_cb,
        ],
        gradient_clip_val=SS_CONFIG['grad_clip_value'],
    )

    start = time.time()
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - start

    val_losses = [x['val_loss'] for x in logger.metrics if 'val_loss' in x]
    best_val_loss = min(val_losses)
    best_epoch = np.argmin(val_losses)

    print(f"\n  Training completed in {elapsed:.0f}s")
    print(f"  Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"  Checkpoint: {checkpoint_cb.best_model_path}")

    return checkpoint_cb.best_model_path, input_dim


def extract_representations_fold0(checkpoint_path, input_dim, splits, use_gpu=False):
    """Extract encoder representations for fold 0."""
    print("\n" + "=" * 70)
    print("STEP 2: Extracting encoder representations for fold 0")
    print("=" * 70)

    scenario = splits[FOLD_IDX]
    train_dataset, val_dataset = prepare_subsequence_dataset(
        scenario, use_gpu=use_gpu, target_interval=True, restrict_to_first_event=False
    )

    # Reconstruct and load model
    model = OPSUMTransformer(
        input_dim=input_dim,
        num_layers=SS_CONFIG['num_layers'],
        model_dim=SS_CONFIG['model_dim'],
        dropout=SS_CONFIG['dropout'],
        ff_dim=2 * SS_CONFIG['model_dim'],
        num_heads=SS_CONFIG['num_head'],
        num_classes=input_dim,
        max_dim=500,
        pos_encode_factor=SS_CONFIG['pos_encode_factor'],
        causal=True,
    )

    lit_model = LitSelfSupervisedModel.load_from_checkpoint(
        checkpoint_path, model=model, lr=0, wd=0, train_noise=0
    )
    model = lit_model.model
    if use_gpu and ch.cuda.is_available():
        model = model.cuda()
    model.eval()

    def extract_agg(dataset, label=""):
        """Extract last-hidden-state representations per subsequence."""
        sampler = BucketBatchSampler(dataset.idx_to_len_map, 1024)
        loader = DataLoader(dataset, batch_sampler=sampler)
        all_reprs, all_labels = [], []
        with ch.no_grad():
            for x, y in loader:
                hidden = model.encode(x)  # (batch, seq_len, 2*model_dim)
                last_h = hidden[:, -1, :]  # (batch, 2*model_dim)
                all_reprs.append(last_h.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        reprs = np.concatenate(all_reprs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print(f"  {label}: {reprs.shape[0]} samples, repr dim = {reprs.shape[1]}")
        return reprs, labels

    train_reprs, train_labels = extract_agg(train_dataset, "Train")
    val_reprs, val_labels = extract_agg(val_dataset, "Val")

    return train_reprs, val_reprs, train_labels, val_labels


def run_xgb(X_train, y_train, X_val, y_val, label=""):
    """Train XGBoost and return metrics."""
    device = "cuda" if ch.cuda.is_available() else "cpu"

    xgb_params = dict(
        learning_rate=XGB_CONFIG['learning_rate'],
        max_depth=XGB_CONFIG['max_depth'],
        n_estimators=XGB_CONFIG['n_estimators'],
        reg_lambda=XGB_CONFIG['reg_lambda'],
        reg_alpha=XGB_CONFIG['reg_alpha'],
        min_child_weight=XGB_CONFIG['min_child_weight'],
        subsample=XGB_CONFIG['subsample'],
        colsample_bytree=XGB_CONFIG['colsample_bytree'],
        colsample_bylevel=XGB_CONFIG['colsample_bylevel'],
        booster=XGB_CONFIG['booster'],
        grow_policy=XGB_CONFIG['grow_policy'],
        gamma=XGB_CONFIG['gamma'],
        scale_pos_weight=XGB_CONFIG['scale_pos_weight'],
        device=device,
    )

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train,
              early_stopping_rounds=XGB_CONFIG['early_stopping_rounds'],
              eval_metric=["auc", "aucpr"],
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=False)

    y_prob = model.predict_proba(X_val)[:, 1].astype('float32')
    auroc = roc_auc_score(y_val, y_prob)
    auprc = average_precision_score(y_val, y_prob)

    print(f"  [{label}] n_features={X_train.shape[1]}, "
          f"best_iter={model.best_iteration}, "
          f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    return {'auroc': auroc, 'auprc': auprc, 'n_features': X_train.shape[1],
            'best_iteration': model.best_iteration}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to existing SS checkpoint (skip training)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_DIR, f'eval_{timestamp}')
    ensure_dir(output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Data: {DATA_PATH}")

    # Load data
    splits = ch.load(DATA_PATH)

    # ── Step 1: Train self-supervised encoder ──────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
        # Infer input_dim from data
        scenario = splits[FOLD_IDX]
        input_dim = scenario[0].shape[2]  # (n_patients, n_timesteps, n_features, n_channels)
        print(f"\nSkipping training, using checkpoint: {ckpt_path}")
    else:
        ckpt_path, input_dim = train_self_supervised_fold0(splits, output_dir, use_gpu=USE_GPU)

    # ── Step 2: Extract representations ────────────────────────────────────
    train_reprs, val_reprs, repr_train_labels, repr_val_labels = \
        extract_representations_fold0(ckpt_path, input_dim, splits, use_gpu=USE_GPU)

    # ── Step 3: Prepare hand-crafted features (XGBoost baseline) ───────────
    print("\n" + "=" * 70)
    print("STEP 3: Running XGBoost A/B comparison on fold 0")
    print("=" * 70)

    scenario = splits[FOLD_IDX]
    hc_X_train, hc_X_val, hc_y_train, hc_y_val = prepare_aggregate_dataset(
        scenario, rescale=True, target_time_to_outcome=6,
        target_interval=True, restrict_to_first_event=False,
        add_lag_features=False, add_rolling_features=False,
    )

    print(f"\n  Hand-crafted features: train={hc_X_train.shape}, val={hc_X_val.shape}")
    print(f"  Encoder representations: train={train_reprs.shape}, val={val_reprs.shape}")
    print(f"  Train positive rate: {hc_y_train.mean():.4f} ({int(hc_y_train.sum())}/{len(hc_y_train)})")
    print(f"  Val positive rate: {hc_y_val.mean():.4f} ({int(hc_y_val.sum())}/{len(hc_y_val)})")

    # Verify labels match between hand-crafted and encoder-extracted data
    assert len(hc_y_train) == len(repr_train_labels), \
        f"Train label mismatch: {len(hc_y_train)} vs {len(repr_train_labels)}"
    assert len(hc_y_val) == len(repr_val_labels), \
        f"Val label mismatch: {len(hc_y_val)} vs {len(repr_val_labels)}"

    # ── A) Baseline: hand-crafted features only ────────────────────────────
    print("\n  --- A) Baseline: hand-crafted features only ---")
    results_a = run_xgb(hc_X_train, hc_y_train, hc_X_val, hc_y_val, label="Baseline")

    # ── B) Encoder representations only ────────────────────────────────────
    print("\n  --- B) Encoder representations only ---")
    results_b = run_xgb(train_reprs, repr_train_labels, val_reprs, repr_val_labels, label="Repr only")

    # ── C) Combined: hand-crafted + encoder representations ────────────────
    print("\n  --- C) Combined: hand-crafted + encoder representations ---")
    combined_X_train = np.concatenate([hc_X_train, train_reprs], axis=1)
    combined_X_val = np.concatenate([hc_X_val, val_reprs], axis=1)
    results_c = run_xgb(combined_X_train, hc_y_train, combined_X_val, hc_y_val, label="Combined")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Fold 0)")
    print("=" * 70)
    print(f"{'Method':<30} {'AUROC':>8} {'AUPRC':>8} {'Features':>10}")
    print("-" * 60)
    print(f"{'A) Hand-crafted only':<30} {results_a['auroc']:>8.4f} {results_a['auprc']:>8.4f} {results_a['n_features']:>10}")
    print(f"{'B) Encoder repr only':<30} {results_b['auroc']:>8.4f} {results_b['auprc']:>8.4f} {results_b['n_features']:>10}")
    print(f"{'C) Combined':<30} {results_c['auroc']:>8.4f} {results_c['auprc']:>8.4f} {results_c['n_features']:>10}")
    print("-" * 60)

    # Save results
    results = {
        'timestamp': timestamp,
        'fold': FOLD_IDX,
        'ss_config': SS_CONFIG,
        'xgb_config': XGB_CONFIG,
        'baseline': results_a,
        'repr_only': results_b,
        'combined': results_c,
        'data_path': DATA_PATH,
        'checkpoint_path': ckpt_path,
    }
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
