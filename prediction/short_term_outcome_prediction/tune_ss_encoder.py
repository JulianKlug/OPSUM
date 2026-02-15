"""Tune self-supervised encoder hyperparameters on fold 0.

Trains multiple configs, extracts representations from each, and evaluates
combined XGBoost AUPRC to find the best encoder setup.
"""

import os
import sys
import json
import time
from datetime import datetime
from functools import partial
from itertools import product

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
from prediction.utils.utils import ensure_dir

ch.set_float32_matmul_precision('high')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
OUTPUT_DIR = '/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/ss_xgb_eval'

# ── XGBoost config (from best hyperopt — held constant) ────────────────────
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

# ── Encoder configs to try ─────────────────────────────────────────────────
CONFIGS = [
    # Original config (baseline, already ran — 2 epoch peak)
    {"name": "baseline", "model_dim": 256, "num_layers": 4, "num_head": 8, "lr": 5e-4, "dropout": 0.3, "train_noise": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Lower learning rate — give model more time to learn
    {"name": "lr_1e-4", "model_dim": 256, "num_layers": 4, "num_head": 8, "lr": 1e-4, "dropout": 0.3, "train_noise": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Even lower LR
    {"name": "lr_5e-5", "model_dim": 256, "num_layers": 4, "num_head": 8, "lr": 5e-5, "dropout": 0.3, "train_noise": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Smaller model — less overfitting risk
    {"name": "small_model", "model_dim": 128, "num_layers": 2, "num_head": 8, "lr": 5e-4, "dropout": 0.3, "train_noise": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Smaller model + lower LR
    {"name": "small_lr_1e-4", "model_dim": 128, "num_layers": 2, "num_head": 8, "lr": 1e-4, "dropout": 0.3, "train_noise": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Higher dropout + more noise
    {"name": "high_reg", "model_dim": 256, "num_layers": 4, "num_head": 8, "lr": 1e-4, "dropout": 0.5, "train_noise": 1e-2, "weight_decay": 1e-3, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Deeper small model
    {"name": "deep_small", "model_dim": 128, "num_layers": 4, "num_head": 8, "lr": 1e-4, "dropout": 0.3, "train_noise": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
    # Medium model
    {"name": "medium", "model_dim": 192, "num_layers": 3, "num_head": 8, "lr": 2e-4, "dropout": 0.4, "train_noise": 1e-3, "weight_decay": 5e-4, "max_epochs": 100, "batch_size": 416, "pos_encode_factor": 0.1, "grad_clip_value": 0.5, "early_stopping_step_limit": 15},
]

FOLD_IDX = 0
USE_GPU = ch.cuda.is_available()


def train_ss_encoder(config, train_dataset, val_dataset, output_dir):
    """Train a single self-supervised encoder config."""
    input_dim = train_dataset[0][0].shape[-1]

    model = OPSUMTransformer(
        input_dim=input_dim,
        num_layers=config['num_layers'],
        model_dim=config['model_dim'],
        dropout=config['dropout'],
        ff_dim=2 * config['model_dim'],
        num_heads=config['num_head'],
        num_classes=input_dim,
        max_dim=500,
        pos_encode_factor=config['pos_encode_factor'],
        causal=True,
    )

    train_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, config['batch_size'])
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)
    val_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    ensure_dir(checkpoint_dir)

    logger = DictLogger(0)
    checkpoint_cb = ModelCheckpoint(
        save_top_k=1, monitor="val_loss", mode="min",
        dirpath=checkpoint_dir,
        filename="ss_{epoch:02d}_{val_loss:.6f}",
    )

    module = LitSelfSupervisedModel(
        model, lr=config['lr'], wd=config['weight_decay'],
        train_noise=config['train_noise'],
    )

    accelerator = 'gpu' if USE_GPU else 'cpu'
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1,
        max_epochs=config['max_epochs'],
        logger=[logger],
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=False,
        callbacks=[
            MyEarlyStopping(step_limit=config['early_stopping_step_limit'],
                            metric='val_loss', direction='min'),
            checkpoint_cb,
        ],
        gradient_clip_val=config['grad_clip_value'],
    )

    start = time.time()
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elapsed = time.time() - start

    val_losses = [x['val_loss'] for x in logger.metrics if 'val_loss' in x]
    best_val_loss = min(val_losses) if val_losses else float('inf')
    best_epoch = np.argmin(val_losses) if val_losses else -1
    total_epochs = len(val_losses)

    return {
        'checkpoint': checkpoint_cb.best_model_path,
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_epoch),
        'total_epochs': total_epochs,
        'train_time_s': elapsed,
    }


def extract_representations(checkpoint_path, config, train_dataset, val_dataset):
    """Extract encoder representations from a checkpoint."""
    input_dim = train_dataset[0][0].shape[-1]

    model = OPSUMTransformer(
        input_dim=input_dim,
        num_layers=config['num_layers'],
        model_dim=config['model_dim'],
        dropout=config['dropout'],
        ff_dim=2 * config['model_dim'],
        num_heads=config['num_head'],
        num_classes=input_dim,
        max_dim=500,
        pos_encode_factor=config['pos_encode_factor'],
        causal=True,
    )

    lit_model = LitSelfSupervisedModel.load_from_checkpoint(
        checkpoint_path, model=model, lr=0, wd=0, train_noise=0
    )
    model = lit_model.model
    if USE_GPU:
        model = model.cuda()
    model.eval()

    def _extract(dataset):
        sampler = BucketBatchSampler(dataset.idx_to_len_map, 1024)
        loader = DataLoader(dataset, batch_sampler=sampler)
        all_reprs, all_labels = [], []
        with ch.no_grad():
            for x, y in loader:
                hidden = model.encode(x)
                last_h = hidden[:, -1, :]
                all_reprs.append(last_h.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        return np.concatenate(all_reprs, axis=0), np.concatenate(all_labels, axis=0)

    train_reprs, train_labels = _extract(train_dataset)
    val_reprs, val_labels = _extract(val_dataset)
    return train_reprs, val_reprs, train_labels, val_labels


def evaluate_xgb(hc_X_train, hc_y_train, hc_X_val, hc_y_val, train_reprs, val_reprs):
    """Run XGBoost with combined features and return AUPRC."""
    combined_X_train = np.concatenate([hc_X_train, train_reprs], axis=1)
    combined_X_val = np.concatenate([hc_X_val, val_reprs], axis=1)

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
    model.fit(combined_X_train, hc_y_train,
              early_stopping_rounds=XGB_CONFIG['early_stopping_rounds'],
              eval_metric=["auc", "aucpr"],
              eval_set=[(combined_X_train, hc_y_train), (combined_X_val, hc_y_val)],
              verbose=False)

    y_prob = model.predict_proba(combined_X_val)[:, 1].astype('float32')
    auroc = roc_auc_score(hc_y_val, y_prob)
    auprc = average_precision_score(hc_y_val, y_prob)
    return {'auroc': auroc, 'auprc': auprc, 'n_features': combined_X_train.shape[1],
            'best_iteration': model.best_iteration}


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_DIR, f'tune_{timestamp}')
    ensure_dir(output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Configs to evaluate: {len(CONFIGS)}")

    # Load data
    print("Loading data...")
    splits = ch.load(DATA_PATH)
    scenario = splits[FOLD_IDX]

    # Prepare transformer dataset (for SS training + repr extraction)
    print("Preparing subsequence dataset...")
    train_dataset, val_dataset = prepare_subsequence_dataset(
        scenario, use_gpu=USE_GPU, target_interval=True, restrict_to_first_event=False
    )
    print(f"  Train: {len(train_dataset)} subsequences, Val: {len(val_dataset)} subsequences")

    # Prepare hand-crafted features (for XGBoost baseline)
    print("Preparing aggregate dataset...")
    hc_X_train, hc_X_val, hc_y_train, hc_y_val = prepare_aggregate_dataset(
        scenario, rescale=True, target_time_to_outcome=6,
        target_interval=True, restrict_to_first_event=False,
    )
    print(f"  Hand-crafted: train={hc_X_train.shape}, val={hc_X_val.shape}")
    print(f"  Positive rate: train={hc_y_train.mean():.4f}, val={hc_y_val.mean():.4f}")

    # XGBoost baseline (no encoder)
    print("\n" + "=" * 80)
    print("BASELINE: XGBoost with hand-crafted features only")
    print("=" * 80)
    device = "cuda" if ch.cuda.is_available() else "cpu"
    baseline_model = xgb.XGBClassifier(
        **{k: v for k, v in XGB_CONFIG.items() if k != 'early_stopping_rounds'},
        device=device,
    )
    baseline_model.fit(hc_X_train, hc_y_train,
                       early_stopping_rounds=XGB_CONFIG['early_stopping_rounds'],
                       eval_metric=["auc", "aucpr"],
                       eval_set=[(hc_X_train, hc_y_train), (hc_X_val, hc_y_val)],
                       verbose=False)
    baseline_prob = baseline_model.predict_proba(hc_X_val)[:, 1].astype('float32')
    baseline_auroc = roc_auc_score(hc_y_val, baseline_prob)
    baseline_auprc = average_precision_score(hc_y_val, baseline_prob)
    print(f"  Baseline AUROC={baseline_auroc:.4f}, AUPRC={baseline_auprc:.4f} ({hc_X_train.shape[1]} features)")

    # Run each encoder config
    all_results = []
    print(f"\n{'=' * 80}")
    print(f"ENCODER TUNING: {len(CONFIGS)} configurations")
    print(f"{'=' * 80}")

    for idx, config in enumerate(CONFIGS):
        name = config['name']
        print(f"\n{'─' * 80}")
        print(f"[{idx+1}/{len(CONFIGS)}] Config: {name}")
        print(f"  model_dim={config['model_dim']}, layers={config['num_layers']}, "
              f"heads={config['num_head']}, lr={config['lr']}, "
              f"dropout={config['dropout']}, noise={config['train_noise']}")

        config_dir = os.path.join(output_dir, name)
        ensure_dir(config_dir)

        # Train
        print("  Training...")
        train_result = train_ss_encoder(config, train_dataset, val_dataset, config_dir)
        print(f"  Trained {train_result['total_epochs']} epochs, "
              f"best val_loss={train_result['best_val_loss']:.6f} at epoch {train_result['best_epoch']}, "
              f"time={train_result['train_time_s']:.0f}s")

        # Extract representations
        print("  Extracting representations...")
        train_reprs, val_reprs, _, _ = extract_representations(
            train_result['checkpoint'], config, train_dataset, val_dataset
        )
        repr_dim = train_reprs.shape[1]
        print(f"  Repr dim: {repr_dim}")

        # Evaluate with XGBoost
        print("  Evaluating XGBoost (combined)...")
        xgb_result = evaluate_xgb(hc_X_train, hc_y_train, hc_X_val, hc_y_val,
                                   train_reprs, val_reprs)
        print(f"  Combined AUROC={xgb_result['auroc']:.4f}, AUPRC={xgb_result['auprc']:.4f}")

        result = {
            'name': name,
            'config': {k: v for k, v in config.items() if k != 'name'},
            'training': train_result,
            'repr_dim': repr_dim,
            'xgb': xgb_result,
            'auprc_delta': xgb_result['auprc'] - baseline_auprc,
        }
        all_results.append(result)

        # Save intermediate results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump({
                'baseline': {'auroc': baseline_auroc, 'auprc': baseline_auprc},
                'configs': all_results,
            }, f, indent=2)

    # Final summary
    print(f"\n{'=' * 80}")
    print("TUNING RESULTS SUMMARY (Fold 0)")
    print(f"{'=' * 80}")
    print(f"{'Config':<20} {'Dim':>5} {'Epochs':>7} {'ValLoss':>8} {'AUROC':>7} {'AUPRC':>7} {'Delta':>7}")
    print("─" * 70)
    print(f"{'Baseline (no enc)':<20} {'':>5} {'':>7} {'':>8} {baseline_auroc:>7.4f} {baseline_auprc:>7.4f} {'':>7}")
    print("─" * 70)

    for r in sorted(all_results, key=lambda x: -x['xgb']['auprc']):
        delta_str = f"+{r['auprc_delta']:.4f}" if r['auprc_delta'] > 0 else f"{r['auprc_delta']:.4f}"
        print(f"{r['name']:<20} {r['repr_dim']:>5} {r['training']['best_epoch']:>7} "
              f"{r['training']['best_val_loss']:>8.5f} {r['xgb']['auroc']:>7.4f} "
              f"{r['xgb']['auprc']:>7.4f} {delta_str:>7}")

    best = max(all_results, key=lambda x: x['xgb']['auprc'])
    print(f"\nBest config: {best['name']} (AUPRC={best['xgb']['auprc']:.4f}, "
          f"delta={best['auprc_delta']:+.4f})")
    print(f"Checkpoint: {best['training']['checkpoint']}")


if __name__ == '__main__':
    main()
