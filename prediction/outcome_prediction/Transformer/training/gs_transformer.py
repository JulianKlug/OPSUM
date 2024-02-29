import os
from functools import partial
from datetime import datetime
import optuna
import torch as ch
from os import path
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.outcome_prediction.Transformer.utils.callbacks import MyEarlyStopping
from prediction.outcome_prediction.Transformer.utils.utils import prepare_dataset, DictLogger
from prediction.utils.utils import ensure_dir

ch.set_float32_matmul_precision('high')





def get_score(trial, all_ds):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ds_ub, ds_b, ds_a = all_ds
    bs = trial.suggest_categorical("batch_size", choices=[16])
    num_layers = trial.suggest_categorical("num_layers", choices=[6])
    model_dim = trial.suggest_categorical("model_dim", choices=[1024])
    train_noise = trial.suggest_loguniform("train_noise", 1e-5, 1e-3)
    is_balanced = trial.suggest_categorical("balanced", [False])
    is_aggregated = trial.suggest_categorical("feature_aggregation", [False])
    wd = trial.suggest_loguniform("weight_decay", 1e-5, 0.0002)
    ff_factor = 2
    ff_dim = ff_factor * model_dim
    dropout = trial.suggest_uniform("dropout", 0.2, 0.5)
    num_heads = trial.suggest_categorical("num_head", [16])
    pos_encode_factor = 1
    lr = trial.suggest_loguniform("lr", 0.0001, 0.001)
    n_lr_warm_up_steps = trial.suggest_categorical("n_lr_warm_up_steps", [0])
    grad_clip = trial.suggest_loguniform('grad_clip_value', 1e-3, 0.2)
    early_stopping_step_limit = trial.suggest_categorical('early_stopping_step_limit', [10])

    val_scores = []
    best_epochs = []
    rolling_val_scores = []

    ds = ds_b if is_balanced else ds_ub
    ds = ds_a if is_aggregated else ds

    input_dim = 84 * 4 if is_aggregated else 84
    
    for i, (train_dataset, val_dataset) in enumerate(ds):
        checkpoint_dir = os.path.join(OUTPUT_FOLDER, f'checkpoints_opsum_transformer_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)
        model = OPSUMTransformer(
            input_dim=input_dim,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=1,
            max_dim=500,
            pos_encode_factor=pos_encode_factor
        )

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True )
        val_loader = DataLoader(val_dataset, batch_size=1024)
        logger = DictLogger(0)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_auroc",
            mode="max",
            dirpath=checkpoint_dir,
            filename="opsum_transformer_{epoch:02d}_{val_auroc:.4f}",
        )

        module = LitModel(model, lr, wd, train_noise, lr_warmup_steps=n_lr_warm_up_steps)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1000, logger=logger,
                             log_every_n_steps = 25, enable_checkpointing=True,
                             callbacks=[MyEarlyStopping(step_limit=early_stopping_step_limit), checkpoint_callback], gradient_clip_val=grad_clip)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_aurocs = np.array([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        best_idx = np.argmax(val_aurocs)
        rolling_val_auroc = np.median(val_aurocs[max(0, best_idx -1): best_idx + 2])
        best_val_score = np.max([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        best_epoch = np.argmax([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        val_scores.append(best_val_score)
        best_epochs.append(best_epoch)
        rolling_val_scores.append(rolling_val_auroc)

    d = dict(trial.params)
    d['median_rolling_val_scores'] = float(np.median(rolling_val_scores))
    d['median_val_scores'] = float(np.median(val_scores))
    d['median_best_epochs'] = float(np.median(best_epochs))
    d['timestamp'] = timestamp
    d['best_cv_fold'] = int(np.argmax(val_scores))
    d['worst_cv_fold_val_score'] = float(np.min(val_scores))
    d['split_file'] = SPLIT_FILE
    text = json.dumps(d)
    text += '\n'
    dest = path.join(OUTPUT_FOLDER, 'gridsearch.jsonl')
    with open(dest, 'a') as handle:
        handle.write(text)
    print("WRITTEN in ", dest)
    return np.median(rolling_val_scores)


if __name__ == '__main__':
    INPUT_FOLDER = '/home/gl/gsu_prepro_01012023_233050/data_splits'
    SPLIT_FILE = 'train_data_splits_3M_Death_ts0.8_rs42_ns5.pth'
    outcome = '_'.join(SPLIT_FILE.split('_')[3:6])
    if outcome.startswith('3M_Death'):
        outcome = '3M_Death'
    elif outcome.startswith('Death_in'):
        outcome = 'Death_in_hospital'

    OUTPUT_FOLDER = '/mnt/data1/klug/output/opsum/transformer_evaluation/'
    OUTPUT_FOLDER = path.join(OUTPUT_FOLDER, outcome)
    ensure_dir(OUTPUT_FOLDER)
    OUTPUT_FOLDER = path.join(OUTPUT_FOLDER, f'transformer_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    study = optuna.create_study(direction='maximize')
    scenarios = ch.load(path.join(INPUT_FOLDER, SPLIT_FILE))
    all_datasets = [prepare_dataset(x) for x in scenarios]
    all_datasets_balanced = [prepare_dataset(x, True) for x in scenarios]
    all_datasets_aggregated = [prepare_dataset(x, False, True) for x in scenarios]
    study.optimize(partial(get_score, all_ds=(all_datasets, all_datasets_balanced, all_datasets_aggregated)), n_trials=1000)

