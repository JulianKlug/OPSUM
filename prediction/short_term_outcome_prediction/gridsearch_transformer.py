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
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.short_term_outcome_prediction.timeseries_decomposition import BucketBatchSampler, \
    prepare_subsequence_dataset
from prediction.utils.utils import ensure_dir

ch.set_float32_matmul_precision('high')


def launch_gridsearch(data_splits_path:str, output_folder:str, n_trials:int=1000, use_gpu:bool=True):
    outcome = '_'.join(os.path.basename(data_splits_path).split('_')[3:6])

    output_folder = path.join(output_folder, outcome)
    ensure_dir(output_folder)
    output_folder = path.join(output_folder, f'transformer_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    ensure_dir(output_folder)

    study = optuna.create_study(direction='maximize')
    splits = ch.load(path.join(data_splits_path))
    all_datasets = [prepare_subsequence_dataset(x, use_gpu=use_gpu) for x in splits]

    study.optimize(partial(get_score, ds=all_datasets, data_splits_path=data_splits_path, output_folder=output_folder,
                           use_gpu=use_gpu), n_trials=n_trials)


def get_score(trial, ds, data_splits_path, output_folder, use_gpu=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_size = trial.suggest_categorical("batch_size", choices=[128])
    num_layers = trial.suggest_categorical("num_layers", choices=[6])
    model_dim = trial.suggest_categorical("model_dim", choices=[1024])
    train_noise = trial.suggest_loguniform("train_noise", 1e-5, 1e-3)
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

    accelerator = 'gpu' if use_gpu else 'cpu'

    val_scores = []
    best_epochs = []
    rolling_val_scores = []

    for i, (train_dataset, val_dataset) in enumerate(ds):
        checkpoint_dir = os.path.join(output_folder, f'checkpoints_short_opsum_transformer_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)

        input_dim = train_dataset[0][0].shape[-1]

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

        train_bucket_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_bucket_sampler,
                                  # shuffling is done in the bucket sampler
                                  shuffle=False, drop_last=False)

        val_bucket_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
        val_loader = DataLoader(val_dataset, batch_sampler=val_bucket_sampler)
        logger = DictLogger(0)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_auroc",
            mode="max",
            dirpath=checkpoint_dir,
            filename="short_opsum_transformer_{epoch:02d}_{val_auroc:.4f}",
        )

        module = LitModel(model, lr, wd, train_noise, lr_warmup_steps=n_lr_warm_up_steps)
        trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=50, logger=logger,
                             log_every_n_steps=25, enable_checkpointing=True,
                             callbacks=[MyEarlyStopping(step_limit=early_stopping_step_limit), checkpoint_callback],
                             gradient_clip_val=grad_clip)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_aurocs = np.array([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        best_idx = np.argmax(val_aurocs)
        rolling_val_auroc = np.median(val_aurocs[max(0, best_idx - 1): best_idx + 2])
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
    d['split_file'] = data_splits_path
    text = json.dumps(d)
    text += '\n'
    dest = path.join(output_folder, 'gridsearch.jsonl')
    with open(dest, 'a') as handle:
        handle.write(text)
    print("WRITTEN in ", dest)
    return np.median(rolling_val_scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-n', '--n_trials', type=int, required=False, default=1000)
    parser.add_argument('-g', '--use_gpu', type=int, required=False, default=1)

    args = parser.parse_args()

    use_gpu = args.use_gpu == 1

    launch_gridsearch(data_splits_path=args.data_splits_path, output_folder=args.output_folder, n_trials=args.n_trials,
                      use_gpu=use_gpu)
