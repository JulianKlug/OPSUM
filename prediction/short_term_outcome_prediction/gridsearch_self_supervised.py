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
from pytorch_lightning import loggers as pl_loggers

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitSelfSupervisedModel
from prediction.outcome_prediction.Transformer.utils.callbacks import MyEarlyStopping
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.short_term_outcome_prediction.timeseries_decomposition import BucketBatchSampler, \
    prepare_subsequence_dataset
from prediction.utils.utils import ensure_dir

ch.set_float32_matmul_precision('high')

DEFAULT_GRIDSEARCH_CONFIG = {
    "n_trials": 1000,
    "batch_size": [416],
    "num_layers": [2, 3, 4],
    "model_dim": [256, 512],
    "train_noise": [1e-5, 1e-3, 1e-2],
    "weight_decay": [1e-3, 1e-4, 5e-4],
    "dropout": [0.3, 0.4, 0.5],
    "num_head": [8, 16],
    "lr": [5e-4, 1e-4, 1e-3],
    "n_lr_warm_up_steps": [0],
    "grad_clip_value": [0.5, 1.0],
    "early_stopping_step_limit": [15],
    "scheduler": ["exponential"],
    "pos_encode_factor": [0.1],
    "max_epochs": 100,
}


def launch_gridsearch_self_supervised(data_splits_path: str, output_folder: str,
                                       gridsearch_config: dict = DEFAULT_GRIDSEARCH_CONFIG,
                                       use_gpu: bool = True,
                                       storage_pwd: str = None, storage_port: int = None,
                                       storage_host: str = 'localhost'):
    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDSEARCH_CONFIG

    outcome = 'self_supervised_pretraining'

    output_folder = path.join(output_folder, outcome)
    ensure_dir(output_folder)
    output_folder = path.join(output_folder, f'ss_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    ensure_dir(output_folder)

    if storage_pwd is not None and storage_port is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(
            url=f'redis://default:{storage_pwd}@{storage_host}:{storage_port}/opsum'
        ))
    else:
        storage = None
    study = optuna.create_study(direction='minimize', storage=storage)
    splits = ch.load(path.join(data_splits_path))

    # Use target_timeseries=False and standard decomposition
    # Labels are unused — the target is the shifted input sequence itself
    all_datasets = [prepare_subsequence_dataset(x, use_gpu=use_gpu,
                                                target_interval=True,
                                                restrict_to_first_event=False,
                                                ) for x in splits]

    study.optimize(partial(get_score_self_supervised, ds=all_datasets, data_splits_path=data_splits_path,
                           output_folder=output_folder,
                           gridsearch_config=gridsearch_config,
                           use_gpu=use_gpu), n_trials=gridsearch_config['n_trials'])


def get_score_self_supervised(trial, ds, data_splits_path, output_folder,
                               gridsearch_config: dict = DEFAULT_GRIDSEARCH_CONFIG, use_gpu=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDSEARCH_CONFIG

    batch_size = trial.suggest_categorical("batch_size", choices=gridsearch_config['batch_size'])
    num_layers = trial.suggest_categorical("num_layers", choices=gridsearch_config['num_layers'])
    model_dim = trial.suggest_categorical("model_dim", choices=gridsearch_config['model_dim'])
    train_noise = trial.suggest_categorical("train_noise", choices=gridsearch_config['train_noise'])
    wd = trial.suggest_categorical("weight_decay", choices=gridsearch_config['weight_decay'])
    ff_factor = 2
    ff_dim = ff_factor * model_dim
    dropout = trial.suggest_categorical("dropout", choices=gridsearch_config['dropout'])
    num_heads = trial.suggest_categorical("num_head", choices=gridsearch_config['num_head'])
    pos_encode_factor = trial.suggest_categorical("pos_encode_factor", choices=gridsearch_config['pos_encode_factor'])
    lr = trial.suggest_categorical("lr", choices=gridsearch_config['lr'])
    n_lr_warm_up_steps = trial.suggest_categorical("n_lr_warm_up_steps", gridsearch_config['n_lr_warm_up_steps'])
    grad_clip = trial.suggest_categorical('grad_clip_value', gridsearch_config['grad_clip_value'])
    early_stopping_step_limit = trial.suggest_categorical('early_stopping_step_limit', gridsearch_config['early_stopping_step_limit'])
    scheduler = trial.suggest_categorical('scheduler', gridsearch_config['scheduler'])

    accelerator = 'gpu' if use_gpu else 'cpu'

    val_losses = []
    val_cos_sims = []
    for i, (train_dataset, val_dataset) in enumerate(ds):
        checkpoint_dir = os.path.join(output_folder, f'checkpoints_ss_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)

        # save trial.params as model config json
        trial_params_path = os.path.join(output_folder, f'trial_params_{timestamp}.json')
        with open(trial_params_path, 'w') as json_file:
            json.dump(trial.params, json_file, indent=4)

        input_dim = train_dataset[0][0].shape[-1]

        # num_classes = input_dim because the projector predicts feature vectors
        model = OPSUMTransformer(
            input_dim=input_dim,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=input_dim,
            max_dim=500,
            pos_encode_factor=pos_encode_factor,
            causal=True
        )

        # No oversampling needed for self-supervised task
        train_bucket_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_bucket_sampler,
                                  shuffle=False, drop_last=False)

        val_bucket_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
        val_loader = DataLoader(val_dataset, batch_sampler=val_bucket_sampler)

        logger = DictLogger(0)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_folder, name='tb_logs',
                                                  version=f'ss_{timestamp}_cv_{i}')
        loggers = [logger, tb_logger]

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath=checkpoint_dir,
            filename="ss_encoder_{epoch:02d}_{val_loss:.6f}",
        )

        module = LitSelfSupervisedModel(model, lr, wd, train_noise,
                                         lr_warmup_steps=n_lr_warm_up_steps,
                                         scheduler=scheduler)
        trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=gridsearch_config['max_epochs'],
                             logger=loggers,
                             log_every_n_steps=25, enable_checkpointing=True,
                             callbacks=[MyEarlyStopping(step_limit=early_stopping_step_limit,
                                                        metric='val_loss', direction='min'),
                                        checkpoint_callback],
                             gradient_clip_val=grad_clip)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_losses_epoch = np.array([x['val_loss'] for x in logger.metrics if 'val_loss' in x])
        best_val_loss = np.min(val_losses_epoch)
        val_losses.append(best_val_loss)

        val_cos_sims_epoch = [x['val_cos_sim'] for x in logger.metrics if 'val_cos_sim' in x]
        if val_cos_sims_epoch:
            best_cos_sim = np.max(val_cos_sims_epoch)
            val_cos_sims.append(best_cos_sim)

    d = dict(trial.params)
    d['model_type'] = 'self_supervised_encoder'
    d["n_trials"] = gridsearch_config['n_trials']
    d['max_epochs'] = gridsearch_config['max_epochs']
    d['median_val_loss'] = float(np.median(val_losses))
    d['best_val_loss'] = float(np.min(val_losses))
    if val_cos_sims:
        d['median_val_cos_sim'] = float(np.median(val_cos_sims))
    d['timestamp'] = timestamp
    d['best_cv_fold'] = int(np.argmin(val_losses))
    d['split_file'] = data_splits_path
    text = json.dumps(d)
    text += '\n'
    dest = path.join(output_folder, f'{os.path.basename(output_folder)}_gridsearch.jsonl')
    with open(dest, 'a') as handle:
        handle.write(text)
    print("WRITTEN in ", dest)
    return np.median(val_losses)


def train_single_config(data_splits_path: str, output_folder: str, config: dict, use_gpu: bool = True):
    """Train self-supervised encoder with a fixed config (no Optuna).

    Trains one model per CV fold, saves best checkpoint for each.
    """
    output_folder = path.join(output_folder, 'self_supervised_pretraining')
    ensure_dir(output_folder)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = path.join(output_folder, f'ss_train_{timestamp}')
    ensure_dir(output_folder)

    splits = ch.load(path.join(data_splits_path))
    all_datasets = [prepare_subsequence_dataset(x, use_gpu=use_gpu,
                                                target_interval=True,
                                                restrict_to_first_event=False,
                                                ) for x in splits]

    accelerator = 'gpu' if use_gpu else 'cpu'

    batch_size = config.get('batch_size', 416)
    num_layers = config.get('num_layers', 4)
    model_dim = config.get('model_dim', 256)
    train_noise = config.get('train_noise', 1e-3)
    wd = config.get('weight_decay', 1e-4)
    ff_dim = 2 * model_dim
    dropout = config.get('dropout', 0.3)
    num_heads = config.get('num_head', 8)
    pos_encode_factor = config.get('pos_encode_factor', 0.1)
    lr = config.get('lr', 5e-4)
    n_lr_warm_up_steps = config.get('n_lr_warm_up_steps', 0)
    grad_clip = config.get('grad_clip_value', 0.5)
    early_stopping_step_limit = config.get('early_stopping_step_limit', 15)
    scheduler = config.get('scheduler', 'exponential')
    max_epochs = config.get('max_epochs', 100)

    # Save config
    config_path = os.path.join(output_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    val_losses = []
    for i, (train_dataset, val_dataset) in enumerate(all_datasets):
        checkpoint_dir = os.path.join(output_folder, f'checkpoints_ss_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)

        input_dim = train_dataset[0][0].shape[-1]

        model = OPSUMTransformer(
            input_dim=input_dim,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=input_dim,
            max_dim=500,
            pos_encode_factor=pos_encode_factor
        )

        train_bucket_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_bucket_sampler,
                                  shuffle=False, drop_last=False)

        val_bucket_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
        val_loader = DataLoader(val_dataset, batch_sampler=val_bucket_sampler)

        logger = DictLogger(0)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_folder, name='tb_logs',
                                                  version=f'ss_{timestamp}_cv_{i}')

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath=checkpoint_dir,
            filename="ss_encoder_{epoch:02d}_{val_loss:.6f}",
        )

        module = LitSelfSupervisedModel(model, lr, wd, train_noise,
                                         lr_warmup_steps=n_lr_warm_up_steps,
                                         scheduler=scheduler)
        trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=max_epochs,
                             logger=[logger, tb_logger],
                             log_every_n_steps=25, enable_checkpointing=True,
                             callbacks=[MyEarlyStopping(step_limit=early_stopping_step_limit,
                                                        metric='val_loss', direction='min'),
                                        checkpoint_callback],
                             gradient_clip_val=grad_clip)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_losses_epoch = np.array([x['val_loss'] for x in logger.metrics if 'val_loss' in x])
        best_val_loss = float(np.min(val_losses_epoch))
        val_losses.append(best_val_loss)
        print(f"Fold {i}: best val_loss = {best_val_loss:.6f}, checkpoint = {checkpoint_callback.best_model_path}")

    print(f"\nMedian val_loss across folds: {np.median(val_losses):.6f}")
    return output_folder


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False, default=None)
    parser.add_argument('-g', '--use_gpu', type=int, required=False, default=1)
    parser.add_argument('--single', action='store_true',
                        help='Train with single fixed config (no Optuna hyperopt)')
    parser.add_argument('-spwd', '--storage_pwd', type=str, required=False, default=None)
    parser.add_argument('-sport', '--storage_port', type=int, required=False, default=None)
    parser.add_argument('-shost', '--storage_host', type=str, required=False, default=None)

    args = parser.parse_args()

    use_gpu = args.use_gpu == 1

    if args.config is not None:
        gridsearch_config = json.load(open(args.config))
    else:
        gridsearch_config = None

    if args.single:
        if gridsearch_config is None:
            gridsearch_config = {
                "batch_size": 416,
                "num_layers": 4,
                "model_dim": 256,
                "train_noise": 1e-3,
                "weight_decay": 1e-4,
                "dropout": 0.3,
                "num_head": 8,
                "lr": 5e-4,
                "n_lr_warm_up_steps": 0,
                "grad_clip_value": 0.5,
                "early_stopping_step_limit": 15,
                "scheduler": "exponential",
                "pos_encode_factor": 0.1,
                "max_epochs": 100,
            }
        train_single_config(data_splits_path=args.data_splits_path, output_folder=args.output_folder,
                            config=gridsearch_config, use_gpu=use_gpu)
    else:
        launch_gridsearch_self_supervised(data_splits_path=args.data_splits_path, output_folder=args.output_folder,
                                          gridsearch_config=gridsearch_config,
                                          use_gpu=use_gpu, storage_pwd=args.storage_pwd,
                                          storage_port=args.storage_port, storage_host=args.storage_host)
