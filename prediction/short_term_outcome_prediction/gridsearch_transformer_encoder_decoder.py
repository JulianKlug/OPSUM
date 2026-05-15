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
from pytorch_lightning import loggers as pl_loggers
import json

from prediction.outcome_prediction.Transformer.architecture import OPSUM_encoder_decoder
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitEncoderDecoderModel
from prediction.outcome_prediction.Transformer.utils.callbacks import MyEarlyStopping
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.short_term_outcome_prediction.evaluation.dec_val_evaluation import encoder_decoder_validation_evaluation
from prediction.short_term_outcome_prediction.timeseries_decomposition import BucketBatchSampler, \
    prepare_subsequence_dataset
from prediction.utils.utils import ensure_dir

ch.set_float32_matmul_precision('high')

DEFAULT_GRIDEARCH_CONFIG = {
    "n_trials": 1000,
    "batch_size": [416],
    "num_layers": [6],
    "num_decoder_layers": [6],
    "model_dim": [1024],
    "train_noise": [1e-5, 1e-3],
    "weight_decay": [1e-5, 0.0002],
    "dropout": [0.1, 0.5],
    "num_head": [16],
    "pos_encode_factor": [0.1, 1],
    "lr": [0.0001, 0.001],
    "n_lr_warm_up_steps": [0],
    "grad_clip_value": [1e-3, 0.2],
    "early_stopping_step_limit": [10],
    "imbalance_factor": 62,
    "max_epochs": 50,
    "target_timeseries_length": 1,
    "loss_function": ['weighted_mse'],
}

def launch_gridsearch_encoder_decoder(data_splits_path:str, output_folder:str, gridsearch_config:dict=DEFAULT_GRIDEARCH_CONFIG, 
                                      normalisation_data_path:str=None, outcome_data_path:str=None,
                                      use_gpu:bool=True,
                                     storage_pwd:str=None, storage_port:int=None, storage_host:str='localhost'):
    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDEARCH_CONFIG

    outcome = '_'.join(os.path.basename(data_splits_path).split('_')[3:6])

    output_folder = path.join(output_folder, outcome)
    ensure_dir(output_folder)
    output_folder = path.join(output_folder, f'transformer_dec_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    ensure_dir(output_folder)

    if storage_pwd is not None and storage_port is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(
            url=f'redis://default:{storage_pwd}@{storage_host}:{storage_port}/opsum'
        ))
    else:
        storage = None
    study = optuna.create_study(direction='maximize', storage=storage)
    splits = ch.load(path.join(data_splits_path))
    all_datasets = [prepare_subsequence_dataset(x, use_gpu=use_gpu, use_target_timeseries=True,
                                                target_timeseries_length=gridsearch_config['target_timeseries_length']) for x in splits]

    study.optimize(partial(get_score_encoder_decoder, ds=all_datasets, data_splits_path=data_splits_path, output_folder=output_folder,
                            gridsearch_config=gridsearch_config,
                            normalisation_data_path=normalisation_data_path, outcome_data_path=outcome_data_path,
                           use_gpu=use_gpu), n_trials=gridsearch_config['n_trials'])


def get_score_encoder_decoder(trial, ds, data_splits_path, output_folder, gridsearch_config:dict=DEFAULT_GRIDEARCH_CONFIG,
                              normalisation_data_path:str=None, outcome_data_path:str=None,
                               use_gpu=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDEARCH_CONFIG
    batch_size = trial.suggest_categorical("batch_size", choices=gridsearch_config['batch_size'])
    num_layers = trial.suggest_categorical("num_layers", choices=gridsearch_config['num_layers'])
    num_decoder_layers = trial.suggest_categorical("num_decoder_layers", choices=gridsearch_config['num_decoder_layers'])
    model_dim = trial.suggest_categorical("model_dim", choices=gridsearch_config['model_dim'])
    train_noise = trial.suggest_loguniform("train_noise", gridsearch_config['train_noise'][0], gridsearch_config['train_noise'][1])
    wd = trial.suggest_loguniform("weight_decay", gridsearch_config['weight_decay'][0], gridsearch_config['weight_decay'][1])
    ff_factor = 2
    ff_dim = ff_factor * model_dim
    dropout = trial.suggest_uniform("dropout", gridsearch_config['dropout'][0], gridsearch_config['dropout'][1])
    num_heads = trial.suggest_categorical("num_head", gridsearch_config['num_head'])
    pos_encode_factor = trial.suggest_uniform("pos_encode_factor", gridsearch_config['pos_encode_factor'][0], gridsearch_config['pos_encode_factor'][1])
    lr = trial.suggest_loguniform("lr", gridsearch_config['lr'][0], gridsearch_config['lr'][1])
    n_lr_warm_up_steps = trial.suggest_categorical("n_lr_warm_up_steps", gridsearch_config['n_lr_warm_up_steps'])
    grad_clip = trial.suggest_loguniform('grad_clip_value', gridsearch_config['grad_clip_value'][0], gridsearch_config['grad_clip_value'][1])
    early_stopping_step_limit = trial.suggest_categorical('early_stopping_step_limit', gridsearch_config['early_stopping_step_limit'])
    loss_function = trial.suggest_categorical('loss_function', gridsearch_config['loss_function'])

    accelerator = 'gpu' if use_gpu else 'cpu'

    val_cos_sim_scores = []
    val_roc_scores = []

    for i, (train_dataset, val_dataset) in enumerate(ds):
        checkpoint_dir = os.path.join(output_folder, f'checkpoints_short_opsum_transformer_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)

        # save trial.params as model config json
        trial_params_path = os.path.join(output_folder, f'trial_params_{timestamp}.json')
        with open(trial_params_path, 'w') as json_file:
            json.dump(trial.params, json_file, indent=4)    


        input_dim = train_dataset[0][0].shape[-1]

        model = OPSUM_encoder_decoder(input_dim=input_dim, num_layers=num_layers, num_decoder_layers=num_decoder_layers,
                                      model_dim=model_dim, ff_dim=ff_dim, num_heads=num_heads, dropout=dropout,
                                      pos_encode_factor=pos_encode_factor, n_tokens=1, max_dim=5000, layer_norm_eps=1e-05)

        train_bucket_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_bucket_sampler,
                                  # shuffling is done in the bucket sampler
                                  shuffle=False, drop_last=False)

        val_bucket_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
        val_loader = DataLoader(val_dataset, batch_sampler=val_bucket_sampler)
        logger = DictLogger(0)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_folder, name='tb_logs', version=f'short_opsum_transformer_{timestamp}_cv_{i}')
        loggers = [logger, tb_logger]

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_cos_sim",
            mode="max",
            dirpath=checkpoint_dir,
            filename="short_opsum_dec_transformer_{epoch:02d}_{val_cos_sim:.4f}",
        )

        module = LitEncoderDecoderModel(model, lr, wd, train_noise, lr_warmup_steps=n_lr_warm_up_steps, 
                                        loss_function=loss_function)
        trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=gridsearch_config['max_epochs'],
                             logger=loggers,
                             log_every_n_steps=25, enable_checkpointing=True,
                             callbacks=[MyEarlyStopping(step_limit=early_stopping_step_limit, metric='val_cos_sim',
                                                        direction='max'),
                                        checkpoint_callback],
                             gradient_clip_val=grad_clip,
                             num_sanity_val_steps=0)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        overall_prediction_results_df = encoder_decoder_validation_evaluation(
                                            data_path=data_splits_path, model_config_path=trial_params_path, model_path=checkpoint_callback.best_model_path, 
                                            normalisation_data_path=normalisation_data_path, outcome_data_path=outcome_data_path, 
                                            cv_fold=i,
                                            use_gpu = use_gpu,  n_time_steps = 72, eval_n_time_steps_before_event = 6)
        
        best_val_cos_sim = np.max([x['val_cos_sim'] for x in logger.metrics if 'val_cos_sim' in x])
        val_cos_sim_scores.append(best_val_cos_sim)
        best_roc = overall_prediction_results_df['overall_roc_augit ac'].max()
        val_roc_scores.append(best_roc)

    d = dict(trial.params)
    d['model_type'] = 'transformer_encoder_decoder'
    d['best_val_roc_auc'] = float(np.max(val_roc_scores))
    d['median_val_cos_sim'] = float(np.median(val_cos_sim_scores))
    d['median_val_roc_auc'] = float(np.median(val_roc_scores))
    d['timestamp'] = timestamp
    d['best_cv_fold'] = int(np.argmax(val_roc_scores))
    d['worst_cv_fold_val_score'] = float(np.min(val_roc_scores))
    d['split_file'] = data_splits_path
    text = json.dumps(d)
    text += '\n'
    dest = path.join(output_folder, f'{os.path.basename(output_folder)}_gridsearch.jsonl')
    with open(dest, 'a') as handle:
        handle.write(text)
    print("WRITTEN in ", dest)
    return float(np.median(val_roc_scores))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-nd', '--normalisation_data_path', type=str, required=False, default=None)
    parser.add_argument('-od', '--outcome_data_path', type=str, required=False, default=None)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False, default=None)
    parser.add_argument('-g', '--use_gpu', type=int, required=False, default=1)
    parser.add_argument('-spwd', '--storage_pwd', type=str, required=False, default=None)
    parser.add_argument('-sport', '--storage_port', type=int, required=False, default=None)
    parser.add_argument('-shost', '--storage_host', type=str, required=False, default=None)

    args = parser.parse_args()

    use_gpu = args.use_gpu == 1

    if args.config is not None:
        gridsearch_config = json.load(open(args.config))
    else:
        gridsearch_config = None

    launch_gridsearch_encoder_decoder(data_splits_path=args.data_splits_path, output_folder=args.output_folder, gridsearch_config=gridsearch_config,
                                        normalisation_data_path=args.normalisation_data_path, outcome_data_path=args.outcome_data_path,
                      use_gpu=use_gpu, storage_pwd=args.storage_pwd, storage_port=args.storage_port, storage_host=args.storage_host)
