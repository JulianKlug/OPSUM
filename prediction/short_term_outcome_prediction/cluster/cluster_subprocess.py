import os
import torch as ch
from os import path
from functools import partial
import optuna
import json

from prediction.short_term_outcome_prediction.gridsearch_transformer import get_score
from prediction.short_term_outcome_prediction.timeseries_decomposition import prepare_subsequence_dataset


def subprocess_cluster_gridsearch(data_splits_path:str, output_folder:str, trial_name:str, gridsearch_config_path: dict,
                                  use_gpu:bool=True,
                                storage_pwd:str=None, storage_port:int=None, storage_host:str='localhost'):
    # load config
    with open(gridsearch_config_path, 'r') as f:
        gridsearch_config = json.load(f)

    if storage_pwd is not None and storage_port is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(
            url=f'redis://default:{storage_pwd}@{storage_host}:{storage_port}/opsum'
        ))
    else:
        storage = None
    study = optuna.load_study(study_name=trial_name, storage=storage)

    splits = ch.load(path.join(data_splits_path))
    all_datasets = [prepare_subsequence_dataset(x, use_gpu=use_gpu) for x in splits]

    study.optimize(partial(get_score, ds=all_datasets, data_splits_path=data_splits_path, output_folder=output_folder,
                            gridsearch_config=gridsearch_config,
                           use_gpu=use_gpu), n_trials=gridsearch_config['n_trials'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-t', '--trial_name', type=str, required=True)
    parser.add_argument('-c', '--gridsearch_config_path', type=str, required=True)
    parser.add_argument('-g', '--use_gpu', type=str, required=False, default=1)
    parser.add_argument('-spwd', '--storage_pwd', type=str, required=False, default=None)
    parser.add_argument('-sport', '--storage_port', type=int, required=False, default=None)
    parser.add_argument('-shost', '--storage_host', type=str, required=False, default='localhost')

    args = parser.parse_args()

    use_gpu = (args.use_gpu == 1) | (args.use_gpu == '1') | (args.use_gpu == 'True')
    subprocess_cluster_gridsearch(args.data_splits_path, args.output_folder, args.trial_name, args.gridsearch_config_path,
                                    use_gpu=use_gpu,
                                    storage_pwd=args.storage_pwd, storage_port=args.storage_port, storage_host=args.storage_host)