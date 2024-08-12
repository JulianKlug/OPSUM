import os
from os import path
from datetime import datetime
import optuna
import json

from prediction.utils.utils import ensure_dir


def launch_cluster_gridsearch(data_splits_path: str, output_folder: str,
                              gridsearch_config_path: str,
                              n_subprocesses: int = 10,
                              use_gpu:bool = True,
                              storage_pwd:str = None, storage_port:int = None, storage_host:str = 'localhost'):
    outcome = '_'.join(os.path.basename(data_splits_path).split('_')[3:6])
    study_name = f'transformer_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    with open(gridsearch_config_path, 'r') as f:
        gridsearch_config = json.load(f)

    output_folder = path.join(output_folder, outcome)
    ensure_dir(output_folder)
    output_folder = path.join(output_folder, study_name)
    ensure_dir(output_folder)
    log_folder = path.join(output_folder, 'logs')
    ensure_dir(log_folder)

    gridsearch_config['study_name'] = study_name
    gridsearch_config['n_subprocesses'] = n_subprocesses
    gridsearch_config['outcome'] = outcome
    with open(path.join(output_folder, 'gridsearch_config.json'), 'w') as f:
        json.dump(gridsearch_config, f)


    # REDIS Setup for SLURM/optuna ref: https://github.com/liukidar/stune
    if storage_pwd is not None and storage_port is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(
            url=f'redis://default:{storage_pwd}@{storage_host}:{storage_port}/opsum'
        ))
    else:
        storage = None
    study = optuna.create_study(study_name=study_name,
                                direction='maximize', storage=storage)

    current_dir = path.dirname(path.abspath(__file__))
    subprocess_py_file_path = path.join(current_dir, 'cluster_subprocess.py')
    subprocess_sbatch_file_path = path.join(current_dir, 'subprocess.sbatch')

    # launch subprocesses with sbatch
    for i in range(n_subprocesses):
        os.system(f'sbatch --export=ALL,data_splits_path={data_splits_path},output_folder={output_folder},'
                  f'trial_name={study_name},gridsearch_config_path={gridsearch_config_path},use_gpu={use_gpu},'
                    f'storage_pwd={storage_pwd},storage_port={storage_port},storage_host={storage_host},'
                  f'subprocess_py_file_path={subprocess_py_file_path} {subprocess_sbatch_file_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-c', '--gridsearch_config_path', type=str, required=True)
    parser.add_argument('-n', '--n_subprocesses', type=int, required=False, default=10)
    parser.add_argument('-g', '--use_gpu', type=int, required=False, default=1)
    parser.add_argument('-spwd', '--storage_pwd', type=str, required=False, default=None)
    parser.add_argument('-sport', '--storage_port', type=int, required=False, default=None)
    parser.add_argument('-shost', '--storage_host', type=str, required=False, default=None)

    args = parser.parse_args()

    use_gpu = args.use_gpu == 1
    launch_cluster_gridsearch(args.data_splits_path, args.output_folder, args.gridsearch_config_path,
                              n_subprocesses=args.n_subprocesses, use_gpu=use_gpu,
                              storage_pwd=args.storage_pwd, storage_port=args.storage_port, storage_host=args.storage_host)

