import os
from functools import partial
from datetime import datetime
import optuna
import torch as ch
from os import path
import numpy as np
import pandas as pd
import json

from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score
import xgboost as xgb

from prediction.short_term_outcome_prediction.timeseries_decomposition import prepare_aggregate_dataset
from prediction.utils.scoring import precision, recall, specificity
from prediction.utils.utils import ensure_dir


DEFAULT_GRIDEARCH_CONFIG = {
    "n_trials": 1000,
    "target_interval": 1,
    "restrict_to_first_event": 0,
    "max_depth": [2, 6, 8, 10, 12],
    "n_estimators": [100, 250, 500, 1000],
    "learning_rate": [0.001, 0.1],
    "reg_lambda": [1, 10, 50, 75],
    "alpha": [0, 1, 10, 25],
    "early_stopping_rounds": [25, 50, 100],
    "scale_pos_weight": [25, 55, 100],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.5, 0.8, 1],
    "colsample_bytree": [0.8, 1],
    "colsample_bylevel": [0.8, 1],
    "booster": ["gbtree", "dart"],
    "grow_policy": ["depthwise", "lossguide"],
    "num_boost_round": [100, 200, 300],
    "gamma": [0, 0.5, 1],
}

def launch_gridsearch_xgb(data_splits_path:str, output_folder:str, gridsearch_config:dict=DEFAULT_GRIDEARCH_CONFIG,
                      storage_pwd:str=None, storage_port:int=None, storage_host:str='localhost'):
    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDEARCH_CONFIG

    outcome = '_'.join(os.path.basename(data_splits_path).split('_')[3:6])

    output_folder = path.join(output_folder, outcome)
    ensure_dir(output_folder)
    output_folder = path.join(output_folder, f'xgb_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    ensure_dir(output_folder)

    if storage_pwd is not None and storage_port is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(
            url=f'redis://default:{storage_pwd}@{storage_host}:{storage_port}/opsum'
        ))
    else:
        storage = None
    study = optuna.create_study(direction='maximize', storage=storage)
    splits = ch.load(path.join(data_splits_path))

    all_datasets = [prepare_aggregate_dataset(x, rescale=True, target_time_to_outcome=6,
                                                target_interval=gridsearch_config['target_interval'], 
                                                restrict_to_first_event=gridsearch_config['restrict_to_first_event'],
                                              ) for x in splits]


    study.optimize(partial(get_score_xgb, ds=all_datasets, data_splits_path=data_splits_path, output_folder=output_folder,
                            gridsearch_config=gridsearch_config, outcome=outcome),
                            n_trials=gridsearch_config['n_trials'])


def get_score_xgb(trial, ds, data_splits_path, output_folder,outcome, gridsearch_config:dict=DEFAULT_GRIDEARCH_CONFIG):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDEARCH_CONFIG

    max_depth = trial.suggest_categorical("max_depth", choices=gridsearch_config['max_depth'])
    n_estimators = trial.suggest_categorical("n_estimators", choices=gridsearch_config['n_estimators'])
    learning_rate = trial.suggest_loguniform("learning_rate", gridsearch_config['learning_rate'][0], gridsearch_config['learning_rate'][1])
    reg_lambda = trial.suggest_categorical("reg_lambda", choices=gridsearch_config['reg_lambda'])
    alpha = trial.suggest_categorical("alpha", choices=gridsearch_config['alpha'])
    early_stopping_rounds = trial.suggest_categorical("early_stopping_rounds", choices=gridsearch_config['early_stopping_rounds'])
    scale_pos_weight = trial.suggest_categorical("scale_pos_weight", choices=gridsearch_config['scale_pos_weight'])
    min_child_weight = trial.suggest_categorical("min_child_weight", choices=gridsearch_config['min_child_weight'])
    subsample = trial.suggest_categorical("subsample", choices=gridsearch_config['subsample'])
    colsample_bytree = trial.suggest_categorical("colsample_bytree", choices=gridsearch_config['colsample_bytree'])
    colsample_bylevel = trial.suggest_categorical("colsample_bylevel", choices=gridsearch_config['colsample_bylevel'])
    booster = trial.suggest_categorical("booster", choices=gridsearch_config['booster'])
    grow_policy = trial.suggest_categorical("grow_policy", choices=gridsearch_config['grow_policy'])
    num_boost_round = trial.suggest_categorical("num_boost_round", choices=gridsearch_config['num_boost_round'])
    gamma = trial.suggest_categorical("gamma", choices=gridsearch_config['gamma'])

    device = "cuda" if ch.cuda.is_available() else "cpu"

    val_auprc = []
    val_auroc = []
    best_epochs = []
    model_df = pd.DataFrame()
    for i, (fold_X_train, fold_X_val, fold_y_train, fold_y_val) in enumerate(ds):
        checkpoint_dir = os.path.join(output_folder, f'checkpoints_short_opsum_xgb_{timestamp}')
        ensure_dir(checkpoint_dir)

        xgb_model = xgb.XGBClassifier(
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            n_estimators=n_estimators,
            reg_lambda=reg_lambda, 
            reg_alpha=alpha,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            booster=booster,
            grow_policy=grow_policy,
            num_boost_round=num_boost_round,
            gamma=gamma,
            device=device
            )
        trained_xgb = xgb_model.fit(fold_X_train, fold_y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=["aucpr", "auc"],
                                    eval_set=[(fold_X_train, fold_y_train), (fold_X_val, fold_y_val)])

        # save trained model
        model_path = os.path.join(checkpoint_dir, f'xgb_{timestamp}_cv_{i}.model')
        trained_xgb.save_model(model_path)


        model_y_val = trained_xgb.predict_proba(fold_X_val)[:, 1].astype('float32')
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0).astype('float32')
        model_acc_val = accuracy_score(fold_y_val, model_y_pred_val)
        model_precision_val = precision(fold_y_val, model_y_pred_val.astype(float)).numpy()
        model_sn_val = recall(fold_y_val, model_y_pred_val).numpy()
        model_auc_val = roc_auc_score(fold_y_val, model_y_val)
        model_mcc_val = matthews_corrcoef(fold_y_val, model_y_pred_val)
        model_sp_val = specificity(fold_y_val, model_y_pred_val).numpy()
        model_auprc_val = average_precision_score(fold_y_val, model_y_val)

        model_y_train = trained_xgb.predict_proba(fold_X_train)[:, 1].astype('float32')
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0).astype('float32')
        model_acc_train = accuracy_score(fold_y_train, model_y_pred_train)
        model_precision_train = precision(fold_y_train, model_y_pred_train.astype(float)).numpy()
        model_sn_train = recall(fold_y_train, model_y_pred_train).numpy()
        model_auc_train = roc_auc_score(fold_y_train, model_y_train)
        model_mcc_train = matthews_corrcoef(fold_y_train, model_y_pred_train)
        model_sp_train = specificity(fold_y_train, model_y_pred_train).numpy()
        model_auprc_train = average_precision_score(fold_y_train, model_y_train)

        # save model performance
        run_performance_df = pd.DataFrame(index=[0])
        run_performance_df['CV'] = i
        run_performance_df['epoch'] = trained_xgb.best_iteration
        run_performance_df['max_depth'] = max_depth
        run_performance_df['n_estimators'] = n_estimators
        run_performance_df['learning_rate'] = learning_rate
        run_performance_df['alpha'] = alpha
        run_performance_df['reg_lambda'] = reg_lambda
        run_performance_df['early_stopping_rounds'] = early_stopping_rounds
        run_performance_df['scale_pos_weight'] = scale_pos_weight
        run_performance_df['min_child_weight'] = min_child_weight
        run_performance_df['subsample'] = subsample
        run_performance_df['colsample_bytree'] = colsample_bytree
        run_performance_df['colsample_bylevel'] = colsample_bylevel
        run_performance_df['booster'] = booster
        run_performance_df['grow_policy'] = grow_policy
        run_performance_df['num_boost_round'] = num_boost_round
        run_performance_df['gamma'] = gamma 
        run_performance_df['moving_average'] = False
        run_performance_df['outcome'] = outcome

        run_performance_df['auc_train'] = model_auc_train
        run_performance_df['auc_val'] = model_auc_val
        run_performance_df['auprc_train'] = model_auprc_train
        run_performance_df['auprc_val'] = model_auprc_val
        run_performance_df['mcc_train'] = model_mcc_train
        run_performance_df['mcc_val'] = model_mcc_val
        run_performance_df['acc_train'] = model_acc_train
        run_performance_df['acc_val'] = model_acc_val
        run_performance_df['precision_train'] = model_precision_train
        run_performance_df['precision_val'] = model_precision_val
        run_performance_df['sn_train'] = model_sn_train
        run_performance_df['sn_val'] = model_sn_val
        run_performance_df['sp_train'] = model_sp_train
        run_performance_df['sp_val'] = model_sp_val
        model_df = pd.concat([model_df, run_performance_df])

        best_val_auprc = model_auprc_val
        best_val_auc = model_auc_val
        best_epoch = trained_xgb.best_iteration
        val_auprc.append(best_val_auprc)
        val_auroc.append(best_val_auc)
        best_epochs.append(best_epoch)

    model_df.to_csv(os.path.join(output_folder, f'xgb_{timestamp}.csv'))

    d = dict(trial.params)
    d["n_trials"] = gridsearch_config['n_trials']
    d['target_interval'] = gridsearch_config['target_interval']
    d['restrict_to_first_event'] = gridsearch_config['restrict_to_first_event']
    d['median_val_auprc'] = float(np.median(val_auprc))
    d['median_val_auc'] = float(np.median(val_auroc))

    d['median_best_epochs'] = float(np.median(best_epochs))
    d['timestamp'] = timestamp
    d['best_cv_fold'] = int(np.argmax(val_auprc))
    d['worst_cv_fold_val_score'] = float(np.min(val_auprc))
    d['split_file'] = data_splits_path
    text = json.dumps(d)
    text += '\n'
    dest = path.join(output_folder, f'{os.path.basename(output_folder)}_gridsearch.jsonl')
    with open(dest, 'a') as handle:
        handle.write(text)
    print("WRITTEN in ", dest)
    return np.median(val_auroc)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False, default=None)

    parser.add_argument('-spwd', '--storage_pwd', type=str, required=False, default=None)
    parser.add_argument('-sport', '--storage_port', type=int, required=False, default=None)
    parser.add_argument('-shost', '--storage_host', type=str, required=False, default=None)

    args = parser.parse_args()

    if args.config is not None:
        gridsearch_config = json.load(open(args.config))
    else:
        gridsearch_config = None

    launch_gridsearch_xgb(data_splits_path=args.data_splits_path, output_folder=args.output_folder, gridsearch_config=gridsearch_config,
                          storage_pwd=args.storage_pwd, storage_port=args.storage_port, storage_host=args.storage_host)
