import os
from functools import partial
from datetime import datetime
import optuna
import torch as ch
from os import path
import numpy as np
import pandas as pd
import json

import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score, fbeta_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from prediction.short_term_outcome_prediction.timeseries_decomposition import prepare_aggregate_dataset, aggregate_and_label_timeseries
from prediction.utils.scoring import precision, recall, specificity
from prediction.utils.utils import ensure_dir


def focal_loss_objective(y_true, y_pred, gamma_fl=2.0, pos_weight=1.0):
    """Focal loss custom objective for XGBoost.

    Operates in logit space (compatible with predict_proba sigmoid).
    When gamma_fl=0, reduces to weighted binary cross-entropy.

    Args:
        y_true: true binary labels
        y_pred: raw margin predictions (logit space)
        gamma_fl: focusing parameter (higher = more focus on hard examples)
        pos_weight: weight for positive class (equivalent to scale_pos_weight)
    """
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)

    w = np.where(y_true == 1, pos_weight, 1.0)

    # Gradient of focal loss w.r.t. logit
    grad_pos = (1 - p) ** gamma_fl * (gamma_fl * p * np.log(p) - (1 - p))
    grad_neg = p ** gamma_fl * (-gamma_fl * (1 - p) * np.log(1 - p) + p)
    grad = (y_true * grad_pos + (1 - y_true) * grad_neg) * w

    # Approximate hessian (weighted)
    hess = np.maximum(2.0 * p * (1.0 - p), 1e-7) * w

    return grad, hess


DEFAULT_GRIDEARCH_CONFIG = {
    "n_trials": 1000,
    "target_interval": 1,
    "restrict_to_first_event": 0,
    "max_depth": [2, 12],
    "n_estimators": [500, 1000, 1500, 2000, 3000],
    "learning_rate": [0.02, 0.08],
    "reg_lambda": [1, 10, 50, 75],
    "alpha": [1, 5, 10, 15, 25],
    "early_stopping_rounds": [50],
    "scale_pos_weight": [5, 10, 25, 45, 55],
    "min_child_weight": [1, 10],
    "subsample": [0.5, 1.0],
    "colsample_bytree": [0.5, 1.0],
    "colsample_bylevel": [1.0],
    "booster": ["dart"],
    "grow_policy": ["depthwise", "lossguide"],
    "num_boost_round": [500],
    "gamma": [0.1, 0.2, 0.5, 0.75, 1.0],
    "max_delta_step": [0, 1, 5, 10],
    "focal_gamma": [0, 1.0, 2.0],
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
    study = optuna.create_study(directions=['maximize', 'maximize'], storage=storage)
    splits = ch.load(path.join(data_splits_path))

    add_lag_features = gridsearch_config.get('add_lag_features', False)
    add_rolling_features = gridsearch_config.get('add_rolling_features', False)
    all_datasets = [prepare_aggregate_dataset(x, rescale=True, target_time_to_outcome=6,
                                                target_interval=gridsearch_config['target_interval'],
                                                restrict_to_first_event=gridsearch_config['restrict_to_first_event'],
                                                add_lag_features=add_lag_features,
                                                add_rolling_features=add_rolling_features,
                                              ) for x in splits]


    study.optimize(partial(get_score_xgb, ds=all_datasets, data_splits_path=data_splits_path, output_folder=output_folder,
                            gridsearch_config=gridsearch_config, outcome=outcome),
                            n_trials=gridsearch_config['n_trials'])


def get_score_xgb(trial, ds, data_splits_path, output_folder,outcome, gridsearch_config:dict=DEFAULT_GRIDEARCH_CONFIG):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if gridsearch_config is None:
        gridsearch_config = DEFAULT_GRIDEARCH_CONFIG

    max_depth = trial.suggest_int("max_depth", gridsearch_config['max_depth'][0], gridsearch_config['max_depth'][-1])
    n_estimators = trial.suggest_categorical("n_estimators", choices=gridsearch_config['n_estimators'])
    learning_rate = trial.suggest_float("learning_rate", gridsearch_config['learning_rate'][0], gridsearch_config['learning_rate'][1], log=True)
    reg_lambda = trial.suggest_categorical("reg_lambda", choices=gridsearch_config['reg_lambda'])
    alpha = trial.suggest_categorical("alpha", choices=gridsearch_config['alpha'])
    early_stopping_rounds = trial.suggest_categorical("early_stopping_rounds", choices=gridsearch_config['early_stopping_rounds'])
    scale_pos_weight = trial.suggest_categorical("scale_pos_weight", choices=gridsearch_config['scale_pos_weight'])
    min_child_weight = trial.suggest_int("min_child_weight", gridsearch_config['min_child_weight'][0], gridsearch_config['min_child_weight'][-1])
    subsample = trial.suggest_float("subsample", gridsearch_config['subsample'][0], gridsearch_config['subsample'][-1])
    colsample_bytree = trial.suggest_float("colsample_bytree", gridsearch_config['colsample_bytree'][0], gridsearch_config['colsample_bytree'][-1])
    colsample_bylevel = trial.suggest_categorical("colsample_bylevel", choices=gridsearch_config['colsample_bylevel'])
    booster = trial.suggest_categorical("booster", choices=gridsearch_config['booster'])
    grow_policy = trial.suggest_categorical("grow_policy", choices=gridsearch_config['grow_policy'])
    num_boost_round = trial.suggest_categorical("num_boost_round", choices=gridsearch_config['num_boost_round'])
    gamma = trial.suggest_categorical("gamma", choices=gridsearch_config['gamma'])
    max_delta_step = trial.suggest_categorical("max_delta_step", choices=gridsearch_config['max_delta_step'])
    focal_gamma = trial.suggest_categorical("focal_gamma", choices=gridsearch_config['focal_gamma'])

    device = "cuda" if ch.cuda.is_available() else "cpu"

    val_auprc = []
    val_auroc = []
    best_epochs = []
    model_df = pd.DataFrame()
    for i, (fold_X_train, fold_X_val, fold_y_train, fold_y_val) in enumerate(ds):
        checkpoint_dir = os.path.join(output_folder, f'checkpoints_short_opsum_xgb_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)

        xgb_params = dict(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            reg_alpha=alpha,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            booster=booster,
            grow_policy=grow_policy,
            num_boost_round=num_boost_round,
            gamma=gamma,
            max_delta_step=max_delta_step,
            device=device,
        )

        if focal_gamma > 0:
            xgb_params['objective'] = partial(focal_loss_objective, gamma_fl=focal_gamma, pos_weight=scale_pos_weight)
            xgb_params['scale_pos_weight'] = 1
        else:
            xgb_params['scale_pos_weight'] = scale_pos_weight

        xgb_model = xgb.XGBClassifier(**xgb_params)
        trained_xgb = xgb_model.fit(fold_X_train, fold_y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=["auc", "aucpr"],
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
        run_performance_df['max_delta_step'] = max_delta_step
        run_performance_df['focal_gamma'] = focal_gamma
        run_performance_df['moving_average'] = False
        run_performance_df['add_lag_features'] = gridsearch_config.get('add_lag_features', False)
        run_performance_df['add_rolling_features'] = gridsearch_config.get('add_rolling_features', False)
        run_performance_df['n_features'] = fold_X_train.shape[1]
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
    d['add_lag_features'] = gridsearch_config.get('add_lag_features', False)
    d['add_rolling_features'] = gridsearch_config.get('add_rolling_features', False)
    d['n_features'] = int(ds[0][0].shape[1])
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
    return np.median(val_auprc), np.median(val_auroc)


def retrain_on_all_data(data_splits_path: str, config: dict, output_dir: str):
    """Retrain XGBoost on all available CV data and save the final model.

    In k-fold CV, each fold's train + val = all data. This function combines
    them, trains a single model on the full dataset, and saves:
      - xgb_final_model.model  (XGBoost model weights)
      - scaler.pkl             (fitted StandardScaler)
      - final_model_config.json (config + training metadata)

    Args:
        data_splits_path: path to the .pth file with CV splits
        config: dict with scalar hyperparameter values (not search ranges).
                Expected keys: max_depth, n_estimators, learning_rate, reg_lambda,
                alpha, scale_pos_weight, min_child_weight, subsample,
                colsample_bytree, etc.
        output_dir: directory to save model artifacts
    """
    # Validate that config has scalar values, not search ranges
    range_keys = [k for k in ('max_depth', 'n_estimators', 'learning_rate', 'reg_lambda',
                               'alpha', 'scale_pos_weight', 'min_child_weight', 'subsample',
                               'colsample_bytree')
                  if k in config and isinstance(config[k], list)]
    if range_keys:
        raise ValueError(
            f"Config contains list values for {range_keys}. "
            "retrain_on_all_data expects a fixed config with scalar values "
            "(e.g. from a hyperopt trial), not a search space config."
        )

    splits = ch.load(data_splits_path)

    # In k-fold CV, fold 0's train + val = all data
    X_train, X_val, y_train, y_val = splits[0]
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val])

    add_lag_features = config.get('add_lag_features', False)
    add_rolling_features = config.get('add_rolling_features', False)
    target_interval = config.get('target_interval', 1)
    restrict_to_first_event = config.get('restrict_to_first_event', 0)

    print(f"Preparing data (lag={add_lag_features}, rolling={add_rolling_features})...")
    all_data, all_labels = aggregate_and_label_timeseries(
        X_all, y_all, target_time_to_outcome=6,
        target_interval=target_interval,
        restrict_to_first_event=restrict_to_first_event,
        add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features,
    )

    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)

    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    n_features = all_data.shape[1]
    n_samples = all_data.shape[0]
    n_positive = int(all_labels.sum())
    print(f"Training on {n_samples} samples ({n_positive} positive, "
          f"{n_positive/n_samples*100:.2f}%), {n_features} features")

    # Build XGBoost params
    device = "cuda" if ch.cuda.is_available() else "cpu"

    focal_gamma = config.get('focal_gamma', 0)
    scale_pos_weight = config.get('scale_pos_weight', 10)

    xgb_params = dict(
        learning_rate=config['learning_rate'],
        max_depth=int(config['max_depth']),
        n_estimators=int(config['n_estimators']),
        reg_lambda=config['reg_lambda'],
        reg_alpha=config.get('alpha', config.get('reg_alpha', 1)),
        min_child_weight=config['min_child_weight'],
        subsample=config['subsample'],
        colsample_bytree=config['colsample_bytree'],
        colsample_bylevel=config.get('colsample_bylevel', 1.0),
        booster=config.get('booster', 'dart'),
        grow_policy=config.get('grow_policy', 'lossguide'),
        gamma=config.get('gamma', 0.5),
        max_delta_step=config.get('max_delta_step', 0),
        device=device,
    )

    if focal_gamma > 0:
        xgb_params['objective'] = partial(focal_loss_objective,
                                          gamma_fl=focal_gamma,
                                          pos_weight=scale_pos_weight)
        xgb_params['scale_pos_weight'] = 1
    else:
        xgb_params['scale_pos_weight'] = scale_pos_weight

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(all_data, all_labels,
              eval_metric=["auc", "aucpr"],
              eval_set=[(all_data, all_labels)],
              verbose=True)

    # Save artifacts
    ensure_dir(output_dir)

    model_path = os.path.join(output_dir, 'xgb_final_model.model')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Save config with training metadata
    config_to_save = {k: v for k, v in config.items()
                      if not callable(v)}
    config_to_save['n_features'] = n_features
    config_to_save['n_training_samples'] = n_samples
    config_to_save['n_positive_samples'] = n_positive
    config_to_save['data_splits_path'] = data_splits_path
    config_to_save['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    config_path = os.path.join(output_dir, 'final_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    print(f"Config saved to {config_path}")

    return model, scaler


def _optimal_thresholds(y_true, y_prob, thresholds=None):
    """Find optimal decision thresholds for multiple metrics.

    Args:
        y_true: true binary labels
        y_prob: predicted probabilities (continuous)
        thresholds: array of thresholds to sweep (default: 0.01-0.50 in 0.005 steps)

    Returns:
        dict mapping metric name -> {threshold, value}
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.505, 0.005)

    best = {
        'youden_j': {'threshold': 0.5, 'value': -1},
        'f1': {'threshold': 0.5, 'value': -1},
        'f2': {'threshold': 0.5, 'value': -1},
        'mcc': {'threshold': 0.5, 'value': -2},
    }

    for t in thresholds:
        y_pred = (y_prob >= t).astype('float32')

        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue

        sn = recall(y_true, y_pred).numpy()
        sp = specificity(y_true, y_pred).numpy()
        j = float(sn + sp - 1)
        if j > best['youden_j']['value']:
            best['youden_j'] = {'threshold': float(t), 'value': j}

        f1 = float(fbeta_score(y_true, y_pred, beta=1, zero_division=0))
        if f1 > best['f1']['value']:
            best['f1'] = {'threshold': float(t), 'value': f1}

        f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        if f2 > best['f2']['value']:
            best['f2'] = {'threshold': float(t), 'value': f2}

        mcc = float(matthews_corrcoef(y_true, y_pred))
        if mcc > best['mcc']['value']:
            best['mcc'] = {'threshold': float(t), 'value': mcc}

    return best


def _metrics_at_threshold(y_true, y_prob, threshold):
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype('float32')
    return {
        'auroc': float(roc_auc_score(y_true, y_prob)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision(y_true, y_pred.astype(float)).numpy()),
        'recall': float(recall(y_true, y_pred).numpy()),
        'specificity': float(specificity(y_true, y_pred).numpy()),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'f1': float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        'f2': float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        'youden_j': float(recall(y_true, y_pred).numpy() + specificity(y_true, y_pred).numpy() - 1),
    }


def evaluate_with_threshold_tuning(data_splits_path: str, config: dict, output_dir: str):
    """Run 5-fold CV with threshold optimization for the best XGB config.

    For each fold:
      - Train XGB with the given config
      - Sweep thresholds on validation predictions
      - Record optimal threshold per metric (Youden's J, F1, F2, MCC)

    Saves:
      - threshold_tuning_results.json: per-fold optimal thresholds + median thresholds
      - threshold_tuning_cv.csv: per-fold metrics at each optimal threshold

    Args:
        data_splits_path: path to .pth file with CV data splits
        config: dict with scalar hyperparameter values
        output_dir: directory to save results
    """
    range_keys = [k for k in ('max_depth', 'n_estimators', 'learning_rate', 'reg_lambda',
                               'alpha', 'scale_pos_weight', 'min_child_weight', 'subsample',
                               'colsample_bytree')
                  if k in config and isinstance(config[k], list)]
    if range_keys:
        raise ValueError(
            f"Config contains list values for {range_keys}. "
            "evaluate_with_threshold_tuning expects a fixed config with scalar values."
        )

    ensure_dir(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    splits = ch.load(data_splits_path)
    outcome = '_'.join(os.path.basename(data_splits_path).split('_')[3:6])

    add_lag_features = config.get('add_lag_features', False)
    add_rolling_features = config.get('add_rolling_features', False)
    target_interval = config.get('target_interval', 1)
    restrict_to_first_event = config.get('restrict_to_first_event', 0)

    print(f"Preparing data (lag={add_lag_features}, rolling={add_rolling_features})...")
    all_datasets = [prepare_aggregate_dataset(x, rescale=True, target_time_to_outcome=6,
                                               target_interval=target_interval,
                                               restrict_to_first_event=restrict_to_first_event,
                                               add_lag_features=add_lag_features,
                                               add_rolling_features=add_rolling_features)
                    for x in splits]

    device = "cuda" if ch.cuda.is_available() else "cpu"
    focal_gamma = config.get('focal_gamma', 0)
    scale_pos_weight = config.get('scale_pos_weight', 10)

    xgb_params = dict(
        learning_rate=config['learning_rate'],
        max_depth=int(config['max_depth']),
        n_estimators=int(config['n_estimators']),
        reg_lambda=config['reg_lambda'],
        reg_alpha=config.get('alpha', config.get('reg_alpha', 1)),
        min_child_weight=config['min_child_weight'],
        subsample=config['subsample'],
        colsample_bytree=config['colsample_bytree'],
        colsample_bylevel=config.get('colsample_bylevel', 1.0),
        booster=config.get('booster', 'dart'),
        grow_policy=config.get('grow_policy', 'lossguide'),
        gamma=config.get('gamma', 0.5),
        max_delta_step=config.get('max_delta_step', 0),
        device=device,
    )

    if focal_gamma > 0:
        xgb_params['objective'] = partial(focal_loss_objective,
                                          gamma_fl=focal_gamma,
                                          pos_weight=scale_pos_weight)
        xgb_params['scale_pos_weight'] = 1
    else:
        xgb_params['scale_pos_weight'] = scale_pos_weight

    early_stopping_rounds = config.get('early_stopping_rounds', 50)
    metric_names = ['youden_j', 'f1', 'f2', 'mcc']
    per_fold_thresholds = {m: [] for m in metric_names}
    per_fold_results = []

    for i, (fold_X_train, fold_X_val, fold_y_train, fold_y_val) in enumerate(all_datasets):
        print(f"\n--- Fold {i} ---")
        print(f"  Train: {len(fold_y_train)} samples, {int(fold_y_train.sum())} positive ({fold_y_train.sum()/len(fold_y_train)*100:.2f}%)")
        print(f"  Val:   {len(fold_y_val)} samples, {int(fold_y_val.sum())} positive ({fold_y_val.sum()/len(fold_y_val)*100:.2f}%)")

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(fold_X_train, fold_y_train,
                  early_stopping_rounds=early_stopping_rounds,
                  eval_metric=["auc", "aucpr"],
                  eval_set=[(fold_X_train, fold_y_train), (fold_X_val, fold_y_val)],
                  verbose=False)

        y_prob_val = model.predict_proba(fold_X_val)[:, 1].astype('float32')

        # Find optimal thresholds
        opt = _optimal_thresholds(fold_y_val, y_prob_val)

        fold_result = {
            'fold': i,
            'best_iteration': int(model.best_iteration),
            'n_val_samples': len(fold_y_val),
            'n_val_positive': int(fold_y_val.sum()),
        }

        # Metrics at default 0.5 threshold
        fold_result['default_0.5'] = _metrics_at_threshold(fold_y_val, y_prob_val, 0.5)

        # Metrics at each optimal threshold
        for m in metric_names:
            t = opt[m]['threshold']
            per_fold_thresholds[m].append(t)
            fold_result[m] = {
                'optimal_threshold': t,
                'optimized_value': opt[m]['value'],
                'metrics_at_threshold': _metrics_at_threshold(fold_y_val, y_prob_val, t),
            }

        per_fold_results.append(fold_result)

        print(f"  AUROC: {fold_result['default_0.5']['auroc']:.4f}, AUPRC: {fold_result['default_0.5']['auprc']:.4f}")
        for m in metric_names:
            print(f"  Best {m}: threshold={opt[m]['threshold']:.3f}, value={opt[m]['value']:.4f}")

    # Compute median thresholds across folds
    median_thresholds = {}
    for m in metric_names:
        thresholds = per_fold_thresholds[m]
        median_thresholds[m] = {
            'median_threshold': float(np.median(thresholds)),
            'per_fold_thresholds': [float(t) for t in thresholds],
            'median_optimized_value': float(np.median([r[m]['optimized_value'] for r in per_fold_results])),
        }

    results = {
        'timestamp': timestamp,
        'config': {k: v for k, v in config.items() if not callable(v)},
        'data_splits_path': data_splits_path,
        'outcome': outcome,
        'n_folds': len(splits),
        'n_features': int(all_datasets[0][0].shape[1]),
        'median_thresholds': median_thresholds,
        'per_fold': per_fold_results,
    }

    results_path = os.path.join(output_dir, f'threshold_tuning_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print(f"\n=== Threshold Tuning Summary ===")
    print(f"{'Metric':<12} {'Median Threshold':>18} {'Median Value':>14}")
    for m in metric_names:
        mt = median_thresholds[m]
        print(f"{m:<12} {mt['median_threshold']:>18.3f} {mt['median_optimized_value']:>14.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=False, default=None)
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain on all CV data with a fixed config (requires -c)')
    parser.add_argument('--threshold_tuning', action='store_true',
                        help='Run 5-fold CV with threshold optimization (requires -c)')

    parser.add_argument('-spwd', '--storage_pwd', type=str, required=False, default=None)
    parser.add_argument('-sport', '--storage_port', type=int, required=False, default=None)
    parser.add_argument('-shost', '--storage_host', type=str, required=False, default=None)

    args = parser.parse_args()

    if args.config is not None:
        gridsearch_config = json.load(open(args.config))
    else:
        gridsearch_config = None

    if args.retrain:
        if gridsearch_config is None:
            parser.error("--retrain requires -c/--config with a fixed hyperparameter config")
        retrain_on_all_data(data_splits_path=args.data_splits_path,
                            config=gridsearch_config,
                            output_dir=args.output_folder)
    elif args.threshold_tuning:
        if gridsearch_config is None:
            parser.error("--threshold_tuning requires -c/--config with a fixed hyperparameter config")
        evaluate_with_threshold_tuning(data_splits_path=args.data_splits_path,
                                       config=gridsearch_config,
                                       output_dir=args.output_folder)
    else:
        launch_gridsearch_xgb(data_splits_path=args.data_splits_path, output_folder=args.output_folder,
                              gridsearch_config=gridsearch_config,
                              storage_pwd=args.storage_pwd, storage_port=args.storage_port,
                              storage_host=args.storage_host)
