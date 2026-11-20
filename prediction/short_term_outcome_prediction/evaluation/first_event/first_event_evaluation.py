"""
END model performance counting only each patient's first END event (reviewer response).

Subset (hour bins, as in aggregate_and_label_timeseries):
    prediction at ts uses bins 0..ts, label = END in bins ts+1..ts+6
    keep ts iff ts < E1 (first END bin); patients without END keep all timesteps
    -> later END events and the post-event course no longer contribute

    t0           E1            E2
    |============|-------------|------> hours
     kept         dropped

Cohorts:
    training       out-of-fold predictions: final config retrained per CV fold, predicted on its validation fold
    internal_test  final model on held-out test set
    external_test  final model on MIMIC

Each cohort is reported on all timesteps (reference) and on the first-event subset,
with patient-level bootstrap 95% CI.
"""
import argparse
import json
import os
from enum import Enum
from functools import partial

import numpy as np
import pandas as pd
import torch as ch
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.concern_timing_analysis import \
    predict_hourly, load_thresholds, PREDICTION_HORIZON_H
from prediction.short_term_outcome_prediction.gridsearch_aggregate_xgb import focal_loss_objective
from prediction.short_term_outcome_prediction.testing.test_xgb import _compute_metrics
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.utils import ensure_dir

N_BOOTSTRAP = 1000
SEED = 42
CI_LOW_Q, CI_HIGH_Q = 2.5, 97.5
EVENT_BIN = 'relative_sample_date_hourly_cat'


class Cohort(str, Enum):
    TRAINING = 'training_cv'
    INTERNAL_TEST = 'internal_test'
    EXTERNAL_TEST = 'external_test'


class Subset(str, Enum):
    ALL_EVENTS = 'all_events'
    FIRST_EVENT = 'first_event_only'


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------

def restrict_to_first_event(preds: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    """Keep timesteps ts < first END bin (patients without END keep all)."""
    first = y_df.groupby('case_admission_id')[EVENT_BIN].min().rename('first_end_bin')
    df = preds.merge(first, left_on='case_admission_id', right_index=True, how='left')
    keep = df.first_end_bin.isna() | (df.ts < df.first_end_bin)
    return df[keep].drop(columns='first_end_bin')


# ---------------------------------------------------------------------------
# Out-of-fold predictions (training cohort)
# ---------------------------------------------------------------------------

def _xgb_params(config: dict) -> dict:
    """Same parameters as evaluate_with_threshold_tuning (gridsearch_aggregate_xgb.py)."""
    scale_pos_weight = config.get('scale_pos_weight', 10)
    params = dict(
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
        device='cuda' if ch.cuda.is_available() else 'cpu',
        scale_pos_weight=scale_pos_weight,
    )

    focal_gamma = config.get('focal_gamma', 0)
    if focal_gamma > 0:
        params['objective'] = partial(focal_loss_objective, gamma_fl=focal_gamma, pos_weight=scale_pos_weight)
        params['scale_pos_weight'] = 1
    return params


def _aggregate(X: np.ndarray, y_df: pd.DataFrame, config: dict):
    return aggregate_and_label_timeseries(
        X, y_df,
        target_time_to_outcome=PREDICTION_HORIZON_H,
        target_interval=config.get('target_interval', True),
        restrict_to_first_event=config.get('restrict_to_first_event', False),
        add_lag_features=config.get('add_lag_features', False),
        add_rolling_features=config.get('add_rolling_features', False),
    )


def predict_out_of_fold(splits_path: str, config: dict) -> tuple:
    """Retrain per fold, predict its validation patients. Each training patient is validated exactly once."""
    splits = ch.load(splits_path, weights_only=False)
    params = _xgb_params(config)
    frames, y_dfs = [], []

    for fold, (X_train, X_val, y_train, y_val) in enumerate(splits):
        train_data, train_labels = _aggregate(X_train, y_train, config)
        val_data, val_labels = _aggregate(X_val, y_val, config)

        # same scaling and early stopping as threshold tuning
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(np.concatenate(train_data))
        y_fit = np.concatenate(train_labels)
        X_eval = scaler.transform(np.concatenate(val_data))
        y_eval = np.concatenate(val_labels)

        model = xgb.XGBClassifier(**params)
        model.fit(X_fit, y_fit, early_stopping_rounds=config.get('early_stopping_rounds', 50),
                  eval_metric=['auc', 'aucpr'], eval_set=[(X_fit, y_fit), (X_eval, y_eval)], verbose=False)

        prob = model.predict_proba(X_eval)[:, 1]
        cids = np.concatenate([np.full(len(d), cid) for cid, d in zip(X_val[:, 0, 0, 0], val_data)])
        ts = np.concatenate([np.arange(len(d)) for d in val_data])
        frames.append(pd.DataFrame({'case_admission_id': cids, 'ts': ts, 'y_true': y_eval,
                                    'y_prob': prob, 'fold': fold}))
        y_dfs.append(y_val)
        print(f'Fold {fold}: val AUROC {roc_auc_score(y_eval, prob):.4f}')

    return pd.concat(frames, ignore_index=True), pd.concat(y_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics_with_ci(preds: pd.DataFrame, thresholds: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Point estimates + patient-level bootstrap 95% CI."""
    y, p = preds.y_true.values, preds.y_prob.values
    rows_by_patient = list(preds.groupby('case_admission_id').indices.values())
    n_patients = len(rows_by_patient)

    point = {name: _compute_metrics(y, p, thr) for name, thr in thresholds.items()}

    samples = {name: [] for name in thresholds}
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([rows_by_patient[i] for i in rng.integers(0, n_patients, n_patients)])
        y_bs, p_bs = y[idx], p[idx]
        if len(np.unique(y_bs)) < 2:
            continue

        auroc, auprc = roc_auc_score(y_bs, p_bs), average_precision_score(y_bs, p_bs)
        for name, thr in thresholds.items():
            samples[name].append(_compute_metrics(y_bs, p_bs, thr, auroc=auroc, auprc=auprc))

    rows = []
    for name, thr in thresholds.items():
        bs_df = pd.DataFrame(samples[name])
        for metric, value in point[name].items():
            rows.append({'threshold_name': name, 'threshold': thr, 'metric': metric, 'value': value,
                         'ci_low': np.percentile(bs_df[metric], CI_LOW_Q),
                         'ci_high': np.percentile(bs_df[metric], CI_HIGH_Q)})
    return pd.DataFrame(rows)


def _counts(preds: pd.DataFrame) -> dict:
    return {'n_patients': preds.case_admission_id.nunique(),
            'n_patients_with_positive': preds[preds.y_true == 1].case_admission_id.nunique(),
            'n_timepoints': len(preds), 'n_positive': int(preds.y_true.sum())}


def evaluate_cohort(preds: pd.DataFrame, y_df: pd.DataFrame, thresholds: dict, cohort: Cohort) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    subsets = {Subset.ALL_EVENTS: preds, Subset.FIRST_EVENT: restrict_to_first_event(preds, y_df)}

    frames = []
    for subset, df in subsets.items():
        metrics = _metrics_with_ci(df, thresholds, rng)
        frames.append(metrics.assign(cohort=cohort.value, subset=subset.value, **_counts(df)))
        print(f'{cohort.value} / {subset.value}: {_counts(df)}, '
              f'AUROC {roc_auc_score(df.y_true, df.y_prob):.4f}')
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(splits_path: str, test_data_path: str, external_data_path: str, model_dir: str,
        threshold_path: str, output_dir: str):
    ensure_dir(output_dir)
    with open(os.path.join(model_dir, 'final_model_config.json')) as f:
        config = json.load(f)
    thresholds = load_thresholds(threshold_path)

    # hourly predictions per cohort
    cohort_preds = {Cohort.TRAINING: predict_out_of_fold(splits_path, config)}
    for cohort, path in [(Cohort.INTERNAL_TEST, test_data_path), (Cohort.EXTERNAL_TEST, external_data_path)]:
        X, y_df = ch.load(path, weights_only=False)
        cohort_preds[cohort] = (predict_hourly(X, y_df, model_dir), y_df)

    pd.concat([p.assign(cohort=c.value) for c, (p, _) in cohort_preds.items()]) \
        .to_csv(os.path.join(output_dir, 'hourly_predictions.csv'), index=False)

    results = pd.concat([evaluate_cohort(p, y_df, thresholds, c) for c, (p, y_df) in cohort_preds.items()])
    leading = ['cohort', 'subset', 'threshold_name', 'threshold', 'metric', 'value', 'ci_low', 'ci_high']
    results = results[leading + [c for c in results.columns if c not in leading]]
    results.to_csv(os.path.join(output_dir, 'first_event_performance.csv'), index=False)
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--splits_path', required=True, help='CV data splits .pth (training cohort)')
    parser.add_argument('-d', '--test_data_path', required=True, help='internal test .pth')
    parser.add_argument('-e', '--external_data_path', required=True, help='external (MIMIC) test .pth')
    parser.add_argument('-m', '--model_dir', required=True)
    parser.add_argument('-t', '--threshold_path', required=True, help='threshold_tuning_results_*.json')
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()
    run(args.splits_path, args.test_data_path, args.external_data_path, args.model_dir,
        args.threshold_path, args.output_dir)
