"""
Alarm burden of saved hourly predictions, over a grid of alarm policies.

    test data (.pth) ──┐
                       ├─> per-patient scores + event times ─> alarm_policy ─> alarm_policy_results.csv
    predictions (.pkl) ┘

Usage:
    1. Select the policy on pooled out-of-fold validation predictions
       (saved by gridsearch_aggregate_xgb.py --threshold_tuning):
        python prediction/short_term_outcome_prediction/testing/evaluate_alarm_policy.py --data_layout cv_splits \
            -d /path/to/train_data_splits.pth -p /path/to/validation_predictions_dir -c /path/to/final_model_config.json \
            -t 0.2 0.24 0.3 -k 1 2 3 -r 0 3 5 -n 0 -s 0.8

    2. Evaluate the single frozen policy on the test set:
        python prediction/short_term_outcome_prediction/testing/evaluate_alarm_policy.py \
            -d /path/to/test_data.pth -p /path/to/test_predictions.pkl -c /path/to/final_model_config.json \
            -t 0.24 -k 2 -r 5
"""
import argparse
import json
import os
import pickle
from enum import Enum

import numpy as np
import torch as ch

from prediction.short_term_outcome_prediction.evaluation.alarm_policy import (
    AlarmTrigger, EventCensoring, evaluate_policy_grid, select_policy, DEFAULT_HORIZON_HOURS, DEFAULT_N_BOOTSTRAP)
from prediction.short_term_outcome_prediction.gridsearch_aggregate_xgb import VALIDATION_PREDICTIONS_FILE_PATTERN
from prediction.utils.utils import ensure_dir

RESULTS_FILE_NAME = 'alarm_policy_results.csv'


class DataLayout(Enum):
    # data: (X, y) tuple; predictions: single .pkl file
    SINGLE_SET = 'single_set'
    # data: list of (X_train, X_val, y_train, y_val) folds; predictions: directory with one .pkl per fold
    CV_SPLITS = 'cv_splits'


def _event_times_per_patient(X, y_df):
    """For every patient (in data order), hours at which an event starts."""
    return [
        y_df[y_df.case_admission_id == case_admission_id].relative_sample_date_hourly_cat.values.astype(int)
        for case_admission_id in X[:, 0, 0, 0]
    ]


def _split_scores_by_patient(flat_scores, event_times_per_patient, n_time_steps, restrict_to_first_event):
    """
    Undo the concatenation of predictions over patients.
    Mirrors aggregate_and_label_timeseries: a patient spans n_time_steps, or stops at the first event if restricted.
    """
    n_hours_per_patient = [
        int(event_times.min()) if (restrict_to_first_event and len(event_times) > 0) else n_time_steps
        for event_times in event_times_per_patient
    ]
    assert sum(n_hours_per_patient) == len(flat_scores), \
        f'Predictions do not match data: got {len(flat_scores)} scores, expected {sum(n_hours_per_patient)}'

    patient_boundaries = np.cumsum(n_hours_per_patient)[:-1]

    return np.split(flat_scores, patient_boundaries)


def _load_patient_set(X, y_df, predictions_path, restrict_to_first_event):
    """
    Returns:
        (scores_per_patient, event_times_per_patient)
    """
    with open(predictions_path, 'rb') as f:
        _, flat_scores = pickle.load(f)

    event_times_per_patient = _event_times_per_patient(X, y_df)
    scores_per_patient = _split_scores_by_patient(flat_scores, event_times_per_patient, n_time_steps=X.shape[1],
                                                  restrict_to_first_event=restrict_to_first_event)

    return scores_per_patient, event_times_per_patient


def _load_pooled_validation_folds(data_splits_path, predictions_dir, restrict_to_first_event):
    """Pool the validation patients of all folds: every patient is scored by a model that has not seen them."""
    scores_per_patient, event_times_per_patient = [], []

    for fold, (_, X_val, _, y_val) in enumerate(ch.load(data_splits_path)):
        fold_predictions_path = os.path.join(predictions_dir, VALIDATION_PREDICTIONS_FILE_PATTERN.format(fold=fold))
        fold_scores, fold_event_times = _load_patient_set(X_val, y_val, fold_predictions_path,
                                                          restrict_to_first_event)
        scores_per_patient += fold_scores
        event_times_per_patient += fold_event_times

    return scores_per_patient, event_times_per_patient


def evaluate_alarm_policy(data_path, predictions_path, model_config_path, thresholds,
                          n_consecutive_hours_options, refractory_hours_options,
                          data_layout=DataLayout.SINGLE_SET,
                          horizon_hours=DEFAULT_HORIZON_HOURS, censoring=EventCensoring.AT_FIRST_EVENT,
                          n_bootstrap=DEFAULT_N_BOOTSTRAP, output_dir=None, trigger=AlarmTrigger.ABOVE_THRESHOLD):
    with open(model_config_path) as f:
        restrict_to_first_event = bool(json.load(f).get('restrict_to_first_event', False))

    if data_layout is DataLayout.CV_SPLITS:
        scores_per_patient, event_times_per_patient = _load_pooled_validation_folds(
            data_path, predictions_path, restrict_to_first_event)
    else:
        X, y_df = ch.load(data_path)
        scores_per_patient, event_times_per_patient = _load_patient_set(X, y_df, predictions_path,
                                                                        restrict_to_first_event)

    policy_grid_df = evaluate_policy_grid(scores_per_patient, event_times_per_patient, thresholds,
                                          n_consecutive_hours_options, refractory_hours_options,
                                          horizon_hours=horizon_hours, censoring=censoring, n_bootstrap=n_bootstrap,
                                          trigger=trigger)

    if output_dir is None:
        output_dir = predictions_path if data_layout is DataLayout.CV_SPLITS else os.path.dirname(predictions_path)
    ensure_dir(output_dir)
    policy_grid_df.to_csv(os.path.join(output_dir, RESULTS_FILE_NAME), index=False)

    return policy_grid_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help='Path to (X, y) .pth file, or to data splits .pth file with --data_layout cv_splits')
    parser.add_argument('-p', '--predictions_path', type=str, required=True,
                        help='Path to test_predictions.pkl (test_xgb.py --final), or with --data_layout cv_splits '
                             'to the directory of validation predictions (gridsearch_aggregate_xgb.py --threshold_tuning)')
    parser.add_argument('--data_layout', type=str, default=DataLayout.SINGLE_SET.value,
                        choices=[data_layout.value for data_layout in DataLayout])
    parser.add_argument('-c', '--model_config_path', type=str, required=True, help='Path to final_model_config.json')
    parser.add_argument('-t', '--thresholds', type=float, nargs='+', required=True)
    parser.add_argument('-k', '--n_consecutive_hours', type=int, nargs='+', default=[1])
    parser.add_argument('-r', '--refractory_hours', type=int, nargs='+', default=[0])
    parser.add_argument('-H', '--horizon_hours', type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument('--censoring', type=str, default=EventCensoring.AT_FIRST_EVENT.value,
                        choices=[censoring.value for censoring in EventCensoring])
    parser.add_argument('--trigger', type=str, default=AlarmTrigger.ABOVE_THRESHOLD.value,
                        choices=[trigger.value for trigger in AlarmTrigger],
                        help='What an alarm requires on top of a score above threshold')
    parser.add_argument('-n', '--n_bootstrap', type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument('-s', '--min_event_sensitivity', type=float, default=None,
                        help='If given, print the policy with least false alarms reaching this sensitivity')
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    cli_args = parser.parse_args()

    policy_grid_df = evaluate_alarm_policy(
        data_path=cli_args.data_path,
        predictions_path=cli_args.predictions_path,
        model_config_path=cli_args.model_config_path,
        thresholds=cli_args.thresholds,
        n_consecutive_hours_options=cli_args.n_consecutive_hours,
        refractory_hours_options=cli_args.refractory_hours,
        data_layout=DataLayout(cli_args.data_layout),
        horizon_hours=cli_args.horizon_hours,
        censoring=EventCensoring(cli_args.censoring),
        n_bootstrap=cli_args.n_bootstrap,
        output_dir=cli_args.output_dir,
        trigger=AlarmTrigger(cli_args.trigger),
    )

    summary_columns = ['threshold', 'n_consecutive_hours', 'refractory_hours', 'event_sensitivity', 'alarm_ppv',
                       'false_alarms_per_true_alarm', 'false_alarms_per_patient_day', 'lead_time_hours_median',
                       'fraction_event_free_patients_never_alarmed']
    print(policy_grid_df[summary_columns].round(3).to_string(index=False))

    if cli_args.min_event_sensitivity is not None:
        print('\nSelected policy:')
        print(select_policy(policy_grid_df, cli_args.min_event_sensitivity)[summary_columns].to_string())
