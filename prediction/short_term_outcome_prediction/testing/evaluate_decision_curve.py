"""
Decision curve analysis of hourly END predictions in the test set.

    derivation out-of-fold predictions ──fit──> recalibration (a, b)
    test predictions ──apply (a, b)──> recalibrated ──> net benefit per threshold probability (+ bootstrap CI)

Outputs:
    decision_curve_results.csv   one row per (score scale, threshold probability)
    recalibration.json           a, b, calibration before/after, recalibrated value of the operating threshold
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from prediction.short_term_outcome_prediction.evaluation.decision_curve import (
    ScoreScale, fit_recalibration, calibration_summary, decision_curve, bootstrap_decision_curve,
    DEFAULT_N_BOOTSTRAP)

# Each test patient contributes one prediction per hour of the first 72 h
N_TIME_STEPS = 72

# Operating threshold on the raw score used in the paper
OPERATING_THRESHOLD = 0.24

# 0.5% to 20% in steps of 0.5%
THRESHOLD_PROBABILITIES = np.round(np.arange(0.005, 0.2001, 0.005), 3)

OOF_PREDICTIONS_GLOB = 'validation_predictions_fold_*.pkl'
RESULTS_FILE = 'decision_curve_results.csv'
RECALIBRATION_FILE = 'recalibration.json'


def _load_pooled_oof_predictions(oof_dir):
    fold_paths = sorted(glob.glob(os.path.join(oof_dir, OOF_PREDICTIONS_GLOB)))
    if not fold_paths:
        raise FileNotFoundError(f'No {OOF_PREDICTIONS_GLOB} in {oof_dir}')

    folds = [pickle.load(open(path, 'rb')) for path in fold_paths]
    y_true = np.concatenate([np.asarray(y) for y, _ in folds])
    y_prob = np.concatenate([np.asarray(p) for _, p in folds])

    return y_true, y_prob


def _load_test_predictions_per_patient(predictions_path, n_time_steps):
    """(n_patients, n_hours) arrays: flat predictions are ordered patient by patient"""
    y_true, y_prob = pickle.load(open(predictions_path, 'rb'))
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(y_true) % n_time_steps:
        raise ValueError(f'{len(y_true)} predictions are not a multiple of {n_time_steps} hours')

    return y_true.reshape(-1, n_time_steps), y_prob.reshape(-1, n_time_steps)


def _curve_with_ci(y_true_per_patient, y_prob_per_patient, threshold_probabilities, score_scale, n_bootstrap):
    curve_df = decision_curve(y_true_per_patient.ravel(), y_prob_per_patient.ravel(), threshold_probabilities)
    ci_df = bootstrap_decision_curve(y_true_per_patient, y_prob_per_patient, threshold_probabilities, n_bootstrap)

    curve_df = curve_df.merge(ci_df, on='threshold_probability')
    curve_df.insert(0, 'score_scale', score_scale.value)

    return curve_df


def evaluate_decision_curve(oof_dir, test_predictions_path, output_dir, n_time_steps=N_TIME_STEPS,
                            n_bootstrap=DEFAULT_N_BOOTSTRAP):
    y_oof, p_oof = _load_pooled_oof_predictions(oof_dir)
    y_test, p_test_raw = _load_test_predictions_per_patient(test_predictions_path, n_time_steps)

    recalibration = fit_recalibration(y_oof, p_oof)
    p_test_recalibrated = recalibration.apply(p_test_raw)

    # Operating threshold mapped to the probability scale; added to the grid so its net benefit is reported
    operating_threshold_probability = float(recalibration.apply(OPERATING_THRESHOLD))
    recalibrated_grid = np.sort(np.append(THRESHOLD_PROBABILITIES, operating_threshold_probability))

    results_df = pd.concat([
        _curve_with_ci(y_test, p_test_recalibrated, recalibrated_grid, ScoreScale.RECALIBRATED, n_bootstrap),
        _curve_with_ci(y_test, p_test_raw, THRESHOLD_PROBABILITIES, ScoreScale.RAW, n_bootstrap),
    ])

    recalibration_info = {
        'intercept': recalibration.intercept,
        'slope': recalibration.slope,
        'operating_threshold_raw': OPERATING_THRESHOLD,
        'operating_threshold_probability': operating_threshold_probability,
        'n_oof_predictions': int(len(y_oof)),
        'n_test_patients': int(len(y_test)),
        'n_test_predictions': int(y_test.size),
        'calibration': {
            'oof_raw': calibration_summary(y_oof, p_oof),
            'test_raw': calibration_summary(y_test.ravel(), p_test_raw.ravel()),
            'test_recalibrated': calibration_summary(y_test.ravel(), p_test_recalibrated.ravel()),
        },
        'n_bootstrap': n_bootstrap,
    }

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, RESULTS_FILE), index=False)
    with open(os.path.join(output_dir, RECALIBRATION_FILE), 'w') as f:
        json.dump(recalibration_info, f, indent=2)

    return results_df, recalibration_info


def main():
    parser = argparse.ArgumentParser(description='Decision curve analysis of hourly END predictions')
    parser.add_argument('-v', '--oof_dir', required=True,
                        help='Directory with validation_predictions_fold_*.pkl (fold models trained as final model)')
    parser.add_argument('-p', '--test_predictions', required=True, help='test_predictions.pkl')
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-n', '--n_bootstrap', type=int, default=DEFAULT_N_BOOTSTRAP)
    args = parser.parse_args()

    _, recalibration_info = evaluate_decision_curve(args.oof_dir, args.test_predictions, args.output_dir,
                                                    n_bootstrap=args.n_bootstrap)
    print(json.dumps(recalibration_info, indent=2))


if __name__ == '__main__':
    main()
