import argparse
import os
import json
import pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch as ch
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, accuracy_score,
                             average_precision_score, fbeta_score, roc_curve,
                             precision_recall_curve)

from prediction.short_term_outcome_prediction.logistic_regression.evaluate_logistic_regression import raw_features_and_labels
from prediction.utils.utils import ensure_dir


def _compute_metrics(y_true, y_prob, threshold, auroc=None, auprc=None):
    """Compute all metrics at a given threshold.

    Args:
        y_true: true binary labels
        y_prob: predicted probabilities
        threshold: decision threshold
        auroc: precomputed AUROC (optional, avoids redundant computation)
        auprc: precomputed AUPRC (optional, avoids redundant computation)
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + tn + fp + fn

    if auroc is None:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc = float('nan')
    if auprc is None:
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            auprc = float('nan')

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    youden_j = rec + spec - 1

    return {
        'auroc': float(auroc), 'auprc': float(auprc),
        'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
        'specificity': float(spec), 'mcc': float(mcc), 'f1': float(f1), 'f2': float(f2),
        'npv': float(npv), 'youden_j': float(youden_j),
    }


def test_final_model(test_data_path, model_dir, threshold_results_path=None,
                     output_dir=None, n_bootstrap=1000):
    """Test the final logistic regression model with bootstrap confidence intervals.

    Args:
        test_data_path: path to test data .pth file (X_test, y_test tuple)
        model_dir: directory containing lr_final_model.pkl, scaler.pkl, final_model_config.json
        threshold_results_path: path to threshold tuning results JSON (optional)
        output_dir: directory to save results (default: model_dir/test_results)
        n_bootstrap: number of bootstrap iterations
    """
    # 1. Load model artifacts
    print("Loading model artifacts...")
    with open(os.path.join(model_dir, 'lr_final_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, 'final_model_config.json')) as f:
        config = json.load(f)

    n_features_expected = config.get('n_features')

    # 2. Load thresholds
    thresholds = {'default_0.5': 0.5}
    if threshold_results_path is not None:
        with open(threshold_results_path) as f:
            threshold_data = json.load(f)
        median_thresholds = threshold_data.get('median_thresholds', {})
        for metric_name, info in median_thresholds.items():
            thresholds[metric_name] = info['median_threshold']
    print(f"Thresholds: {thresholds}")

    # 3. Prepare test data (raw features, no temporal aggregation)
    print("Loading and preparing test data...")
    X_test_raw, y_test_raw = ch.load(test_data_path)
    n_patients = X_test_raw.shape[0]
    n_time_steps = X_test_raw.shape[1]

    test_data_list, test_labels_list = raw_features_and_labels(
        X_test_raw, y_test_raw,
        target_time_to_outcome=6,
        target_interval=True,
    )

    # Build timestep index before concatenation
    timestep_index = np.concatenate([np.arange(len(labels)) for labels in test_labels_list])

    test_data = np.concatenate(test_data_list)
    y_test = np.concatenate(test_labels_list)

    X_test_scaled = scaler.transform(test_data)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    n_features = X_test_scaled.shape[1]
    if n_features_expected is not None:
        assert n_features == n_features_expected, \
            f"Feature count mismatch: got {n_features}, expected {n_features_expected}"

    n_samples = len(y_test)
    n_positive = int(y_test.sum())
    print(f"Test data: {n_samples} samples, {n_positive} positive ({n_positive/n_samples*100:.2f}%)")
    print(f"Features: {n_features}")

    # 4. Compute predictions
    print("Computing predictions...")
    y_prob = model.predict_proba(X_test_scaled)[:, 1].astype('float32')

    # 5. Point estimates and curve data
    print("Computing point estimates...")
    point_estimates = {}
    for thresh_name, thresh_val in thresholds.items():
        point_estimates[thresh_name] = _compute_metrics(y_test, y_prob, thresh_val)

    # ROC curve (point estimate)
    fpr_raw, tpr_raw, roc_thresholds_raw = roc_curve(y_test, y_prob)
    # PR curve (point estimate)
    prec_raw, rec_raw, pr_thresholds_raw = precision_recall_curve(y_test, y_prob)

    # Grids for interpolation
    fpr_grid = np.linspace(0, 1, 200)
    recall_grid = np.linspace(0, 1, 200)

    tpr_point_interp = np.interp(fpr_grid, fpr_raw, tpr_raw)
    prec_point_interp = np.interp(recall_grid, rec_raw[::-1], prec_raw[::-1])

    # Per-timepoint point estimates
    unique_timesteps = np.unique(timestep_index).astype(int)
    per_timepoint_point = {}
    for t in unique_timesteps:
        mask = timestep_index == t
        y_t = y_test[mask]
        p_t = y_prob[mask]
        if len(y_t) < 2 or y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        auroc_t = float(roc_auc_score(y_t, p_t))
        auprc_t = float(average_precision_score(y_t, p_t))
        per_timepoint_point[int(t)] = {}
        for thresh_name, thresh_val in thresholds.items():
            per_timepoint_point[int(t)][thresh_name] = _compute_metrics(
                y_t, p_t, thresh_val, auroc=auroc_t, auprc=auprc_t)

    # 6. Stratified bootstrap
    print(f"Running {n_bootstrap} bootstrap iterations...")
    bs_results = {thresh_name: [] for thresh_name in thresholds}
    bs_tpr = []
    bs_prec = []
    per_timepoint_bs = defaultdict(list)

    for b in tqdm(range(n_bootstrap)):
        idx = resample(np.arange(n_samples), stratify=y_test, random_state=b)
        y_bs = y_test[idx]
        p_bs = y_prob[idx]

        # Overall metrics at each threshold
        auroc_bs = float(roc_auc_score(y_bs, p_bs))
        auprc_bs = float(average_precision_score(y_bs, p_bs))
        for thresh_name, thresh_val in thresholds.items():
            m = _compute_metrics(y_bs, p_bs, thresh_val, auroc=auroc_bs, auprc=auprc_bs)
            bs_results[thresh_name].append(m)

        # ROC curve
        fpr_bs, tpr_bs, _ = roc_curve(y_bs, p_bs)
        bs_tpr.append(np.interp(fpr_grid, fpr_bs, tpr_bs))

        # PR curve
        prec_bs, rec_bs, _ = precision_recall_curve(y_bs, p_bs)
        bs_prec.append(np.interp(recall_grid, rec_bs[::-1], prec_bs[::-1]))

        # Per-timepoint
        ts_for_idx = timestep_index[idx]
        for t in unique_timesteps:
            ts_mask = ts_for_idx == t
            if ts_mask.sum() < 2:
                continue
            y_t_bs = y_bs[ts_mask]
            p_t_bs = p_bs[ts_mask]
            if y_t_bs.sum() == 0 or y_t_bs.sum() == len(y_t_bs):
                continue
            try:
                auroc_t_bs = float(roc_auc_score(y_t_bs, p_t_bs))
                auprc_t_bs = float(average_precision_score(y_t_bs, p_t_bs))
            except ValueError:
                continue
            for thresh_name, thresh_val in thresholds.items():
                m = _compute_metrics(y_t_bs, p_t_bs, thresh_val, auroc=auroc_t_bs, auprc=auprc_t_bs)
                per_timepoint_bs[(int(t), thresh_name)].append(m)

    # 7. Aggregate results
    print("Aggregating results...")
    metric_names = list(point_estimates[list(thresholds.keys())[0]].keys())

    # Overall bootstrap stats
    bootstrap_stats = {}
    for thresh_name in thresholds:
        bootstrap_stats[thresh_name] = {}
        for metric in metric_names:
            values = np.array([m[metric] for m in bs_results[thresh_name]])
            values = values[~np.isnan(values)]
            bootstrap_stats[thresh_name][metric] = {
                'median': float(np.percentile(values, 50)),
                'std': float(np.std(values)),
                'lower_ci': float(np.percentile(values, 2.5)),
                'upper_ci': float(np.percentile(values, 97.5)),
            }

    # ROC curve bands
    bs_tpr_arr = np.array(bs_tpr)
    roc_curve_data = {
        'fpr_grid': fpr_grid,
        'tpr_point': tpr_point_interp,
        'tpr_mean': np.mean(bs_tpr_arr, axis=0),
        'tpr_lower_ci': np.percentile(bs_tpr_arr, 2.5, axis=0),
        'tpr_upper_ci': np.percentile(bs_tpr_arr, 97.5, axis=0),
        'fpr': fpr_raw,
        'tpr': tpr_raw,
        'thresholds': roc_thresholds_raw,
    }

    # PR curve bands
    bs_prec_arr = np.array(bs_prec)
    pr_curve_data = {
        'recall_grid': recall_grid,
        'precision_point': prec_point_interp,
        'precision_mean': np.mean(bs_prec_arr, axis=0),
        'precision_lower_ci': np.percentile(bs_prec_arr, 2.5, axis=0),
        'precision_upper_ci': np.percentile(bs_prec_arr, 97.5, axis=0),
        'precision': prec_raw,
        'recall': rec_raw,
        'thresholds': pr_thresholds_raw,
    }

    # Per-timepoint aggregation
    per_timepoint_rows = []
    for t in unique_timesteps:
        for thresh_name in thresholds:
            if int(t) not in per_timepoint_point:
                continue
            pt_point = per_timepoint_point[int(t)].get(thresh_name, {})
            bs_list = per_timepoint_bs.get((int(t), thresh_name), [])
            for metric in metric_names:
                row = {
                    'timestep': int(t),
                    'threshold_name': thresh_name,
                    'threshold_value': thresholds[thresh_name],
                    'metric': metric,
                    'point_estimate': pt_point.get(metric, float('nan')),
                }
                if bs_list:
                    values = np.array([m[metric] for m in bs_list])
                    values = values[~np.isnan(values)]
                    if len(values) > 0:
                        row['median'] = float(np.percentile(values, 50))
                        row['std'] = float(np.std(values))
                        row['lower_ci'] = float(np.percentile(values, 2.5))
                        row['upper_ci'] = float(np.percentile(values, 97.5))
                    else:
                        row['median'] = row['std'] = row['lower_ci'] = row['upper_ci'] = float('nan')
                else:
                    row['median'] = row['std'] = row['lower_ci'] = row['upper_ci'] = float('nan')
                per_timepoint_rows.append(row)

    # 8. Save outputs
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'test_results')
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # test_results.json
    results_json = {
        'metadata': {
            'model': 'LogisticRegression',
            'model_dir': model_dir,
            'test_data_path': test_data_path,
            'threshold_results_path': threshold_results_path,
            'n_test_samples': n_samples,
            'n_positive': n_positive,
            'positive_rate': n_positive / n_samples,
            'n_features': n_features,
            'feature_type': config.get('feature_type', 'raw_values_only'),
            'n_bootstrap': n_bootstrap,
            'n_patients': int(n_patients),
            'n_time_steps': int(n_time_steps),
            'timestamp': timestamp,
        },
        'thresholds': {},
    }
    for thresh_name, thresh_val in thresholds.items():
        results_json['thresholds'][thresh_name] = {
            'threshold_value': thresh_val,
            'point_estimates': point_estimates[thresh_name],
            'bootstrap': bootstrap_stats[thresh_name],
        }

    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # test_results.csv
    csv_rows = []
    for thresh_name, thresh_val in thresholds.items():
        for metric in metric_names:
            csv_rows.append({
                'threshold_name': thresh_name,
                'threshold_value': thresh_val,
                'metric': metric,
                'point_estimate': point_estimates[thresh_name][metric],
                'median': bootstrap_stats[thresh_name][metric]['median'],
                'std': bootstrap_stats[thresh_name][metric]['std'],
                'lower_ci': bootstrap_stats[thresh_name][metric]['lower_ci'],
                'upper_ci': bootstrap_stats[thresh_name][metric]['upper_ci'],
            })
    pd.DataFrame(csv_rows).to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    # test_predictions.pkl
    with open(os.path.join(output_dir, 'test_predictions.pkl'), 'wb') as f:
        pickle.dump((y_test, y_prob), f)

    # roc_curve_data.pkl
    with open(os.path.join(output_dir, 'roc_curve_data.pkl'), 'wb') as f:
        pickle.dump(roc_curve_data, f)

    # pr_curve_data.pkl
    with open(os.path.join(output_dir, 'pr_curve_data.pkl'), 'wb') as f:
        pickle.dump(pr_curve_data, f)

    # per_timepoint_results.csv
    if per_timepoint_rows:
        pd.DataFrame(per_timepoint_rows).to_csv(
            os.path.join(output_dir, 'per_timepoint_results.csv'), index=False)

    # 9. Print summary
    print(f"\n{'='*70}")
    print(f"LOGISTIC REGRESSION TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Samples: {n_samples} ({n_positive} positive, {n_positive/n_samples*100:.2f}%)")
    print(f"Features: {n_features} (raw values only)")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print()

    for thresh_name, thresh_val in thresholds.items():
        print(f"--- Threshold: {thresh_name} = {thresh_val:.4f} ---")
        pe = point_estimates[thresh_name]
        bs = bootstrap_stats[thresh_name]
        for metric in metric_names:
            print(f"  {metric:>15s}: {pe[metric]:.4f}  "
                  f"[{bs[metric]['lower_ci']:.4f}, {bs[metric]['upper_ci']:.4f}]")
        print()

    print(f"Results saved to {output_dir}")
    return results_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test final logistic regression model with bootstrap CIs')
    parser.add_argument('-d', '--test_data_path', type=str, required=True,
                        help='Path to test data .pth file')
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help='Directory containing lr_final_model.pkl, scaler.pkl, final_model_config.json')
    parser.add_argument('-t', '--threshold_results_path', type=str, default=None,
                        help='Path to threshold tuning results JSON (optional)')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory (default: model_dir/test_results)')
    parser.add_argument('-n', '--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    cli_args = parser.parse_args()

    test_final_model(
        test_data_path=cli_args.test_data_path,
        model_dir=cli_args.model_dir,
        threshold_results_path=cli_args.threshold_results_path,
        output_dir=cli_args.output_dir,
        n_bootstrap=cli_args.n_bootstrap,
    )
