import argparse
import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

from prediction.utils.utils import ensure_dir


def paired_bootstrap_comparison(y_true, y_prob_a, y_prob_b,
                                threshold_a=0.5, threshold_b=0.5,
                                n_bootstrap=10000, seed=42):
    """Paired bootstrap test comparing two models on the same test set.

    For each bootstrap iteration, the same resampled indices are used for both
    models, preserving the paired structure.

    Args:
        y_true: true binary labels
        y_prob_a: predicted probabilities from model A
        y_prob_b: predicted probabilities from model B
        threshold_a: decision threshold for model A (for MCC)
        threshold_b: decision threshold for model B (for MCC)
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns:
        dict with per-metric results: point_diff, mean_diff, median_diff,
        std_diff, lower_ci, upper_ci, p_value
    """
    n_samples = len(y_true)
    metrics = ['auroc', 'auprc', 'mcc']

    # Point estimates
    point_a = {
        'auroc': roc_auc_score(y_true, y_prob_a),
        'auprc': average_precision_score(y_true, y_prob_a),
        'mcc': matthews_corrcoef(y_true, (y_prob_a >= threshold_a).astype(int)),
    }
    point_b = {
        'auroc': roc_auc_score(y_true, y_prob_b),
        'auprc': average_precision_score(y_true, y_prob_b),
        'mcc': matthews_corrcoef(y_true, (y_prob_b >= threshold_b).astype(int)),
    }
    point_diff = {m: point_a[m] - point_b[m] for m in metrics}

    # Bootstrap
    diffs = {m: [] for m in metrics}

    for b in tqdm(range(n_bootstrap), desc='Bootstrap comparison'):
        idx = resample(np.arange(n_samples), stratify=y_true, random_state=seed + b)
        y_bs = y_true[idx]
        pa_bs = y_prob_a[idx]
        pb_bs = y_prob_b[idx]

        auroc_a = roc_auc_score(y_bs, pa_bs)
        auroc_b = roc_auc_score(y_bs, pb_bs)
        diffs['auroc'].append(auroc_a - auroc_b)

        auprc_a = average_precision_score(y_bs, pa_bs)
        auprc_b = average_precision_score(y_bs, pb_bs)
        diffs['auprc'].append(auprc_a - auprc_b)

        mcc_a = matthews_corrcoef(y_bs, (pa_bs >= threshold_a).astype(int))
        mcc_b = matthews_corrcoef(y_bs, (pb_bs >= threshold_b).astype(int))
        diffs['mcc'].append(mcc_a - mcc_b)

    # Aggregate
    min_p = 1.0 / n_bootstrap  # minimum resolvable p-value
    results = {}
    for m in metrics:
        d = np.array(diffs[m])
        p_count = np.sum(d <= 0)  # count where A <= B
        p_value = p_count / n_bootstrap
        results[m] = {
            'model_a': float(point_a[m]),
            'model_b': float(point_b[m]),
            'point_diff': float(point_diff[m]),
            'mean_diff': float(np.mean(d)),
            'median_diff': float(np.median(d)),
            'std_diff': float(np.std(d)),
            'lower_ci': float(np.percentile(d, 2.5)),
            'upper_ci': float(np.percentile(d, 97.5)),
            'p_value': float(p_value),
            'p_value_str': f'< {min_p:.1g}' if p_count == 0 else f'{p_value:.4f}',
        }

    return results


def compare_models(pred_path_a, pred_path_b, label_a='Model A', label_b='Model B',
                   threshold_a=0.5, threshold_b=0.5,
                   output_dir=None, n_bootstrap=10000):
    """Compare two models using paired bootstrap tests.

    Args:
        pred_path_a: path to test_predictions.pkl for model A (y_true, y_prob)
        pred_path_b: path to test_predictions.pkl for model B (y_true, y_prob)
        label_a: display name for model A
        label_b: display name for model B
        threshold_a: decision threshold for model A
        threshold_b: decision threshold for model B
        output_dir: directory to save results
        n_bootstrap: number of bootstrap iterations
    """
    # Load predictions
    with open(pred_path_a, 'rb') as f:
        y_true_a, y_prob_a = pickle.load(f)
    with open(pred_path_b, 'rb') as f:
        y_true_b, y_prob_b = pickle.load(f)

    # Verify same test set
    assert len(y_true_a) == len(y_true_b), \
        f"Sample count mismatch: {len(y_true_a)} vs {len(y_true_b)}"
    assert np.array_equal(y_true_a, y_true_b), \
        "Ground truth labels differ — models must be evaluated on the same test set"

    y_true = np.asarray(y_true_a)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    n_samples = len(y_true)
    n_positive = int(y_true.sum())
    print(f"Test set: {n_samples} samples, {n_positive} positive ({n_positive/n_samples*100:.2f}%)")
    print(f"Comparing: {label_a} (threshold={threshold_a}) vs {label_b} (threshold={threshold_b})")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print()

    # Run comparison
    results = paired_bootstrap_comparison(
        y_true, y_prob_a, y_prob_b,
        threshold_a=threshold_a, threshold_b=threshold_b,
        n_bootstrap=n_bootstrap,
    )

    # Print summary
    print(f"{'='*70}")
    print(f"PAIRED BOOTSTRAP COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*70}")
    print(f"{'Metric':>10s}  {label_a:>10s}  {label_b:>10s}  {'Diff':>10s}  {'95% CI':>22s}  {'p-value':>10s}")
    print(f"{'-'*70}")
    for m in ['auroc', 'auprc', 'mcc']:
        r = results[m]
        p_val = r['p_value']
        sig = '*' if p_val < 0.05 else ''
        sig = '**' if p_val < 0.01 else sig
        sig = '***' if p_val < 0.001 or (p_val == 0.0) else sig
        print(f"{m:>10s}  {r['model_a']:10.4f}  {r['model_b']:10.4f}  "
              f"{r['point_diff']:+10.4f}  [{r['lower_ci']:+.4f}, {r['upper_ci']:+.4f}]  "
              f"{r['p_value_str']:>10s} {sig}")
    print()

    # Save outputs
    if output_dir is not None:
        ensure_dir(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON with full results
        output_json = {
            'metadata': {
                'model_a': label_a,
                'model_b': label_b,
                'pred_path_a': pred_path_a,
                'pred_path_b': pred_path_b,
                'threshold_a': threshold_a,
                'threshold_b': threshold_b,
                'n_test_samples': n_samples,
                'n_positive': n_positive,
                'n_bootstrap': n_bootstrap,
                'timestamp': timestamp,
            },
            'results': results,
        }
        json_path = os.path.join(output_dir, 'model_comparison_test.json')
        with open(json_path, 'w') as f:
            json.dump(output_json, f, indent=2)

        # CSV summary
        csv_rows = []
        for m in ['auroc', 'auprc', 'mcc']:
            r = results[m]
            csv_rows.append({
                'metric': m,
                f'{label_a}': round(r['model_a'], 4),
                f'{label_b}': round(r['model_b'], 4),
                'diff': round(r['point_diff'], 4),
                'mean_diff': round(r['mean_diff'], 4),
                'std_diff': round(r['std_diff'], 4),
                'lower_ci': round(r['lower_ci'], 4),
                'upper_ci': round(r['upper_ci'], 4),
                'p_value': r['p_value_str'],
            })
        csv_path = os.path.join(output_dir, 'model_comparison_test.csv')
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

        print(f"Results saved to {output_dir}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paired bootstrap comparison of two models on the same test set')
    parser.add_argument('-a', '--pred_a', type=str, required=True,
                        help='Path to test_predictions.pkl for model A')
    parser.add_argument('-b', '--pred_b', type=str, required=True,
                        help='Path to test_predictions.pkl for model B')
    parser.add_argument('--label_a', type=str, default='Model A',
                        help='Display name for model A')
    parser.add_argument('--label_b', type=str, default='Model B',
                        help='Display name for model B')
    parser.add_argument('--threshold_a', type=float, default=0.5,
                        help='Decision threshold for model A (for MCC)')
    parser.add_argument('--threshold_b', type=float, default=0.5,
                        help='Decision threshold for model B (for MCC)')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('-n', '--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000)')
    cli_args = parser.parse_args()

    compare_models(
        pred_path_a=cli_args.pred_a,
        pred_path_b=cli_args.pred_b,
        label_a=cli_args.label_a,
        label_b=cli_args.label_b,
        threshold_a=cli_args.threshold_a,
        threshold_b=cli_args.threshold_b,
        output_dir=cli_args.output_dir,
        n_bootstrap=cli_args.n_bootstrap,
    )
