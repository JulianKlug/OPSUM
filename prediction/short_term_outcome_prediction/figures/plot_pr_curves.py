"""Precision-Recall curve comparison: XGBoost vs Logistic Regression."""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from plot_config import (
    COLOR_XGB, COLOR_LR,
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure, make_ci_legend,
)


def _load_model_data(model_dir):
    """Load PR curve data and test results from a model directory."""
    with open(os.path.join(model_dir, 'pr_curve_data.pkl'), 'rb') as f:
        pr_data = pickle.load(f)
    with open(os.path.join(model_dir, 'test_results.json'), 'r') as f:
        results = json.load(f)
    return pr_data, results


def _get_auprc_str(results):
    """Format AUPRC with CI from test_results.json."""
    thresh = results['thresholds']['default_0.5']
    auprc = thresh['point_estimates']['auprc']
    lower = thresh['bootstrap']['auprc']['lower_ci']
    upper = thresh['bootstrap']['auprc']['upper_ci']
    return f'{auprc:.3f} [{lower:.3f}, {upper:.3f}]'


def _get_prevalence(results):
    """Get the positive rate (prevalence) from metadata."""
    return results['metadata']['positive_rate']


def plot_pr(ax, xgb_dir, lr_dir, tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL):
    """Plot PR curves for both models on the given axes."""
    xgb_pr, xgb_res = _load_model_data(xgb_dir)
    lr_pr, lr_res = _load_model_data(lr_dir)

    prevalence = _get_prevalence(xgb_res)

    # XGBoost
    ax.plot(xgb_pr['recall_grid'], xgb_pr['precision_point'],
            color=COLOR_XGB, label=f'XGBoost (AUPRC = {_get_auprc_str(xgb_res)})')
    ax.fill_between(xgb_pr['recall_grid'], xgb_pr['precision_lower_ci'], xgb_pr['precision_upper_ci'],
                    color=COLOR_XGB, alpha=0.2)

    # Logistic Regression
    ax.plot(lr_pr['recall_grid'], lr_pr['precision_point'],
            color=COLOR_LR, label=f'Logistic Regression (AUPRC = {_get_auprc_str(lr_res)})')
    ax.fill_between(lr_pr['recall_grid'], lr_pr['precision_lower_ci'], lr_pr['precision_upper_ci'],
                    color=COLOR_LR, alpha=0.2)

    # No-skill baseline at prevalence
    ax.axhline(y=prevalence, color='grey', lw=1, linestyle='--', alpha=0.5,
               label=f'Prevalence ({prevalence:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', fontsize=label_size)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    make_ci_legend(ax, [COLOR_XGB, COLOR_LR], ['XGBoost', 'Logistic Regression'])
    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot PR curves for XGB vs LR')
    parser.add_argument('--xgb_dir', required=True, help='Path to XGB test_results directory')
    parser.add_argument('--lr_dir', required=True, help='Path to LR test_results directory')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots()
    plot_pr(ax, args.xgb_dir, args.lr_dir)
    save_figure(fig, args.output_dir, 'pr_curve_comparison')
    plt.close(fig)


if __name__ == '__main__':
    main()
