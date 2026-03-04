"""ROC curve comparison: XGBoost vs Logistic Regression."""

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
    """Load ROC curve data and test results from a model directory."""
    with open(os.path.join(model_dir, 'roc_curve_data.pkl'), 'rb') as f:
        roc_data = pickle.load(f)
    with open(os.path.join(model_dir, 'test_results.json'), 'r') as f:
        results = json.load(f)
    return roc_data, results


def _get_auroc_str(results):
    """Format AUROC with CI from test_results.json."""
    # Use default_0.5 threshold (AUROC is threshold-independent)
    thresh = results['thresholds']['default_0.5']
    auroc = thresh['point_estimates']['auroc']
    lower = thresh['bootstrap']['auroc']['lower_ci']
    upper = thresh['bootstrap']['auroc']['upper_ci']
    return f'{auroc:.3f} [{lower:.3f}, {upper:.3f}]'


def plot_roc(ax, xgb_dir, lr_dir, tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL):
    """Plot ROC curves for both models on the given axes."""
    xgb_roc, xgb_res = _load_model_data(xgb_dir)
    lr_roc, lr_res = _load_model_data(lr_dir)

    # XGBoost
    ax.plot(xgb_roc['fpr_grid'], xgb_roc['tpr_point'],
            color=COLOR_XGB, label=f'XGBoost (AUROC = {_get_auroc_str(xgb_res)})')
    ax.fill_between(xgb_roc['fpr_grid'], xgb_roc['tpr_lower_ci'], xgb_roc['tpr_upper_ci'],
                    color=COLOR_XGB, alpha=0.2)

    # Logistic Regression
    ax.plot(lr_roc['fpr_grid'], lr_roc['tpr_point'],
            color=COLOR_LR, label=f'Logistic Regression (AUROC = {_get_auroc_str(lr_res)})')
    ax.fill_between(lr_roc['fpr_grid'], lr_roc['tpr_lower_ci'], lr_roc['tpr_upper_ci'],
                    color=COLOR_LR, alpha=0.2)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', alpha=0.5)

    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=label_size)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    make_ci_legend(ax, [COLOR_XGB, COLOR_LR], ['XGBoost', 'Logistic Regression'])
    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot ROC curves for XGB vs LR')
    parser.add_argument('--xgb_dir', required=True, help='Path to XGB test_results directory')
    parser.add_argument('--lr_dir', required=True, help='Path to LR test_results directory')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots()
    plot_roc(ax, args.xgb_dir, args.lr_dir)
    save_figure(fig, args.output_dir, 'roc_curve_comparison')
    plt.close(fig)


if __name__ == '__main__':
    main()
