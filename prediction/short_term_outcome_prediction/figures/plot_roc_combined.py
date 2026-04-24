"""Combined ROC curves: internal (GSU) and external (MIMIC) on a single plot."""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from plot_config import (
    COLOR_XGB, COLOR_LR,
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure,
)
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple


def _load_model_data(model_dir):
    """Load ROC curve data and test results from a model directory."""
    with open(os.path.join(model_dir, 'roc_curve_data.pkl'), 'rb') as f:
        roc_data = pickle.load(f)
    with open(os.path.join(model_dir, 'test_results.json'), 'r') as f:
        results = json.load(f)
    return roc_data, results


def _get_auroc_str(results):
    """Format AUROC with CI from test_results.json."""
    thresh = results['thresholds']['default_0.5']
    auroc = thresh['point_estimates']['auroc']
    lower = thresh['bootstrap']['auroc']['lower_ci']
    upper = thresh['bootstrap']['auroc']['upper_ci']
    return f'{auroc:.3f} [{lower:.3f}, {upper:.3f}]'


def plot_roc_combined(ax,
                      xgb_int_dir, lr_int_dir,
                      xgb_ext_dir, lr_ext_dir,
                      tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL):
    """Plot ROC curves for both models on both datasets on the same axes.

    Internal (GSU) curves are solid, external (MIMIC) curves are dashed.
    """
    xgb_int_roc, xgb_int_res = _load_model_data(xgb_int_dir)
    lr_int_roc, lr_int_res = _load_model_data(lr_int_dir)
    xgb_ext_roc, xgb_ext_res = _load_model_data(xgb_ext_dir)
    lr_ext_roc, lr_ext_res = _load_model_data(lr_ext_dir)

    # XGBoost – internal (solid)
    ax.plot(xgb_int_roc['fpr_grid'], xgb_int_roc['tpr_point'],
            color=COLOR_XGB, linestyle='-',
            label=f'XGBoost – Internal (AUROC = {_get_auroc_str(xgb_int_res)})')
    ax.fill_between(xgb_int_roc['fpr_grid'],
                    xgb_int_roc['tpr_lower_ci'], xgb_int_roc['tpr_upper_ci'],
                    color=COLOR_XGB, alpha=0.15)

    # XGBoost – external (dashed)
    ax.plot(xgb_ext_roc['fpr_grid'], xgb_ext_roc['tpr_point'],
            color=COLOR_XGB, linestyle='--',
            label=f'XGBoost – External (AUROC = {_get_auroc_str(xgb_ext_res)})')
    ax.fill_between(xgb_ext_roc['fpr_grid'],
                    xgb_ext_roc['tpr_lower_ci'], xgb_ext_roc['tpr_upper_ci'],
                    color=COLOR_XGB, alpha=0.08)

    # LR – internal (solid)
    ax.plot(lr_int_roc['fpr_grid'], lr_int_roc['tpr_point'],
            color=COLOR_LR, linestyle='-',
            label=f'Logistic Regression – Internal (AUROC = {_get_auroc_str(lr_int_res)})')
    ax.fill_between(lr_int_roc['fpr_grid'],
                    lr_int_roc['tpr_lower_ci'], lr_int_roc['tpr_upper_ci'],
                    color=COLOR_LR, alpha=0.15)

    # LR – external (dashed)
    ax.plot(lr_ext_roc['fpr_grid'], lr_ext_roc['tpr_point'],
            color=COLOR_LR, linestyle='--',
            label=f'Logistic Regression – External (AUROC = {_get_auroc_str(lr_ext_res)})')
    ax.fill_between(lr_ext_roc['fpr_grid'],
                    lr_ext_roc['tpr_lower_ci'], lr_ext_roc['tpr_upper_ci'],
                    color=COLOR_LR, alpha=0.08)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', alpha=0.5)

    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=label_size)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Custom legend: model lines + dataset style + CI bands
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Add dataset-style indicators
    legend_handles.append(Line2D([0], [0], color='black', linestyle='-', lw=1.5))
    legend_labels.append('Internal (GSU)')
    legend_handles.append(Line2D([0], [0], color='black', linestyle='--', lw=1.5))
    legend_labels.append('External (MIMIC)')

    # Add CI band patch
    ci_patches = tuple(mpatches.Patch(color=c, alpha=0.15) for c in [COLOR_XGB, COLOR_LR])
    legend_handles.append(ci_patches)
    legend_labels.append('95% CI')

    ax.legend(legend_handles, legend_labels,
              handler_map={tuple: HandlerTuple(ndivide=None)},
              fontsize=tick_size - 1, loc='lower right')
    return ax


def main():
    parser = argparse.ArgumentParser(
        description='Combined ROC curves: internal + external validation')
    parser.add_argument('--xgb_int', required=True,
                        help='XGB internal test results directory')
    parser.add_argument('--lr_int', required=True,
                        help='LR internal test results directory')
    parser.add_argument('--xgb_ext', required=True,
                        help='XGB external test results directory')
    parser.add_argument('--lr_ext', required=True,
                        help='LR external test results directory')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots()
    plot_roc_combined(ax,
                      args.xgb_int, args.lr_int,
                      args.xgb_ext, args.lr_ext)
    save_figure(fig, args.output_dir, 'roc_curve_combined')
    plt.close(fig)


if __name__ == '__main__':
    main()
