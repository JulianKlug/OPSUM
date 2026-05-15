"""AUROC over the 72h admission timeline: XGBoost vs Logistic Regression."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from plot_config import (
    COLOR_XGB, COLOR_LR,
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure, make_ci_legend,
)


def _load_timepoint_auroc(model_dir):
    """Load per-timepoint AUROC from per_timepoint_results.csv."""
    df = pd.read_csv(os.path.join(model_dir, 'per_timepoint_results.csv'))
    auroc = df[(df['metric'] == 'auroc') & (df['threshold_name'] == 'default_0.5')].copy()
    auroc = auroc.sort_values('timestep').reset_index(drop=True)
    return auroc


def plot_auroc_time(ax, xgb_dir, lr_dir, tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL):
    """Plot AUROC over timesteps for both models on the given axes."""
    xgb_auroc = _load_timepoint_auroc(xgb_dir)
    lr_auroc = _load_timepoint_auroc(lr_dir)

    # XGBoost
    ax.plot(xgb_auroc['timestep'], xgb_auroc['point_estimate'],
            color=COLOR_XGB, label='XGBoost')
    ax.fill_between(xgb_auroc['timestep'], xgb_auroc['lower_ci'], xgb_auroc['upper_ci'],
                    color=COLOR_XGB, alpha=0.2)

    # Logistic Regression
    ax.plot(lr_auroc['timestep'], lr_auroc['point_estimate'],
            color=COLOR_LR, label='Logistic Regression')
    ax.fill_between(lr_auroc['timestep'], lr_auroc['lower_ci'], lr_auroc['upper_ci'],
                    color=COLOR_LR, alpha=0.2)

    ax.set_xlabel('Time after admission (hours)', fontsize=label_size)
    ax.set_ylabel('AUROC', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.set_ylim([0.5, 1.0])

    make_ci_legend(ax, [COLOR_XGB, COLOR_LR], ['XGBoost', 'Logistic Regression'])
    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot AUROC over time for XGB vs LR')
    parser.add_argument('--xgb_dir', required=True, help='Path to XGB test_results directory')
    parser.add_argument('--lr_dir', required=True, help='Path to LR test_results directory')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots()
    plot_auroc_time(ax, args.xgb_dir, args.lr_dir)
    save_figure(fig, args.output_dir, 'auroc_over_time')
    plt.close(fig)


if __name__ == '__main__':
    main()
