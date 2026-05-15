"""Calibration plot: XGBoost vs Logistic Regression."""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from plot_config import (
    COLOR_XGB, COLOR_LR,
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure,
)


def _load_predictions(model_dir):
    """Load (y_true, y_prob) tuple from test_predictions.pkl."""
    with open(os.path.join(model_dir, 'test_predictions.pkl'), 'rb') as f:
        y_true, y_prob = pickle.load(f)
    return y_true, y_prob


def plot_calibration(ax, xgb_dir, lr_dir, n_bins=10,
                     tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL):
    """Plot calibration curves for both models on the given axes."""
    y_true_xgb, y_prob_xgb = _load_predictions(xgb_dir)
    y_true_lr, y_prob_lr = _load_predictions(lr_dir)

    # Quantile strategy: better for imbalanced data (1.97% positive rate)
    prob_true_xgb, prob_pred_xgb = calibration_curve(
        y_true_xgb, y_prob_xgb, n_bins=n_bins, strategy='quantile')
    prob_true_lr, prob_pred_lr = calibration_curve(
        y_true_lr, y_prob_lr, n_bins=n_bins, strategy='quantile')

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], ':', color='grey', lw=1, alpha=0.5, label='Perfect calibration')

    # XGBoost
    ax.plot(prob_pred_xgb, prob_true_xgb, marker='o', color=COLOR_XGB, label='XGBoost')

    # Logistic Regression
    ax.plot(prob_pred_lr, prob_true_lr, marker='o', color=COLOR_LR, label='Logistic Regression')

    ax.set_xlabel('Predicted Probability', fontsize=label_size)
    ax.set_ylabel('Observed Frequency', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend()
    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot calibration curves for XGB vs LR')
    parser.add_argument('--xgb_dir', required=True, help='Path to XGB test_results directory')
    parser.add_argument('--lr_dir', required=True, help='Path to LR test_results directory')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots()
    plot_calibration(ax, args.xgb_dir, args.lr_dir)
    save_figure(fig, args.output_dir, 'calibration_comparison')
    plt.close(fig)


if __name__ == '__main__':
    main()
