"""Supplementary figure: decision curve analysis.

    A. Net benefit vs threshold probability: recalibrated XGBoost, treat all, treat none.
    B. Calibration of hourly predictions in the test set, before and after recalibration.
"""

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve

from plot_config import (
    COLOR_XGB,
    PUB_TICK, PUB_LABEL, PUB_SUBPLOT_NUM,
    setup_theme, save_figure, CM,
)
from prediction.short_term_outcome_prediction.evaluation.decision_curve import ScoreScale, Recalibration

# Pre-specified clinically relevant range of threshold probabilities
RELEVANT_RANGE = (0.02, 0.10)

# Panel A y-axis: treat all falls steeply below 0; the region of interest is between 0 and prevalence
NET_BENEFIT_Y_LIMITS = (-0.004, 0.021)

# Panel B: quantile bins, log axes (most predictions are below 10%)
N_CALIBRATION_BINS = 10
CALIBRATION_AXIS_LIMITS = (0.0005, 1)
CALIBRATION_TICKS = [0.001, 0.01, 0.1, 1]

COLOR_RAW = '#7F96CB'
COLOR_TREAT_ALL = 'grey'
COLOR_TREAT_NONE = 'black'
COLOR_RELEVANT_RANGE = '#EFEFEF'
CI_ALPHA = 0.2
LINE_WIDTH = 1.2
REFERENCE_LINE_WIDTH = 0.8
STEM = 'decision_curve'


def _style_axes(ax):
    ax.tick_params('both', labelsize=PUB_TICK)
    ax.xaxis.label.set_size(PUB_LABEL)
    ax.yaxis.label.set_size(PUB_LABEL)


def plot_net_benefit(ax, results_df, operating_threshold_probability):
    curve_df = results_df[results_df.score_scale == ScoreScale.RECALIBRATED.value]
    threshold_pct = curve_df.threshold_probability * 100

    ax.axvspan(RELEVANT_RANGE[0] * 100, RELEVANT_RANGE[1] * 100, color=COLOR_RELEVANT_RANGE, zorder=0,
               label='Pre-specified range')
    ax.axhline(0, color=COLOR_TREAT_NONE, ls=':', lw=REFERENCE_LINE_WIDTH, label='Treat none')
    ax.plot(threshold_pct, curve_df.treat_all_net_benefit, color=COLOR_TREAT_ALL, lw=REFERENCE_LINE_WIDTH,
            label='Treat all')

    ax.fill_between(threshold_pct, curve_df.net_benefit_lower_ci, curve_df.net_benefit_upper_ci,
                    color=COLOR_XGB, alpha=CI_ALPHA, lw=0)
    ax.plot(threshold_pct, curve_df.net_benefit, color=COLOR_XGB, lw=LINE_WIDTH, label='XGBoost (recalibrated)')

    # Operating threshold of the paper (raw score 0.24) on the probability scale
    operating_row = curve_df.iloc[(curve_df.threshold_probability - operating_threshold_probability).abs().argmin()]
    ax.scatter(operating_row.threshold_probability * 100, operating_row.net_benefit, color=COLOR_XGB,
               edgecolor='white', s=30, zorder=5, label='Operating threshold')

    ax.set_xlim(0, curve_df.threshold_probability.max() * 100)
    ax.set_ylim(*NET_BENEFIT_Y_LIMITS)
    ax.set_xlabel('Threshold probability (%)')
    ax.set_ylabel('Net benefit')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=COLOR_XGB, alpha=CI_ALPHA))
    labels.append('95% CI')
    ax.legend(handles, labels, fontsize=PUB_TICK, frameon=False, loc='upper right')
    _style_axes(ax)


def plot_calibration(ax, y_true, y_prob_raw, recalibration):
    ax.plot(CALIBRATION_AXIS_LIMITS, CALIBRATION_AXIS_LIMITS, ':', color='grey', lw=REFERENCE_LINE_WIDTH,
            label='Perfect calibration')

    for y_prob, color, label in [(y_prob_raw, COLOR_RAW, 'Raw score'),
                                 (recalibration.apply(y_prob_raw), COLOR_XGB, 'Recalibrated')]:
        observed, predicted = calibration_curve(y_true, y_prob, n_bins=N_CALIBRATION_BINS, strategy='quantile')
        ax.plot(predicted, observed, marker='o', ms=3, color=color, lw=LINE_WIDTH, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*CALIBRATION_AXIS_LIMITS)
    ax.set_ylim(*CALIBRATION_AXIS_LIMITS)
    ax.set_aspect('equal')

    # Percent labels: 0.001 -> 0.1%
    tick_labels = [f'{tick * 100:g}%' for tick in CALIBRATION_TICKS]
    ax.set_xticks(CALIBRATION_TICKS)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(CALIBRATION_TICKS)
    ax.set_yticklabels(tick_labels)
    ax.minorticks_off()

    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.legend(fontsize=PUB_TICK, frameon=False, loc='upper left')
    _style_axes(ax)


def main():
    parser = argparse.ArgumentParser(description='Supplementary figure: decision curve analysis')
    parser.add_argument('--results', required=True, help='decision_curve_results.csv')
    parser.add_argument('--recalibration', required=True, help='recalibration.json')
    parser.add_argument('--test_predictions', required=True, help='test_predictions.pkl')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    results_df = pd.read_csv(args.results)
    with open(args.recalibration) as f:
        recalibration_info = json.load(f)
    recalibration = Recalibration(intercept=recalibration_info['intercept'], slope=recalibration_info['slope'])
    with open(args.test_predictions, 'rb') as f:
        y_true, y_prob_raw = (np.asarray(a) for a in pickle.load(f))

    setup_theme()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18 * CM, 8 * CM), gridspec_kw={'width_ratios': [1.6, 1]})
    plot_net_benefit(ax_a, results_df, recalibration_info['operating_threshold_probability'])
    plot_calibration(ax_b, y_true, y_prob_raw, recalibration)

    fig.tight_layout()

    # Panel labels on one line: panel B is shorter (square aspect), so align on the top of panel A
    label_y = ax_a.get_position().y1 + 0.06
    for label, ax in [('A.', ax_a), ('B.', ax_b)]:
        fig.text(ax.get_position().x0 - 0.06, label_y, label, fontsize=PUB_SUBPLOT_NUM, fontweight='bold', va='top')

    save_figure(fig, args.output_dir, STEM)
    plt.close(fig)


if __name__ == '__main__':
    main()
