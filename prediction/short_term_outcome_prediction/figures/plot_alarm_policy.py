"""Supplementary figure: alarm policy.

    A. Event-level sensitivity vs false alarms per patient-day across thresholds, one line per refractory period
       (derivation cohort), with the policies at the reported threshold evaluated in the test set.
    B. Worked example: risk score of a patient with and a patient without END,
       hourly alarms vs alarms under the alarm policy.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as ch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_config import (
    COLOR_XGB,
    PUB_TICK, PUB_LABEL, PUB_SUBPLOT_NUM,
    setup_theme, save_figure, CM,
)
from prediction.short_term_outcome_prediction.evaluation.alarm_policy import AlarmPolicy, compute_alarm_times
from prediction.short_term_outcome_prediction.testing.evaluate_alarm_policy import _load_patient_set

THRESHOLD = 0.24
HOURLY_POLICY = AlarmPolicy(threshold=THRESHOLD, n_consecutive_hours=1, refractory_hours=0)
ALARM_POLICY = AlarmPolicy(threshold=THRESHOLD, n_consecutive_hours=1, refractory_hours=3)

# Panel A: one line per refractory period, at a persistence of 1 h
N_CONSECUTIVE_HOURS_SHOWN = 1
REFRACTORY_HOURS_SHOWN = [0, 1, 3, 5]
FALSE_ALARM_RATE_TICKS = [0.5, 1, 2, 5, 10]

# Panel B: an example patient with END has its event within this range, to leave room for alarms before it
EXAMPLE_EVENT_HOUR_RANGE = (24, 60)
EXAMPLE_N_FALSE_ALARMS_RANGE = (1, 4)
EXAMPLE_N_ALARMS_WITHOUT_END_RANGE = (2, 3)

COLOR_FALSE_ALARM = '#E74C3C'
COLOR_TEST_SET = 'black'
COLOR_REFERENCE = 'grey'
COLOR_REFRACTORY = '#D9D9D9'
COLOR_DETECTION_WINDOW = '#EFEFEF'

LINE_WIDTH = 1.2
REFERENCE_LINE_WIDTH = 0.8
ALARM_MARKER_SIZE = 60
STEM = 'alarm_policy'


def _policy_row(policy_grid_df, policy):
    is_policy = (np.isclose(policy_grid_df.threshold, policy.threshold)
                 & (policy_grid_df.n_consecutive_hours == policy.n_consecutive_hours)
                 & (policy_grid_df.refractory_hours == policy.refractory_hours))
    return policy_grid_df[is_policy].iloc[0]


def _ci_error(policy_row, metric):
    """Asymmetric error bar lengths, as expected by ax.errorbar."""
    return [[policy_row[metric] - policy_row[f'{metric}_lower_ci']],
            [policy_row[f'{metric}_upper_ci'] - policy_row[metric]]]


def plot_sensitivity_vs_alarm_burden(ax, validation_grid_df, test_grid_df,
                                     tick_size=PUB_TICK, label_size=PUB_LABEL):
    # Lighter to darker as the refractory period lengthens; skip the lightest tint, illegible on white
    colors = sns.light_palette(COLOR_XGB, n_colors=len(REFRACTORY_HOURS_SHOWN) + 1)[1:]

    # Derivation cohort: every point of a line is a threshold
    for color, refractory_hours in zip(colors, REFRACTORY_HOURS_SHOWN):
        line_df = validation_grid_df[(validation_grid_df.n_consecutive_hours == N_CONSECUTIVE_HOURS_SHOWN)
                                     & (validation_grid_df.refractory_hours == refractory_hours)]
        line_df = line_df.sort_values('threshold')
        ax.plot(line_df.false_alarms_per_patient_day, line_df.event_sensitivity, '-o', color=color,
                lw=LINE_WIDTH, ms=2.5, label=f'R = {refractory_hours} h')

        # Ring: where the reported threshold lies on this line
        at_threshold = line_df[np.isclose(line_df.threshold, THRESHOLD)].iloc[0]
        ax.plot(at_threshold.false_alarms_per_patient_day, at_threshold.event_sensitivity, 'o', ms=5,
                mfc='white', mec=color, mew=LINE_WIDTH, zorder=4)

    # Test set: the two policies of panel B
    for policy, name in [(HOURLY_POLICY, 'Hourly'), (ALARM_POLICY, 'Alarm policy')]:
        policy_row = _policy_row(test_grid_df, policy)
        ax.errorbar(policy_row.false_alarms_per_patient_day, policy_row.event_sensitivity,
                    xerr=_ci_error(policy_row, 'false_alarms_per_patient_day'),
                    yerr=_ci_error(policy_row, 'event_sensitivity'),
                    fmt='D', color=COLOR_TEST_SET, ms=3.5, lw=REFERENCE_LINE_WIDTH, capsize=1.5, zorder=5)
        ax.annotate(name, (policy_row.false_alarms_per_patient_day, policy_row.event_sensitivity),
                    xytext=(5, -9), textcoords='offset points', fontsize=tick_size)

    ax.set_xscale('log')
    ax.set_xticks(FALSE_ALARM_RATE_TICKS)
    ax.set_xticklabels([f'{tick:g}' for tick in FALSE_ALARM_RATE_TICKS])
    ax.minorticks_off()
    ax.set_xlabel('False alarms per patient-day (log scale)', fontsize=label_size)
    ax.set_ylabel('Event-level sensitivity', fontsize=label_size)
    ax.tick_params('both', labelsize=tick_size)

    # Two legends, placed where no line runs: lines (lower right), markers (upper left)
    lines_legend = ax.legend(fontsize=tick_size, loc='lower right', title='Refractory period\n(derivation cohort)',
                             title_fontsize=tick_size)
    ax.add_artist(lines_legend)

    marker_handles = [
        Line2D([0], [0], marker='o', ls='', ms=5, mfc='white', mec=colors[-1], mew=LINE_WIDTH,
               label=f'Threshold = {THRESHOLD}'),
        Line2D([0], [0], marker='D', ls='', ms=3.5, color=COLOR_TEST_SET,
               label=f'Test set, threshold = {THRESHOLD} (95% CI)'),
    ]
    ax.legend(handles=marker_handles, fontsize=tick_size, loc='upper left')

    return ax


def _censor_at_first_event(scores, event_times):
    """
    Returns:
        (scores up to the first event, hour of the first event or None)
    """
    if len(event_times) == 0:
        return scores, None

    first_event_hour = int(event_times.min())
    return scores[:first_event_hour], first_event_hour


def _is_true_alarm(alarm_times, first_event_hour):
    if first_event_hour is None:
        return np.zeros(len(alarm_times), dtype=bool)
    return alarm_times >= first_event_hour - ALARM_POLICY.horizon_hours


def _pick_example_patients(scores_per_patient, event_times_per_patient):
    """
    First patients, in data order, whose alarms illustrate the policy:
    with END, a few false alarms then a true one; without END, a few alarms.

    Returns:
        (index of a patient with END, index of a patient without END)
    """
    idx_with_end, idx_without_end = None, None

    for idx, (scores, event_times) in enumerate(zip(scores_per_patient, event_times_per_patient)):
        scores, first_event_hour = _censor_at_first_event(scores, event_times)
        alarm_times = compute_alarm_times(scores, ALARM_POLICY)
        n_true_alarms = int(_is_true_alarm(alarm_times, first_event_hour).sum())
        n_false_alarms = len(alarm_times) - n_true_alarms

        if first_event_hour is None:
            n_min, n_max = EXAMPLE_N_ALARMS_WITHOUT_END_RANGE
            if idx_without_end is None and n_min <= n_false_alarms <= n_max:
                idx_without_end = idx
            continue

        if idx_with_end is not None:
            continue
        if not EXAMPLE_EVENT_HOUR_RANGE[0] <= first_event_hour <= EXAMPLE_EVENT_HOUR_RANGE[1]:
            continue
        n_min, n_max = EXAMPLE_N_FALSE_ALARMS_RANGE
        if n_true_alarms >= 1 and n_min <= n_false_alarms <= n_max:
            idx_with_end = idx

    if idx_with_end is None or idx_without_end is None:
        raise ValueError('No example patient matches the selection criteria; widen the EXAMPLE_* ranges')

    return idx_with_end, idx_without_end


def plot_example_patient(ax_score, ax_alarms, scores, event_times, n_time_steps,
                         tick_size=PUB_TICK, label_size=PUB_LABEL):
    scores, first_event_hour = _censor_at_first_event(scores, event_times)

    # Risk score against the threshold
    ax_score.plot(np.arange(len(scores)), scores, color=COLOR_XGB, lw=LINE_WIDTH)
    ax_score.axhline(THRESHOLD, color=COLOR_REFERENCE, lw=REFERENCE_LINE_WIDTH, ls='--')
    ax_score.annotate(f'Threshold = {THRESHOLD}', (n_time_steps - 0.5, THRESHOLD), xytext=(0, 2),
                      textcoords='offset points', color=COLOR_REFERENCE, fontsize=tick_size, ha='right', va='bottom')

    # END onset, preceded by the window in which an alarm counts as true
    if first_event_hour is not None:
        for ax in (ax_score, ax_alarms):
            ax.axvspan(first_event_hour - ALARM_POLICY.horizon_hours, first_event_hour,
                       color=COLOR_DETECTION_WINDOW, lw=0)
            ax.axvline(first_event_hour, color='black', lw=LINE_WIDTH)

    # One row of alarms per policy
    rows = [('Hourly', HOURLY_POLICY), ('Alarm policy', ALARM_POLICY)]
    for row_y, (_, policy) in zip(range(len(rows) - 1, -1, -1), rows):
        alarm_times = compute_alarm_times(scores, policy)
        is_true_alarm = _is_true_alarm(alarm_times, first_event_hour)

        # Hours silenced after each alarm
        silencing_alarm_times = alarm_times if policy.refractory_hours > 0 else []
        for alarm_time in silencing_alarm_times:
            ax_alarms.plot([alarm_time, min(alarm_time + policy.refractory_hours, len(scores) - 1)], [row_y, row_y],
                           color=COLOR_REFRACTORY, lw=3, solid_capstyle='butt', zorder=1)

        for is_true, color in [(True, COLOR_XGB), (False, COLOR_FALSE_ALARM)]:
            shown_alarm_times = alarm_times[is_true_alarm == is_true]
            ax_alarms.scatter(shown_alarm_times, np.full(len(shown_alarm_times), row_y), marker='|',
                              s=ALARM_MARKER_SIZE, color=color, lw=LINE_WIDTH, zorder=3)

    ax_score.set_ylabel('Risk score', fontsize=label_size)
    ax_score.tick_params('both', labelsize=tick_size, labelbottom=False)
    ax_score.set_xlim(-1, n_time_steps)

    ax_alarms.set_yticks(range(len(rows) - 1, -1, -1))
    ax_alarms.set_yticklabels([name for name, _ in rows])
    ax_alarms.set_ylim(-0.6, len(rows) - 0.4)
    ax_alarms.set_xlim(-1, n_time_steps)
    ax_alarms.set_xlabel('Hours since admission', fontsize=label_size)
    ax_alarms.tick_params('both', labelsize=tick_size)
    ax_alarms.grid(False)

    return ax_score, ax_alarms


def _alarm_legend_handles():
    return [
        Line2D([0], [0], marker='|', ls='', ms=7, mew=LINE_WIDTH, color=COLOR_XGB, label='True alarm'),
        Line2D([0], [0], marker='|', ls='', ms=7, mew=LINE_WIDTH, color=COLOR_FALSE_ALARM, label='False alarm'),
        Line2D([0], [0], color=COLOR_REFRACTORY, lw=3, label=f'Refractory period ({ALARM_POLICY.refractory_hours} h)'),
        Line2D([0], [0], color='black', lw=LINE_WIDTH, label='END onset'),
        Patch(color=COLOR_DETECTION_WINDOW, label=f'{ALARM_POLICY.horizon_hours} h before END onset'),
    ]


def main():
    parser = argparse.ArgumentParser(description='Supplementary figure: alarm policy')
    parser.add_argument('--validation_grid', required=True,
                        help='alarm_policy_results.csv of the derivation cohort (pooled out-of-fold predictions)')
    parser.add_argument('--test_grid', required=True,
                        help=f'alarm_policy_results.csv of the test set, with the policies at threshold {THRESHOLD}')
    parser.add_argument('--test_data', required=True, help='Test data (X, y) .pth file')
    parser.add_argument('--test_predictions', required=True, help='test_predictions.pkl')
    parser.add_argument('--model_config', required=True, help='final_model_config.json')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    validation_grid_df = pd.read_csv(args.validation_grid)
    test_grid_df = pd.read_csv(args.test_grid)

    with open(args.model_config) as f:
        restrict_to_first_event = bool(json.load(f).get('restrict_to_first_event', False))
    X_test, y_test_df = ch.load(args.test_data)
    n_time_steps = X_test.shape[1]
    scores_per_patient, event_times_per_patient = _load_patient_set(X_test, y_test_df, args.test_predictions,
                                                                    restrict_to_first_event)

    setup_theme()
    fig = plt.figure(figsize=(18 * CM, 17 * CM))

    # Panel A on top, panel B below
    outer_grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 2.9], hspace=0.3)
    ax_a = fig.add_subplot(outer_grid[0])
    plot_sensitivity_vs_alarm_burden(ax_a, validation_grid_df, test_grid_df)

    # Panel B: patient with END on the left, without END on the right
    bottom_grid = outer_grid[1].subgridspec(2, 2, height_ratios=[1.5, 0.7], hspace=0.1, wspace=0.28)
    idx_with_end, idx_without_end = _pick_example_patients(scores_per_patient, event_times_per_patient)
    first_score_ax = None
    for column, (idx, title) in enumerate([(idx_with_end, 'Patient with END'), (idx_without_end, 'Patient without END')]):
        ax_score = fig.add_subplot(bottom_grid[0, column])
        ax_alarms = fig.add_subplot(bottom_grid[1, column])
        plot_example_patient(ax_score, ax_alarms, scores_per_patient[idx], event_times_per_patient[idx], n_time_steps)
        ax_score.set_title(title, fontsize=PUB_LABEL, loc='left')
        first_score_ax = first_score_ax or ax_score

    fig.legend(handles=_alarm_legend_handles(), loc='lower center', ncol=5, fontsize=PUB_TICK, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    # Panel labels
    for label, ax, x_offset in [('A.', ax_a, -0.07), ('B.', first_score_ax, -0.16)]:
        ax.text(x_offset, 1.12, label, transform=ax.transAxes, fontsize=PUB_SUBPLOT_NUM, fontweight='bold', va='top')

    save_figure(fig, args.output_dir, STEM)
    plt.close(fig)


if __name__ == '__main__':
    main()
