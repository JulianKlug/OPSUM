import numpy as np
import pandas as pd
import pytest

from prediction.short_term_outcome_prediction.evaluation.alarm_policy import (
    AlarmPolicy, AlarmTrigger, EventCensoring, compute_alarm_times, score_patient, score_cohort, summarise_burden,
    bootstrap_burden, evaluate_policy_grid, select_policy)

THRESHOLD = 0.5
LOW, HIGH = 0.1, 0.9


def _scores(above_threshold_flags):
    return np.where(np.array(above_threshold_flags) == 1, HIGH, LOW)


def test_every_hour_above_threshold_alarms_without_k_or_refractory():
    """k=1, R=0 must reproduce the per-hour analysis"""
    alarm_times = compute_alarm_times(_scores([0, 1, 1, 0, 1]), AlarmPolicy(THRESHOLD))

    assert alarm_times.tolist() == [1, 2, 4]


def test_score_equal_to_threshold_alarms():
    alarm_times = compute_alarm_times([THRESHOLD], AlarmPolicy(THRESHOLD))

    assert alarm_times.tolist() == [0]


def test_alarm_requires_k_consecutive_hours():
    policy = AlarmPolicy(THRESHOLD, n_consecutive_hours=3)

    # runs of 2 never alarm; the run of 3 alarms on its 3rd hour
    alarm_times = compute_alarm_times(_scores([1, 1, 0, 1, 1, 0, 1, 1, 1]), policy)

    assert alarm_times.tolist() == [8]


def test_refractory_period_silences_following_hours():
    policy = AlarmPolicy(THRESHOLD, refractory_hours=3)

    # alarm at 0 silences hours 1-3, re-armed at 4
    alarm_times = compute_alarm_times(_scores([1, 1, 1, 1, 1, 1]), policy)

    assert alarm_times.tolist() == [0, 4]


def test_k_and_refractory_combined():
    policy = AlarmPolicy(THRESHOLD, n_consecutive_hours=2, refractory_hours=3)

    alarm_times = compute_alarm_times(_scores([0, 1, 1, 1, 1, 1, 1]), policy)

    assert alarm_times.tolist() == [2, 6]


def test_rising_score_trigger_alarms_only_when_score_exceeds_previous_hour():
    policy = AlarmPolicy(THRESHOLD, trigger=AlarmTrigger.RISING_SCORE)

    # hour 0 has no previous score; hour 2 is a plateau, hour 4 a decrease, hour 6 a rise that stays below threshold
    alarm_times = compute_alarm_times([0.6, 0.7, 0.7, 0.8, 0.75, 0.2, 0.3, 0.55], policy)

    assert alarm_times.tolist() == [0, 1, 3, 7]


def test_new_peak_trigger_alarms_only_above_score_of_last_alarm():
    policy = AlarmPolicy(THRESHOLD, trigger=AlarmTrigger.NEW_PEAK)

    # hours 2-3 rise again but stay below the score alarmed at hour 1
    alarm_times = compute_alarm_times([0.6, 0.8, 0.7, 0.75, 0.9], policy)

    assert alarm_times.tolist() == [0, 1, 4]


def test_new_peak_trigger_starts_afresh_after_score_falls_below_threshold():
    policy = AlarmPolicy(THRESHOLD, trigger=AlarmTrigger.NEW_PEAK)

    alarm_times = compute_alarm_times([0.9, 0.2, 0.6], policy)

    assert alarm_times.tolist() == [0, 2]


def test_new_peak_trigger_compares_to_last_alarm_after_refractory_period():
    policy = AlarmPolicy(THRESHOLD, refractory_hours=2, trigger=AlarmTrigger.NEW_PEAK)

    # rise at hours 1-2 is silenced; hour 3 is above the score alarmed at hour 0
    alarm_times = compute_alarm_times([0.6, 0.7, 0.8, 0.75, 0.7], policy)

    assert alarm_times.tolist() == [0, 3]


@pytest.mark.parametrize('invalid_policy_kwargs', [
    {'n_consecutive_hours': 0}, {'refractory_hours': -1}, {'horizon_hours': 0}])
def test_invalid_policy_is_rejected(invalid_policy_kwargs):
    with pytest.raises(ValueError):
        AlarmPolicy(THRESHOLD, **invalid_policy_kwargs)


def test_alarm_within_horizon_before_event_is_true_and_detects_event():
    flags = np.zeros(20)
    flags[6] = 1

    tally = score_patient(_scores(flags), [10], AlarmPolicy(THRESHOLD, horizon_hours=6))

    assert (tally.n_true_alarms, tally.n_false_alarms) == (1, 0)
    assert (tally.n_events, tally.n_detected_events) == (1, 1)
    assert tally.lead_times_hours == (4,)


def test_alarm_earlier_than_horizon_is_false_and_event_missed():
    flags = np.zeros(20)
    flags[3] = 1

    # event at 10, horizon 6 -> detection window is hours 4-9
    tally = score_patient(_scores(flags), [10], AlarmPolicy(THRESHOLD, horizon_hours=6))

    assert (tally.n_true_alarms, tally.n_false_alarms) == (0, 1)
    assert (tally.n_events, tally.n_detected_events) == (1, 0)
    assert tally.lead_times_hours == ()


def test_horizon_bounds_match_training_label():
    """Label at ts is positive if event in (ts, ts + H]: hour e - H is in, hour e is out"""
    policy = AlarmPolicy(THRESHOLD, horizon_hours=6)
    flags_at_window_start, flags_at_event = np.zeros(20), np.zeros(20)
    flags_at_window_start[4] = 1
    flags_at_event[10] = 1

    tally_at_window_start = score_patient(_scores(flags_at_window_start), [10], policy, EventCensoring.NONE)
    tally_at_event = score_patient(_scores(flags_at_event), [10], policy, EventCensoring.NONE)

    assert tally_at_window_start.n_true_alarms == 1
    assert tally_at_event.n_true_alarms == 0


def test_lead_time_counts_from_earliest_alarm_in_window():
    flags = np.zeros(20)
    flags[[5, 8]] = 1

    tally = score_patient(_scores(flags), [10], AlarmPolicy(THRESHOLD, horizon_hours=6))

    assert tally.n_true_alarms == 2
    assert tally.n_detected_events == 1
    assert tally.lead_times_hours == (5,)


def test_censoring_at_first_event_ignores_later_hours_and_events():
    flags = np.zeros(30)
    flags[[15, 22]] = 1

    tally = score_patient(_scores(flags), [10, 25], AlarmPolicy(THRESHOLD), EventCensoring.AT_FIRST_EVENT)

    assert tally.n_observed_hours == 10
    assert (tally.n_true_alarms, tally.n_false_alarms) == (0, 0)
    assert (tally.n_events, tally.n_detected_events) == (1, 0)


def test_without_censoring_alarms_are_scored_against_later_events():
    flags = np.zeros(30)
    flags[[15, 22]] = 1

    tally = score_patient(_scores(flags), [10, 25], AlarmPolicy(THRESHOLD), EventCensoring.NONE)

    assert tally.n_observed_hours == 30
    assert (tally.n_true_alarms, tally.n_false_alarms) == (1, 1)
    assert (tally.n_events, tally.n_detected_events) == (2, 1)


def test_event_at_hour_zero_is_not_predictable():
    tally = score_patient(_scores(np.ones(10)), [0], AlarmPolicy(THRESHOLD))

    assert tally.had_event
    assert tally.n_events == 0
    assert tally.n_observed_hours == 0


def test_patient_without_event_only_yields_false_alarms():
    tally = score_patient(_scores([0, 1, 0, 1]), [], AlarmPolicy(THRESHOLD))

    assert not tally.had_event
    assert (tally.n_true_alarms, tally.n_false_alarms) == (0, 2)


def _toy_cohort():
    """3 patients x 48h: one detected event, one event-free with 2 false alarms, one event-free and silent"""
    detected = np.zeros(48)
    detected[20] = 1
    false_alarms = np.zeros(48)
    false_alarms[[5, 30]] = 1
    silent = np.zeros(48)

    scores_per_patient = [_scores(detected), _scores(false_alarms), _scores(silent)]
    event_times_per_patient = [[24], [], []]

    return scores_per_patient, event_times_per_patient


def test_summarise_burden():
    scores_per_patient, event_times_per_patient = _toy_cohort()

    summary = summarise_burden(score_cohort(scores_per_patient, event_times_per_patient, AlarmPolicy(THRESHOLD)))

    # 24h (censored at event) + 48h + 48h
    assert summary['patient_days'] == pytest.approx(5)
    assert summary['n_alarms'] == 3
    assert summary['event_sensitivity'] == pytest.approx(1)
    assert summary['alarm_ppv'] == pytest.approx(1 / 3)
    assert summary['false_alarms_per_true_alarm'] == pytest.approx(2)
    assert summary['false_alarms_per_patient_day'] == pytest.approx(2 / 5)
    assert summary['lead_time_hours_median'] == pytest.approx(4)
    assert summary['fraction_event_free_patients_never_alarmed'] == pytest.approx(1 / 2)


def test_summarise_burden_without_alarms_or_events_yields_nan():
    summary = summarise_burden(score_cohort([_scores(np.zeros(24))], [[]], AlarmPolicy(THRESHOLD)))

    assert np.isnan(summary['alarm_ppv'])
    assert np.isnan(summary['event_sensitivity'])
    assert summary['false_alarms_per_patient_day'] == 0


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    scores_per_patient, event_times_per_patient = _toy_cohort()
    tallies = score_cohort(scores_per_patient, event_times_per_patient, AlarmPolicy(THRESHOLD))
    point_estimate = summarise_burden(tallies)['false_alarms_per_patient_day']

    lower_ci, upper_ci = bootstrap_burden(tallies, n_bootstrap=200)['false_alarms_per_patient_day']

    assert lower_ci <= point_estimate <= upper_ci
    assert bootstrap_burden(tallies, n_bootstrap=200)['false_alarms_per_patient_day'] == (lower_ci, upper_ci)


def test_policy_grid_has_one_row_per_policy_and_stricter_policy_alarms_less():
    scores_per_patient, event_times_per_patient = _toy_cohort()

    grid_df = evaluate_policy_grid(scores_per_patient, event_times_per_patient, thresholds=[THRESHOLD],
                                   n_consecutive_hours_options=[1, 2], refractory_hours_options=[0, 6],
                                   n_bootstrap=10)

    assert len(grid_df) == 4
    assert 'alarm_ppv_lower_ci' in grid_df.columns
    n_alarms_by_k = grid_df.groupby('n_consecutive_hours').n_alarms.max()
    assert n_alarms_by_k[2] < n_alarms_by_k[1]


def test_select_policy_minimises_false_alarms_among_sensitive_enough_policies():
    grid_df = pd.DataFrame({
        'threshold': [0.2, 0.3, 0.4],
        'event_sensitivity': [0.9, 0.8, 0.6],
        'false_alarms_per_patient_day': [3.0, 1.5, 0.5],
    })

    selected_policy = select_policy(grid_df, min_event_sensitivity=0.8)

    assert selected_policy.threshold == 0.3


def test_select_policy_fails_if_no_policy_is_sensitive_enough():
    grid_df = pd.DataFrame({'event_sensitivity': [0.5], 'false_alarms_per_patient_day': [1.0]})

    with pytest.raises(ValueError):
        select_policy(grid_df, min_event_sensitivity=0.8)
