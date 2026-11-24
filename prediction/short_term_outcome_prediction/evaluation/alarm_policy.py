"""
Alarm policy: turn hourly risk scores into discrete alarms, then score burden per alarm instead of per hour.

Why: hourly PPV counts one sustained excursion as many false positives, and one caught event as many true positives.

    score ─┐         ┌───────────┐
        θ ─┼─────────┼───────────┼──────
           └─────────┘           └────
                      ^ k hours ≥ θ → ALARM
                       |─ refractory R ─| silent, then re-armed

    Alarm at hour t is true     <=> an event starts in (t, t + H]   (same window as the training label)
    Event at hour e is detected <=> an alarm fired in [e - H, e)
"""
from dataclasses import dataclass, fields
from enum import Enum
from itertools import product

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24

# An event at hour 0 has no preceding score, hence cannot be predicted
FIRST_PREDICTABLE_EVENT_HOUR = 1

# Stands for a score that does not exist (before the first hour, before the first alarm): any score exceeds it
NO_SCORE = -np.inf

DEFAULT_HORIZON_HOURS = 6
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_SEED = 42

LOWER_CI_PERCENTILE = 2.5
UPPER_CI_PERCENTILE = 97.5
LOWER_CI_SUFFIX = '_lower_ci'
UPPER_CI_SUFFIX = '_upper_ci'

SENSITIVITY_METRIC = 'event_sensitivity'
FALSE_ALARM_RATE_METRIC = 'false_alarms_per_patient_day'


class EventCensoring(Enum):
    """What happens to a patient's timeline once END has occurred."""
    # Stop at the first event: later hours and later events are ignored
    AT_FIRST_EVENT = 'at_first_event'
    # Keep the full timeline: every event counts, alarms after an event are scored against the next one
    NONE = 'none'


class AlarmTrigger(Enum):
    """What an alarm requires, on top of a score above threshold."""
    ABOVE_THRESHOLD = 'above_threshold'
    # Score higher than at the previous hour.            e.g. [.3 .4 .4 .5 .45] -> alarms at hours 0, 1, 3
    RISING_SCORE = 'rising_score'
    # Score higher than at the last alarm of the ongoing excursion above threshold.
    #                                                    e.g. [.3 .5 .4 .45 .6] -> alarms at hours 0, 1, 4
    NEW_PEAK = 'new_peak'


@dataclass(frozen=True)
class AlarmPolicy:
    threshold: float
    n_consecutive_hours: int = 1
    refractory_hours: int = 0
    horizon_hours: int = DEFAULT_HORIZON_HOURS
    trigger: AlarmTrigger = AlarmTrigger.ABOVE_THRESHOLD

    def __post_init__(self):
        if self.n_consecutive_hours < 1:
            raise ValueError('n_consecutive_hours must be >= 1')
        if self.refractory_hours < 0:
            raise ValueError('refractory_hours must be >= 0')
        if self.horizon_hours < 1:
            raise ValueError('horizon_hours must be >= 1')


@dataclass(frozen=True)
class PatientTally:
    """Alarm counts of a single patient. Kept per patient so that the bootstrap can resample patients."""
    n_observed_hours: int
    n_true_alarms: int
    n_false_alarms: int
    n_events: int
    n_detected_events: int
    lead_times_hours: tuple
    had_event: bool


def compute_alarm_times(scores, policy: AlarmPolicy) -> np.ndarray:
    """
    Hours at which an alarm fires.
    e.g. k=2, R=3, above threshold = [0 1 1 1 1 1 1] -> alarms at hours 2 and 6
    """
    alarm_times = []
    n_hours_above = 0
    silenced_until = -1
    previous_score = NO_SCORE
    last_alarm_score = NO_SCORE

    for hour, score in enumerate(np.asarray(scores, dtype=float)):
        is_above = score >= policy.threshold
        n_hours_above = n_hours_above + 1 if is_above else 0

        # Score to exceed for the trigger to be met
        score_to_exceed = _score_to_exceed(policy.trigger, previous_score, last_alarm_score)
        previous_score = score

        # Excursion above threshold is over: the next one starts afresh
        if not is_above:
            last_alarm_score = NO_SCORE

        if n_hours_above < policy.n_consecutive_hours:
            continue
        if hour <= silenced_until:
            continue
        if score <= score_to_exceed:
            continue

        alarm_times.append(hour)
        silenced_until = hour + policy.refractory_hours
        last_alarm_score = score

    return np.array(alarm_times, dtype=int)


def _score_to_exceed(trigger: AlarmTrigger, previous_score: float, last_alarm_score: float) -> float:
    if trigger is AlarmTrigger.RISING_SCORE:
        return previous_score
    if trigger is AlarmTrigger.NEW_PEAK:
        return last_alarm_score
    return NO_SCORE


def score_patient(scores, event_times, policy: AlarmPolicy,
                  censoring: EventCensoring = EventCensoring.AT_FIRST_EVENT) -> PatientTally:
    """
    Args:
        scores: hourly risk scores of one patient
        event_times: hours at which an event starts (empty if none)
    """
    scores = np.asarray(scores)
    event_times = np.sort(np.asarray(event_times, dtype=int))
    had_event = len(event_times) > 0

    if had_event and censoring is EventCensoring.AT_FIRST_EVENT:
        event_times = event_times[:1]
        scores = scores[:event_times[0]]

    alarm_times = compute_alarm_times(scores, policy)

    # An alarm is true if an event starts within the horizon following it
    n_true_alarms = sum(
        np.any((event_times > alarm_time) & (event_times <= alarm_time + policy.horizon_hours))
        for alarm_time in alarm_times
    )

    # An event is detected if an alarm fired within the horizon preceding it; lead time counts from the earliest one
    predictable_event_times = event_times[event_times >= FIRST_PREDICTABLE_EVENT_HOUR]
    lead_times_hours = []
    for event_time in predictable_event_times:
        preceding_alarm_times = alarm_times[(alarm_times >= event_time - policy.horizon_hours)
                                            & (alarm_times < event_time)]
        if len(preceding_alarm_times) == 0:
            continue
        lead_times_hours.append(int(event_time - preceding_alarm_times.min()))

    return PatientTally(
        n_observed_hours=len(scores),
        n_true_alarms=int(n_true_alarms),
        n_false_alarms=len(alarm_times) - int(n_true_alarms),
        n_events=len(predictable_event_times),
        n_detected_events=len(lead_times_hours),
        lead_times_hours=tuple(lead_times_hours),
        had_event=had_event,
    )


def score_cohort(scores_per_patient, event_times_per_patient, policy: AlarmPolicy,
                 censoring: EventCensoring = EventCensoring.AT_FIRST_EVENT) -> list:
    return [score_patient(scores, event_times, policy, censoring)
            for scores, event_times in zip(scores_per_patient, event_times_per_patient)]


def _ratio(numerator, denominator) -> float:
    if denominator == 0:
        return float('nan')
    return float(numerator / denominator)


def _lead_time_percentile(lead_times_hours, percentile) -> float:
    if len(lead_times_hours) == 0:
        return float('nan')
    return float(np.percentile(lead_times_hours, percentile))


def summarise_burden(tallies) -> dict:
    """Cohort-level alarm burden from per-patient tallies."""
    n_true_alarms = sum(tally.n_true_alarms for tally in tallies)
    n_false_alarms = sum(tally.n_false_alarms for tally in tallies)
    n_alarms = n_true_alarms + n_false_alarms
    n_events = sum(tally.n_events for tally in tallies)
    n_detected_events = sum(tally.n_detected_events for tally in tallies)
    patient_days = sum(tally.n_observed_hours for tally in tallies) / HOURS_PER_DAY
    lead_times_hours = [lead_time for tally in tallies for lead_time in tally.lead_times_hours]

    # Patients in whom any alarm is by definition a false one
    event_free_tallies = [tally for tally in tallies if not tally.had_event]
    n_event_free_never_alarmed = sum(tally.n_false_alarms == 0 for tally in event_free_tallies)

    return {
        'n_patients': len(tallies),
        'patient_days': float(patient_days),
        'n_alarms': n_alarms,
        'n_true_alarms': n_true_alarms,
        'n_false_alarms': n_false_alarms,
        'n_events': n_events,
        'n_detected_events': n_detected_events,
        SENSITIVITY_METRIC: _ratio(n_detected_events, n_events),
        'alarm_ppv': _ratio(n_true_alarms, n_alarms),
        'false_alarms_per_true_alarm': _ratio(n_false_alarms, n_true_alarms),
        'alarms_per_patient_day': _ratio(n_alarms, patient_days),
        FALSE_ALARM_RATE_METRIC: _ratio(n_false_alarms, patient_days),
        'lead_time_hours_q1': _lead_time_percentile(lead_times_hours, 25),
        'lead_time_hours_median': _lead_time_percentile(lead_times_hours, 50),
        'lead_time_hours_q3': _lead_time_percentile(lead_times_hours, 75),
        'fraction_event_free_patients_never_alarmed': _ratio(n_event_free_never_alarmed, len(event_free_tallies)),
    }


def bootstrap_burden(tallies, n_bootstrap: int = DEFAULT_N_BOOTSTRAP, seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    """
    95% CI of every burden metric.
    Resamples patients, not hours: hours within a patient are correlated.

    Returns:
        {metric: (lower_ci, upper_ci)}
    """
    rng = np.random.default_rng(seed)
    n_patients = len(tallies)

    bootstrapped_summaries = []
    for _ in range(n_bootstrap):
        resampled_idx = rng.integers(0, n_patients, size=n_patients)
        bootstrapped_summaries.append(summarise_burden([tallies[idx] for idx in resampled_idx]))
    bootstrapped_df = pd.DataFrame(bootstrapped_summaries)

    confidence_intervals = {}
    for metric in bootstrapped_df.columns:
        values = bootstrapped_df[metric].dropna().values
        if len(values) == 0:
            confidence_intervals[metric] = (float('nan'), float('nan'))
            continue
        confidence_intervals[metric] = (float(np.percentile(values, LOWER_CI_PERCENTILE)),
                                        float(np.percentile(values, UPPER_CI_PERCENTILE)))

    return confidence_intervals


def evaluate_policy_grid(scores_per_patient, event_times_per_patient,
                         thresholds, n_consecutive_hours_options, refractory_hours_options,
                         horizon_hours: int = DEFAULT_HORIZON_HOURS,
                         censoring: EventCensoring = EventCensoring.AT_FIRST_EVENT,
                         n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                         trigger: AlarmTrigger = AlarmTrigger.ABOVE_THRESHOLD) -> pd.DataFrame:
    """
    Burden of every (threshold, k, R) combination.

    Returns:
        one row per policy: policy parameters, burden metrics, and <metric>_lower_ci / <metric>_upper_ci
    """
    rows = []
    for threshold, n_consecutive_hours, refractory_hours in product(thresholds, n_consecutive_hours_options,
                                                                    refractory_hours_options):
        policy = AlarmPolicy(threshold=threshold, n_consecutive_hours=n_consecutive_hours,
                             refractory_hours=refractory_hours, horizon_hours=horizon_hours, trigger=trigger)
        tallies = score_cohort(scores_per_patient, event_times_per_patient, policy, censoring)

        row = {field.name: getattr(policy, field.name) for field in fields(policy)}
        row['trigger'] = trigger.value
        row['censoring'] = censoring.value
        row.update(summarise_burden(tallies))

        for metric, (lower_ci, upper_ci) in bootstrap_burden(tallies, n_bootstrap=n_bootstrap).items():
            row[metric + LOWER_CI_SUFFIX] = lower_ci
            row[metric + UPPER_CI_SUFFIX] = upper_ci

        rows.append(row)

    return pd.DataFrame(rows)


def select_policy(policy_grid_df: pd.DataFrame, min_event_sensitivity: float) -> pd.Series:
    """
    Pre-specified selection rule: least false alarms per patient-day among policies reaching the target sensitivity.
    To be applied on validation data only; the selected policy is then frozen for the test set.
    """
    eligible_df = policy_grid_df[policy_grid_df[SENSITIVITY_METRIC] >= min_event_sensitivity]
    if eligible_df.empty:
        raise ValueError(f'No policy reaches an event sensitivity of {min_event_sensitivity}')

    return eligible_df.loc[eligible_df[FALSE_ALARM_RATE_METRIC].idxmin()]
