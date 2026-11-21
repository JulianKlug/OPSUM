"""
Renders what is known about a patient at one hour into text an LLM can read.

    PatientHistory (hours 0..t only) ──> baseline       static facts and imaging
                                     ├─> current status vitals, scales and labs at hour t
                                     ├─> trajectory     hourly table of key variables
                                     └─> XGB explanation risk curve, top SHAP drivers and their curves

A PatientHistory holds nothing later than hour t, so no text built from it
can leak the future.
"""
from typing import List, NamedTuple

import numpy as np

from prediction.short_term_outcome_prediction.second_stage_filter.feature_layout import (
    AggregatedFeature, Block, ClinicalUnits)

BINARY_ON_THRESHOLD = 0.5
BINARY_NAME_PREFIXES = ('categorical_', 'medhist_')

VITAL_SIGNALS = {
    'NIHSS': 'NIHSS',
    'systolic_blood_pressure': 'Systolic BP (mmHg)',
    'diastolic_blood_pressure': 'Diastolic BP (mmHg)',
    'mean_blood_pressure': 'Mean BP (mmHg)',
    'heart_rate': 'Heart rate (/min)',
    'oxygen_saturation': 'SpO2 (%)',
    'respiratory_rate': 'Respiratory rate (/min)',
}
SINGLE_VALUE_VITALS = {
    'Glasgow Coma Scale': 'GCS',
    'temperature': 'Temperature (C)',
    'FIO2': 'FiO2 (%)',
    'glucose': 'Glucose (mmol/L)',
}
DEMOGRAPHICS = {'age': 'Age (years)', 'weight': 'Weight (kg)'}
IMAGING_VOLUME_PREFIXES = ('cbf_', 'cbv_', 'tmax_')

TRAJECTORY_COLUMNS = {
    'median_NIHSS': 'NIHSS',
    'max_NIHSS': 'NIHSSmax',
    'Glasgow Coma Scale': 'GCS',
    'median_systolic_blood_pressure': 'SBP',
    'median_diastolic_blood_pressure': 'DBP',
    'median_heart_rate': 'HR',
    'median_oxygen_saturation': 'SpO2',
    'median_respiratory_rate': 'RR',
    'temperature': 'Temp',
    'glucose': 'Gluc',
}

# Every hour of the recent past, a coarser grid before: keeps the table short on long stays
RECENT_WINDOW_HOURS = 12
EARLIER_HOURS_STEP = 4

N_TOP_SHAP_DRIVERS = 10
N_SHAP_CURVES = 5


class PatientHistory(NamedTuple):
    """One patient from admission (row 0) to the alarm hour (last row)."""
    features: np.ndarray     # (hours, aggregated columns), z-scored
    shap_values: np.ndarray  # (hours, aggregated columns + 1), log-odds, last column = base value
    xgb_scores: np.ndarray   # (hours,), probabilities

    @property
    def alarm_hour(self) -> int:
        return len(self.xgb_scores) - 1


class NarrativeContext(NamedTuple):
    """Fixed knowledge about the dataset, shared by all patients."""
    layout: List[AggregatedFeature]
    raw_feature_names: List[str]
    clinical_units: ClinicalUnits
    english_names: dict


def format_number(value: float) -> str:
    """Three significant digits, no trailing noise: 163.72 -> '164', 1.2345 -> '1.23'."""
    return f'{value:.3g}'


def english_name(raw_feature: str, context: NarrativeContext) -> str:
    return context.english_names.get(raw_feature, raw_feature)


def is_binary(raw_feature: str, context: NarrativeContext) -> bool:
    """One-hot and yes/no variables. A few one-hots were normalised by the pipeline, hence the prefix check."""
    return raw_feature.startswith(BINARY_NAME_PREFIXES) or not context.clinical_units.is_normalised(raw_feature)


def raw_series(history: PatientHistory, raw_feature: str, context: NarrativeContext) -> np.ndarray:
    """Hourly values of one raw variable in clinical units. Raw variables are the first columns of the layout."""
    column = context.raw_feature_names.index(raw_feature)

    return context.clinical_units.convert_raw(history.features[:, column], raw_feature)


def current_value(history: PatientHistory, raw_feature: str, context: NarrativeContext) -> float:
    return float(raw_series(history, raw_feature, context)[-1])


def render_baseline(history: PatientHistory, context: NarrativeContext) -> str:
    """Demographics, the yes/no facts that apply to this patient, and imaging volumes."""
    lines = [f'{label}: {format_number(current_value(history, name, context))}'
             for name, label in DEMOGRAPHICS.items()]

    present_facts = [english_name(name, context) for name in context.raw_feature_names
                     if is_binary(name, context) and current_value(history, name, context) > BINARY_ON_THRESHOLD]
    lines.append('Present: ' + ('; '.join(present_facts) if present_facts else 'none recorded'))

    imaging_volumes = [f'{name} {format_number(current_value(history, name, context))} mL'
                       for name in context.raw_feature_names if name.startswith(IMAGING_VOLUME_PREFIXES)]
    lines.append('Perfusion imaging volumes: ' + ', '.join(imaging_volumes))

    return '\n'.join(lines)


def render_current_status(history: PatientHistory, context: NarrativeContext) -> str:
    """Vitals and scales of the alarm hour (median with range), then the latest labs."""
    lines = []
    for signal, label in VITAL_SIGNALS.items():
        minimum, median, maximum = (format_number(current_value(history, f'{statistic}_{signal}', context))
                                    for statistic in ('min', 'median', 'max'))
        lines.append(f'{label}: {median} (range {minimum}-{maximum})')

    lines += [f'{label}: {format_number(current_value(history, name, context))}'
              for name, label in SINGLE_VALUE_VITALS.items()]

    already_shown = set(SINGLE_VALUE_VITALS) | set(DEMOGRAPHICS)
    labs = [f'{english_name(name, context)} {format_number(current_value(history, name, context))}'
            for name in context.raw_feature_names
            if not is_binary(name, context)
            and name not in already_shown
            and not name.startswith(IMAGING_VOLUME_PREFIXES + ('min_', 'median_', 'max_'))]
    lines.append('Latest labs: ' + ', '.join(labs))

    return '\n'.join(lines)


def trajectory_hours(alarm_hour: int) -> np.ndarray:
    """
    Hours shown in the trajectory table.

    e.g. alarm at hour 30 -> 0, 4, 8, 12, 16 (coarse) then 19, 20, ..., 30 (every hour)
    """
    recent_start = max(alarm_hour - RECENT_WINDOW_HOURS + 1, 0)
    earlier = np.arange(0, recent_start, EARLIER_HOURS_STEP)
    recent = np.arange(recent_start, alarm_hour + 1)

    return np.concatenate([earlier, recent])


def render_trajectory(history: PatientHistory, context: NarrativeContext) -> str:
    """Hourly table of the key variables from admission to the alarm hour."""
    hours = trajectory_hours(history.alarm_hour)
    series = {label: raw_series(history, name, context) for name, label in TRAJECTORY_COLUMNS.items()}

    lines = ['hour | ' + ' | '.join(series)]
    for hour in hours:
        lines.append(f'{hour} | ' + ' | '.join(format_number(values[hour]) for values in series.values()))

    return '\n'.join(lines)


def describe_feature(feature: AggregatedFeature, context: NarrativeContext) -> str:
    """e.g. (ROLLING_TREND, 'max_NIHSS') -> 'maximum NIHSS, trend per hour over last 6h'"""
    if feature.block == Block.HOUR:
        return feature.block.value

    return f'{english_name(feature.raw_feature, context)}, {feature.block.value}'


def render_xgb_explanation(history: PatientHistory, context: NarrativeContext) -> str:
    """
    The first-stage model's view: its risk over the recent hours, what drives the
    current score, and how the main drivers' contributions evolved.
    """
    recent = slice(max(history.alarm_hour - RECENT_WINDOW_HOURS + 1, 0), history.alarm_hour + 1)
    contributions = history.shap_values[:, :-1]

    # Rank by absolute contribution at the alarm hour: strong protective factors matter as much as risk factors
    top_columns = np.argsort(-np.abs(contributions[-1]))[:N_TOP_SHAP_DRIVERS]

    lines = [
        f'Risk score now: {history.xgb_scores[-1]:.3f}',
        'Risk score, recent hours (oldest to now): ' + ' '.join(f'{score:.2f}' for score in history.xgb_scores[recent]),
        'Top drivers now (value -> contribution in log-odds, positive raises risk):',
    ]

    for column in top_columns:
        feature = context.layout[column]
        value = context.clinical_units.convert(float(history.features[-1, column]), feature)
        lines.append(f'- {describe_feature(feature, context)}: {format_number(value)} -> {contributions[-1, column]:+.2f}')

    lines.append('Contribution curves of the main drivers, recent hours (oldest to now):')
    for column in top_columns[:N_SHAP_CURVES]:
        curve = ' '.join(f'{contribution:+.2f}' for contribution in contributions[recent, column])
        lines.append(f'- {describe_feature(context.layout[column], context)}: {curve}')

    return '\n'.join(lines)
