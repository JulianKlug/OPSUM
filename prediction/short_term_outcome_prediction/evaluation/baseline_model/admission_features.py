"""
Admission (baseline) features and labels for the END baseline model (reviewer response).

    X (n, 72, labels, [cid, ts, label, value])
        hour 0 values ------------------------------------> admission table (1 row / patient)
    preprocessed_features_*.csv (imaging only)                   ^
        first measured value within ADMISSION_IMAGING_WINDOW_H --+ overrides hour 0 (else population-imputed)

    t0    perfusion CT (e.g. 3 h)               24 h
    |-------x-----------------------------------|----> hours
    admission imaging = first measured value in [0, 24)

Labels mirror aggregate_and_label_timeseries (target_interval=True):
    prediction at ts, label = END in bins ts+1..ts+6
    patient label = any hourly label (END after hour 0 within the observation window)
"""
import numpy as np
import pandas as pd

from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.concern_timing_analysis import \
    PREDICTION_HORIZON_H
from prediction.short_term_outcome_prediction.evaluation.first_event.first_event_evaluation import EVENT_BIN

# X last axis: [case_admission_id, ts, sample_label, value]
CID_COL, LABEL_COL, VALUE_COL = 0, 2, 3

ADMISSION_IMAGING_WINDOW_H = 24
IMPUTED_SOURCE_TAG = 'imputed'
CSV_CHUNK_ROWS = 2_000_000
TIMESTEP = 'ts'

IMAGING_LABELS = [
    'cbf_lt_20', 'cbf_lt_30', 'cbf_lt_34', 'cbf_lt_38',
    'cbv_lt_34', 'cbv_lt_38', 'cbv_lt_42',
    'tmax_gt_4', 'tmax_gt_6', 'tmax_gt_8', 'tmax_gt_10',
    'hypoperfusion_with_mismatch', 'hypoperfusion_without_mismatch',
    'vascular_occlusion', 'vascular_stenosis_over_50p',
]

# at hour 0, min_/max_ NIHSS and vitals duplicate the median_ value -> keep median_ only
HOUR0_REDUNDANT_PREFIXES = ('min_', 'max_')

# one-hot groups enter / leave a model together (reference level absent)
CATEGORICAL_GROUP_PREFIXES = (
    'categorical_onset_to_admission_time_', 'categorical_ivt_', 'categorical_iat_',
    'prestroke_disability_(rankin)_', 'referral_',
)

ONSET_TO_ADMISSION = ['categorical_onset_to_admission_time_541-1440min',
                      'categorical_onset_to_admission_time_<270min',
                      'categorical_onset_to_admission_time_>1440min',
                      'categorical_onset_to_admission_time_intra_hospital',
                      'categorical_onset_to_admission_time_onset_unknown']
IVT = ['categorical_ivt_91-270min', 'categorical_ivt_<90min', 'categorical_ivt_>540min', 'categorical_ivt_no_ivt']
EVT = ['categorical_iat_<270min', 'categorical_iat_>540min', 'categorical_iat_no_iat']

# literature predictor -> model features (see literature_review.md)
LITERATURE_PREDICTORS = {
    'age': ['age'],
    'admission_nihss': ['median_NIHSS'],
    'glucose': ['glucose'],
    'systolic_bp': ['median_systolic_blood_pressure'],
    'hypertension': ['medhist_hypertension_yes'],
    'diabetes': ['medhist_diabetes_yes'],
    'atrial_fibrillation': ['medhist_atrial_fibr._yes'],
    'prior_cerebrovascular_event': ['medhist_cerebrovascular_event_true'],
    'prior_antiplatelet': ['antiplatelet_drugs_yes'],
    'onset_to_admission': ONSET_TO_ADMISSION,
    'ivt': IVT,
    'evt': EVT,
    'vessel_occlusion': ['vascular_occlusion'],
    'core_volume': ['cbf_lt_30'],
}


def literature_features() -> list:
    return [f for features in LITERATURE_PREDICTORS.values() for f in features]


def feature_group(feature: str) -> str:
    """'categorical_ivt_<90min' -> 'categorical_ivt_'; non-categorical features are their own group."""
    return next((p for p in CATEGORICAL_GROUP_PREFIXES if feature.startswith(p)), feature)


# ---------------------------------------------------------------------------
# Admission features
# ---------------------------------------------------------------------------

def extract_hour0(X: np.ndarray) -> pd.DataFrame:
    """Hour-0 value of every label, one row per patient (index: case_admission_id)."""
    return pd.DataFrame(X[:, 0, :, VALUE_COL].astype(float),
                        index=pd.Index(X[:, 0, 0, CID_COL], name='case_admission_id'),
                        columns=list(X[0, 0, :, LABEL_COL]))


def first_measured_imaging(features_long: pd.DataFrame, window_h: int = ADMISSION_IMAGING_WINDOW_H) -> pd.DataFrame:
    """
    Long preprocessed features -> first measured (non-imputed) imaging value per patient and label within window.
    Returns columns case_admission_id, sample_label, relative_sample_date_hourly_cat, value.
    """
    df = features_long[features_long.sample_label.isin(IMAGING_LABELS)
                       & ~features_long.source.str.contains(IMPUTED_SOURCE_TAG)
                       & (features_long.relative_sample_date_hourly_cat < window_h)]
    df = df.sort_values('relative_sample_date_hourly_cat')
    return df.drop_duplicates(subset=['case_admission_id', 'sample_label'])[
        ['case_admission_id', 'sample_label', 'relative_sample_date_hourly_cat', 'value']].reset_index(drop=True)


def load_admission_imaging(features_csv_path: str) -> pd.DataFrame:
    """Chunked read of a preprocessed_features_*.csv (> 1 GB), imaging rows only."""
    columns = ['relative_sample_date_hourly_cat', 'case_admission_id', 'sample_label', 'source', 'value']
    chunks = [first_measured_imaging(chunk)
              for chunk in pd.read_csv(features_csv_path, usecols=columns, chunksize=CSV_CHUNK_ROWS)]
    # a patient may span chunks -> reduce again
    return first_measured_imaging(pd.concat(chunks).assign(source=''))


def extract_admission_features(X: np.ndarray, admission_imaging: pd.DataFrame) -> pd.DataFrame:
    """Hour-0 values, imaging replaced by the first measured admission value when available."""
    admission = extract_hour0(X)
    admission = admission.drop(columns=[c for c in admission.columns if c.startswith(HOUR0_REDUNDANT_PREFIXES)])

    imaging = admission_imaging.pivot(index='case_admission_id', columns='sample_label', values='value')
    imaging = imaging.reindex(index=admission.index, columns=[c for c in IMAGING_LABELS if c in admission.columns])
    admission.update(imaging)
    return admission


def count_late_imaging(admission_imaging: pd.DataFrame, cids) -> int:
    """Patients (among cids) with at least one imaging label first measured after hour 0."""
    df = admission_imaging[admission_imaging.case_admission_id.isin(set(cids))]
    latest_first = df.groupby('case_admission_id').relative_sample_date_hourly_cat.max()
    return int((latest_first > 0).sum())


# ---------------------------------------------------------------------------
# Binary rescaling: a few binaries were z-scored in preprocessing -> back to 0/1 for interpretable ORs
# ---------------------------------------------------------------------------

def fit_binary_rescaling(train: pd.DataFrame) -> dict:
    """column -> (low, high) for 2-valued columns not already coded 0/1."""
    params = {}
    for column in train.columns:
        values = np.unique(train[column].dropna())
        if len(values) != 2 or set(values) == {0, 1}:
            continue
        params[column] = (values[0], values[1])
    return params


def apply_binary_rescaling(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    for column, (low, high) in params.items():
        df[column] = (df[column] - low) / (high - low)
    return df


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def hourly_labels(cids, y_df: pd.DataFrame, n_timepoints: int, horizon: int = PREDICTION_HORIZON_H) -> pd.DataFrame:
    """(case_admission_id, ts, y_true): END in bins ts+1..ts+horizon, as aggregate_and_label_timeseries."""
    ts = np.arange(n_timepoints)
    events = y_df.groupby('case_admission_id')[EVENT_BIN].apply(np.asarray)
    frames = []
    for cid in cids:
        bins = events.get(cid, np.array([]))
        labels = ((bins[None, :] > ts[:, None]) & (bins[None, :] <= ts[:, None] + horizon)).any(axis=1)
        frames.append(pd.DataFrame({'case_admission_id': cid, TIMESTEP: ts, 'y_true': labels.astype(int)}))
    return pd.concat(frames, ignore_index=True)


def patient_labels(hourly: pd.DataFrame) -> pd.Series:
    """case_admission_id -> 1 if any hourly label is positive."""
    return hourly.groupby('case_admission_id').y_true.max()


def predictable_events(y_df: pd.DataFrame) -> pd.DataFrame:
    """END events after hour 0 (an event in bin 0 has no prior prediction)."""
    return y_df[y_df[EVENT_BIN] >= 1]
