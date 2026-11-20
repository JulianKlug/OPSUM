"""
Proxies for documented clinical concern, derived from the raw GSU EHR extraction.

Markers (relative hours from t0):
- NIHSS_SURGE: an NIHSS assessment whose interval to the previous one is
  < SURGE_INTERVAL_RATIO x the patient's running median interval
  (e.g. routine q4h, then a check 1h later -> surge).
- FIO2_RISE: an FiO2 value >= FIO2_RISE_PCT points above the previous one
  (e.g. 21% -> 25%, i.e. +1 L/min O2).

Layers:
    raw csv (Extraction_YYYYMMDD)
        -> load_raw_measurements()   [reuses GSU preprocessing functions]
        -> compute_t0()              [first non-FiO2 EHR sample]
        -> detect_*()                [pure functions on (cid, relative_hour)]
"""
import os
from enum import Enum

import numpy as np
import pandas as pd

from preprocessing.geneva_stroke_unit_preprocessing.patient_selection.filter_ehr_patients import filter_ehr_patients
from preprocessing.geneva_stroke_unit_preprocessing.scales_preprocessing.scales_preprocessing import preprocess_scales
from preprocessing.geneva_stroke_unit_preprocessing.utils import create_ehr_case_identification_column
from preprocessing.geneva_stroke_unit_preprocessing.ventilation_preprocessing.ventilation_preprocessing import \
    preprocess_ventilation
from preprocessing.geneva_stroke_unit_preprocessing.vitals_preprocessing.vitals_preprocessing import preprocess_vitals

SURGE_INTERVAL_RATIO = 0.5
MIN_PRIOR_ASSESSMENTS = 3
FIO2_RISE_PCT = 4
TIME_RANGE_H = 72

DATETIME_FORMAT = '%d.%m.%Y %H:%M'
NO_PATIENT_SELECTION = ''
RAW_DELIMITER = ';'
CHUNK_SIZE = 500_000

SCALE_FILE_START = 'scale'
VENTILATION_FILE_START = 'ventilation'
VITALS_FILE_START = 'patientvalue'
LAB_FILE_START = 'labo'
EDS_FILE = 'eds_j1.csv'

NIHSS = 'NIHSS'
FIO2 = 'FIO2'
# scales among selected variables -> contribute to t0
T0_SCALES = [NIHSS, 'Glasgow Coma Scale']


class Marker(str, Enum):
    NIHSS_SURGE = 'nihss_surge'
    FIO2_RISE = 'fio2_rise'


# ---------------------------------------------------------------------------
# Marker detection (pure, testable)
# ---------------------------------------------------------------------------

def detect_nihss_ascertainment_surge(nihss_df: pd.DataFrame) -> pd.DataFrame:
    """
    nihss_df: case_admission_id, relative_hour (one row per NIHSS assessment)
    Returns: case_admission_id, marker, relative_hour (one row per surge assessment)
    """
    rows = []
    for cid, subj in nihss_df.groupby('case_admission_id'):
        # duplicates at the same minute are one assessment
        times = np.sort(subj['relative_hour'].unique())
        intervals = np.diff(times)  # intervals[i - 1] = times[i] - times[i - 1]

        for i in range(MIN_PRIOR_ASSESSMENTS, len(times)):
            # median of intervals among prior assessments only (no look-ahead)
            prior_median = np.median(intervals[:i - 1])
            if intervals[i - 1] >= SURGE_INTERVAL_RATIO * prior_median:
                continue
            rows.append((cid, Marker.NIHSS_SURGE.value, times[i]))

    return pd.DataFrame(rows, columns=['case_admission_id', 'marker', 'relative_hour'])


def detect_fio2_rise(fio2_df: pd.DataFrame) -> pd.DataFrame:
    """
    fio2_df: case_admission_id, relative_hour, fio2 (%)
    Returns: case_admission_id, marker, relative_hour (one row per rise)
    """
    rows = []
    for cid, subj in fio2_df.groupby('case_admission_id'):
        # simultaneous records (ventilation + vitals tables) -> one value per time
        subj = subj.groupby('relative_hour', as_index=False).fio2.max()
        # first value has no predecessor -> never a rise
        rises = subj['fio2'].diff() >= FIO2_RISE_PCT
        for hour in subj.loc[rises, 'relative_hour']:
            rows.append((cid, Marker.FIO2_RISE.value, hour))

    return pd.DataFrame(rows, columns=['case_admission_id', 'marker', 'relative_hour'])


# ---------------------------------------------------------------------------
# Raw data access (wraps GSU preprocessing)
# ---------------------------------------------------------------------------

def _read_filtered(raw_dir: str, file_start: str, keep_row) -> pd.DataFrame:
    """Read all raw files starting with file_start, keeping only rows where keep_row(chunk) is True."""
    parts = []
    for f in sorted(os.listdir(raw_dir)):
        if not f.startswith(file_start):
            continue
        for chunk in pd.read_csv(os.path.join(raw_dir, f), delimiter=RAW_DELIMITER, encoding='utf-8-sig',
                                 dtype=str, chunksize=CHUNK_SIZE):
            parts.append(chunk[keep_row(chunk)])
    return pd.concat(parts, ignore_index=True)


def _keep_cids(cids: set):
    return lambda chunk: create_ehr_case_identification_column(chunk).isin(cids)


def _keep_eds_suffixes(cids: set):
    # scale patient_ids may be overwritten -> prefilter on EDS suffix only, exact filter after correction
    suffixes = {cid.split('_')[-1] for cid in cids}
    return lambda chunk: chunk['eds_end_4digit'].str.zfill(4).isin(suffixes)


def _drop_before_onset_day(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    # mirrors restrict_to_patient_selection: samples before the day of stroke onset are excluded
    onset_day = pd.to_datetime(df['stroke_onset_date'], format=DATETIME_FORMAT).dt.normalize()
    sample_dt = pd.to_datetime(df[date_col], format=DATETIME_FORMAT)
    return df[onset_day.isna() | (sample_dt >= onset_day)]


def load_raw_measurements(raw_dir: str, cids: set) -> dict:
    """
    Returns timestamped EHR measurements (absolute datetimes) for the given case_admission_ids:
      nihss:     case_admission_id, datetime
      t0_scales: case_admission_id, datetime (NIHSS + GCS)
      vitals:  case_admission_id, datetime, vital_name, vital_value
      labs:    case_admission_id, datetime
      onset:   case_admission_id, stroke_onset_date
      ventilation: raw ventilation rows (FiO2 preprocessing needs t0)
    """
    eds_df = pd.read_csv(os.path.join(raw_dir, EDS_FILE), delimiter=RAW_DELIMITER, encoding='utf-8-sig', dtype=str)

    # NIHSS
    scales_df = _read_filtered(raw_dir, SCALE_FILE_START, _keep_eds_suffixes(cids))
    onset_df = scales_df[['patient_id', 'eds_end_4digit', 'stroke_onset_date']].copy()
    scales_df = preprocess_scales(scales_df, eds_df, NO_PATIENT_SELECTION)
    scales_df = scales_df[scales_df.case_admission_id.isin(cids)]
    scales_df = scales_df[scales_df.scale.isin(T0_SCALES)][['case_admission_id', 'event_date', 'scale']]
    scales_df = scales_df.rename(columns={'event_date': 'datetime'})
    nihss = scales_df[scales_df.scale == NIHSS][['case_admission_id', 'datetime']]
    t0_scales = scales_df[['case_admission_id', 'datetime']]

    # Vitals
    vitals_df = _read_filtered(raw_dir, VITALS_FILE_START, _keep_cids(cids))
    vitals_df = _drop_before_onset_day(vitals_df, 'datetime')
    vitals_df = filter_ehr_patients(vitals_df, NO_PATIENT_SELECTION)
    vitals_df = preprocess_vitals(vitals_df)[['case_admission_id', 'datetime', 'vital_name', 'vital_value']]

    # Labs
    lab_df = _read_filtered(raw_dir, LAB_FILE_START, _keep_cids(cids))
    lab_df = _drop_before_onset_day(lab_df, 'sample_date')
    lab_df = filter_ehr_patients(lab_df, NO_PATIENT_SELECTION)
    # preprocess_labs needs equivalent_labels.csv (not available) -> raw sample times; checked by t0 validation
    lab_df['case_admission_id'] = create_ehr_case_identification_column(lab_df)
    lab_df = lab_df[['case_admission_id', 'sample_date']]
    lab_df = lab_df.rename(columns={'sample_date': 'datetime'})

    # Ventilation (raw, FiO2 preprocessing needs t0 -> done in load_fio2)
    ventilation_df = _read_filtered(raw_dir, VENTILATION_FILE_START, _keep_cids(cids))
    ventilation_df = filter_ehr_patients(ventilation_df, NO_PATIENT_SELECTION)

    onset_df['case_admission_id'] = create_ehr_case_identification_column(onset_df)
    onset_df = onset_df[onset_df.case_admission_id.isin(cids)][['case_admission_id', 'stroke_onset_date']]

    return {'nihss': nihss, 't0_scales': t0_scales, 'vitals': vitals_df, 'labs': lab_df, 'ventilation': ventilation_df,
            'onset': onset_df.drop_duplicates('case_admission_id')}


def compute_t0(raw: dict) -> pd.DataFrame:
    """
    t0 = first non-FiO2 EHR sample (labs, scales, vitals), on or after the day of stroke onset.
    Approximates relative_timestamps.py without the registry fallback.
    Returns: case_admission_id, t0 (datetime)
    """
    scales = _drop_onset(raw['t0_scales'], raw['onset'])
    vitals = raw['vitals'][raw['vitals'].vital_name != FIO2][['case_admission_id', 'datetime']]
    samples = pd.concat([scales, vitals, raw['labs']], ignore_index=True)
    samples['datetime'] = pd.to_datetime(samples['datetime'], format=DATETIME_FORMAT)
    t0 = samples.groupby('case_admission_id').datetime.min().rename('t0').reset_index()
    return t0


def _drop_onset(nihss: pd.DataFrame, onset: pd.DataFrame) -> pd.DataFrame:
    df = nihss.merge(onset, on='case_admission_id', how='left')
    return _drop_before_onset_day(df, 'datetime')[['case_admission_id', 'datetime']]


def _to_relative_hours(df: pd.DataFrame, t0: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(t0, on='case_admission_id')
    df['relative_hour'] = (pd.to_datetime(df['datetime'], format=DATETIME_FORMAT) - df['t0']) / pd.Timedelta(hours=1)
    return df[(df.relative_hour >= 0) & (df.relative_hour <= TIME_RANGE_H)]


def relative_nihss(raw: dict, t0: pd.DataFrame) -> pd.DataFrame:
    return _to_relative_hours(_drop_onset(raw['nihss'], raw['onset']), t0)[['case_admission_id', 'relative_hour']]


def relative_vitals(raw: dict, t0: pd.DataFrame) -> pd.DataFrame:
    return _to_relative_hours(raw['vitals'], t0)[['case_admission_id', 'vital_name', 'relative_hour']]


def relative_fio2(raw: dict, t0: pd.DataFrame) -> pd.DataFrame:
    """
    Recorded FiO2 (%) from ventilation + vitals tables, as in variable_database_assembly.
    No first sample dates passed -> preprocess_ventilation adds no inferred room-air rows (they cannot rise).
    """
    no_first_sample_dates = pd.DataFrame(columns=['case_admission_id', 'first_sample_date'])
    ventilation_fio2, _ = preprocess_ventilation(raw['ventilation'].copy(), no_first_sample_dates)
    ventilation_fio2 = ventilation_fio2[['case_admission_id', 'datetime', 'FIO2']].rename(columns={'FIO2': 'fio2'})

    vitals_fio2 = raw['vitals'][raw['vitals'].vital_name == FIO2]
    vitals_fio2 = vitals_fio2[['case_admission_id', 'datetime', 'vital_value']].rename(columns={'vital_value': 'fio2'})

    recorded = pd.concat([ventilation_fio2, vitals_fio2], ignore_index=True)
    recorded['fio2'] = recorded['fio2'].astype(float)
    return _to_relative_hours(recorded, t0)[['case_admission_id', 'relative_hour', 'fio2']]


def compute_markers(raw: dict, t0: pd.DataFrame) -> pd.DataFrame:
    """All concern markers within [0, TIME_RANGE_H] h. Columns: case_admission_id, marker, relative_hour."""
    nihss_markers = detect_nihss_ascertainment_surge(relative_nihss(raw, t0))
    fio2_markers = detect_fio2_rise(relative_fio2(raw, t0))
    return pd.concat([nihss_markers, fio2_markers], ignore_index=True)
