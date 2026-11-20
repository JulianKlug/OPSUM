"""
Recover each patient's time zero (t0) without the stroke registry / imaging files.

The preprocessed features keep, per patient, the hourly bins holding a real
(non-imputed) EHR measurement. Shifting the raw timestamps by a candidate t0
must reproduce those bins:

    raw times       |  x   x  x     x    x  |
    candidate t0 -> [ bin0 | bin1 | bin2 | ...]  compare with preprocessed bins
                                                (Jaccard over (label, bin) pairs)

Offsets (1-min grid around the anchor t0; whole raw span as fallback) are
scored vectorised; the best-scoring interval's midpoint is used as t0.
Stored first_sample_date (END patients) is exact and takes precedence.
"""
from enum import Enum

import numpy as np
import pandas as pd

N_BINS = 72
SEARCH_BEFORE_MIN = 48 * 60
SEARCH_AFTER_MIN = 2 * 60
# fallback when t0 lies days after the first raw sample (e.g. in-hospital stroke, late admission)
WIDE_STEP_MIN = 10
REFINE_RADIUS_MIN = 2 * WIDE_STEP_MIN
OFFSET_CHUNK = 2000  # offsets scored per batch (bounds memory)
MIN_JACCARD = 0.9
EHR_SOURCE = 'EHR'
DATETIME_FORMAT = '%d.%m.%Y %H:%M'

# preprocessed label -> raw label
ALIGNMENT_LABELS = {
    'median_NIHSS': 'NIHSS',
    'median_heart_rate': 'heart_rate',
    'median_systolic_blood_pressure': 'systolic_blood_pressure',
    'median_oxygen_saturation': 'oxygen_saturation',
    'median_respiratory_rate': 'respiratory_rate',
}


class T0Source(str, Enum):
    STORED = 'stored_first_sample_date'
    ALIGNED = 'aligned'
    UNRESOLVED = 'unresolved'


def load_preprocessed_bins(features_path: str) -> pd.DataFrame:
    """case_admission_id, label (raw name), bin: hours with a real EHR measurement."""
    df = pd.read_csv(features_path, usecols=['relative_sample_date_hourly_cat', 'case_admission_id',
                                             'sample_label', 'source'])
    df = df[(df.source == EHR_SOURCE) & df.sample_label.isin(ALIGNMENT_LABELS)
            & (df.relative_sample_date_hourly_cat < N_BINS)]
    df['label'] = df.sample_label.map(ALIGNMENT_LABELS)
    df['bin'] = df.relative_sample_date_hourly_cat.astype(int)
    return df[['case_admission_id', 'label', 'bin']]


def _raw_samples(raw: dict) -> pd.DataFrame:
    vitals = raw['vitals'][['case_admission_id', 'datetime', 'vital_name']].rename(columns={'vital_name': 'label'})
    nihss = raw['nihss'].assign(label='NIHSS')
    samples = pd.concat([vitals, nihss], ignore_index=True)
    samples = samples[samples.label.isin(ALIGNMENT_LABELS.values())]
    samples['datetime'] = pd.to_datetime(samples['datetime'], format=DATETIME_FORMAT)
    return samples


def _score_offsets(minutes: np.ndarray, label_ids: np.ndarray, target: np.ndarray,
                   offsets: np.ndarray, n_codes: int) -> np.ndarray:
    """Jaccard between raw (label, bin) set and target set, for every offset (vectorised)."""
    bins = np.floor((minutes[None, :] - offsets[:, None]) / 60).astype(int)
    in_range = (bins >= 0) & (bins < N_BINS)
    codes = np.where(in_range, label_ids[None, :] * N_BINS + bins, n_codes)  # n_codes = out-of-range sink

    present = np.zeros((len(offsets), n_codes + 1), bool)
    present[np.arange(len(offsets))[:, None], codes] = True
    present = present[:, :n_codes]

    intersection = (present & target).sum(1)
    union = (present | target).sum(1)
    return intersection / np.maximum(union, 1)


def _best_offsets(minutes, label_ids, target, offsets, n_codes):
    """Returns (best Jaccard, offsets reaching it), scoring offsets in chunks."""
    scores = np.concatenate([_score_offsets(minutes, label_ids, target, offsets[i:i + OFFSET_CHUNK], n_codes)
                             for i in range(0, len(offsets), OFFSET_CHUNK)])
    return scores.max(), offsets[scores == scores.max()]


def _align_patient(minutes, label_ids, target, n_codes):
    """
    1. fine search around the first sample (-48 h .. +2 h, 1-min grid)
    2. if not resolved: coarse search over the whole raw span, then 1-min refinement around the best hit
    """
    narrow = np.arange(-SEARCH_BEFORE_MIN, SEARCH_AFTER_MIN + 1)
    best_score, best = _best_offsets(minutes, label_ids, target, narrow, n_codes)
    if best_score >= MIN_JACCARD:
        return best_score, best

    wide = np.arange(-SEARCH_BEFORE_MIN, int(minutes.max()) + 1, WIDE_STEP_MIN)
    _, coarse = _best_offsets(minutes, label_ids, target, wide, n_codes)
    refine = np.unique(np.concatenate([np.arange(o - REFINE_RADIUS_MIN, o + REFINE_RADIUS_MIN + 1) for o in coarse]))
    wide_score, wide_best = _best_offsets(minutes, label_ids, target, refine, n_codes)
    if wide_score > best_score:
        return wide_score, wide_best
    return best_score, best


def align_t0(raw: dict, anchor_t0: pd.DataFrame, prepro_bins: pd.DataFrame, stored_t0: pd.DataFrame) -> pd.DataFrame:
    """
    anchor_t0: case_admission_id, t0 (first EHR sample; search centre)
    stored_t0: case_admission_id, first_sample_date (exact, END patients)
    Returns: case_admission_id, t0, jaccard, interval_min, t0_source
    """
    label_index = {label: i for i, label in enumerate(ALIGNMENT_LABELS.values())}
    n_codes = len(label_index) * N_BINS

    samples = _raw_samples(raw)
    samples['label_id'] = samples.label.map(label_index)
    prepro_bins = prepro_bins.assign(code=prepro_bins.label.map(label_index) * N_BINS + prepro_bins.bin)
    target_codes = prepro_bins.groupby('case_admission_id').code.apply(np.unique)
    anchors = anchor_t0.set_index('case_admission_id').t0

    rows = []
    for cid, subj in samples.groupby('case_admission_id'):
        if cid not in anchors.index:
            continue
        target = np.zeros(n_codes, bool)
        if cid in target_codes.index:
            target[target_codes[cid]] = True

        minutes = ((subj.datetime - anchors[cid]) / pd.Timedelta(minutes=1)).values
        jaccard, best = _align_patient(minutes, subj.label_id.values, target, n_codes)

        rows.append({'case_admission_id': cid,
                     't0': anchors[cid] + pd.Timedelta(minutes=(best.min() + best.max()) / 2),
                     'jaccard': jaccard, 'interval_min': best.max() - best.min()})

    aligned = pd.DataFrame(rows)
    aligned['t0_source'] = np.where(aligned.jaccard >= MIN_JACCARD, T0Source.ALIGNED.value,
                                    T0Source.UNRESOLVED.value)

    # stored t0 is exact -> overrides alignment
    stored = stored_t0.drop_duplicates('case_admission_id').set_index('case_admission_id').first_sample_date
    stored = pd.to_datetime(stored)
    aligned['stored_t0'] = aligned.case_admission_id.map(stored)
    aligned['stored_minus_aligned_min'] = (aligned.stored_t0 - aligned.t0) / pd.Timedelta(minutes=1)
    has_stored = aligned.stored_t0.notna()
    aligned.loc[has_stored, 't0'] = aligned.loc[has_stored, 'stored_t0']
    aligned.loc[has_stored, 't0_source'] = T0Source.STORED.value
    return aligned
