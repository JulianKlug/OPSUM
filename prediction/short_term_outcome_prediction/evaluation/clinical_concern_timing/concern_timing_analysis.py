"""
Timing of END model alerts relative to proxies of clinical concern (reviewer response).

Hour-bin convention (as in aggregate_and_label_timeseries):
    prediction at ts uses bins 0..ts, label = END in bins ts+1..ts+6
    -> marker in bin C is available to the model from ts = C onwards
    -> alert at A "precedes concern" iff A < C

    t0      A (alert)        C (concern)   E (END)
    |-------|================|-------------|------> hours
            <---- C - A ----><--- E - C --->
            <------------- E - A ---------->

Steps:
    0. t0              (stored for END patients, else aligned to preprocessed bins; t0_alignment.py)
    1. concern markers (concern_markers.py)
    2. hourly predictions on the test set
    3. per-event lead times
    4. metrics on pre-concern timepoints vs full test set
"""
import argparse
import json
import os
import pickle
from enum import Enum

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as ch
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.concern_markers import \
    load_raw_measurements, compute_t0, compute_markers, Marker
from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.t0_alignment import \
    align_t0, load_preprocessed_bins, T0Source
from prediction.short_term_outcome_prediction.testing.test_xgb import _compute_metrics
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.utils import ensure_dir

PREDICTION_HORIZON_H = 6
N_BOOTSTRAP = 1000
SEED = 42
DEFAULT_THRESHOLD = 0.5
PRIMARY_THRESHOLD = 'youden_j'
ANY_MARKER = 'any'


class Timing(str, Enum):
    BEFORE = 'alert_before_concern'
    SAME_HOUR = 'same_hour'
    AFTER = 'alert_after_concern'
    NO_MARKER = 'no_marker_before_end'
    NOT_DETECTED = 'not_detected'


# ---------------------------------------------------------------------------
# Step 2: predictions
# ---------------------------------------------------------------------------

def predict_hourly(X: np.ndarray, y_df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    """Same pipeline as test_final_model (testing/test_xgb.py), kept per patient and timestep."""
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'xgb_final_model.model'))
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'final_model_config.json')) as f:
        config = json.load(f)

    data_list, labels_list = aggregate_and_label_timeseries(
        X, y_df,
        target_time_to_outcome=PREDICTION_HORIZON_H,
        target_interval=config.get('target_interval', True),
        restrict_to_first_event=config.get('restrict_to_first_event', False),
        add_lag_features=config.get('add_lag_features', False),
        add_rolling_features=config.get('add_rolling_features', False),
    )

    cids = X[:, 0, 0, 0]
    frames = []
    for cid, data, labels in zip(cids, data_list, labels_list):
        prob = model.predict_proba(scaler.transform(data))[:, 1]
        n = len(labels)
        frames.append(pd.DataFrame({'case_admission_id': cid, 'ts': np.arange(n),
                                    'y_true': labels, 'y_prob': prob[:n]}))
    return pd.concat(frames, ignore_index=True)


def load_thresholds(threshold_path: str) -> dict:
    thresholds = {'default_0.5': DEFAULT_THRESHOLD}
    with open(threshold_path) as f:
        for name, info in json.load(f).get('median_thresholds', {}).items():
            thresholds[name] = info['median_threshold']
    return thresholds


# ---------------------------------------------------------------------------
# Step 3: lead times
# ---------------------------------------------------------------------------

def _classify(alert, concern) -> Timing:
    if np.isnan(alert):
        return Timing.NOT_DETECTED
    if np.isnan(concern):
        return Timing.NO_MARKER
    if alert < concern:
        return Timing.BEFORE
    if alert == concern:
        return Timing.SAME_HOUR
    return Timing.AFTER


def event_lead_times(preds: pd.DataFrame, y_df: pd.DataFrame, markers: pd.DataFrame,
                     threshold: float, marker_type: str) -> pd.DataFrame:
    """
    One row per END event (bin E >= 1):
      A = first ts in [E-6, E-1] with prob >= threshold
      C = first marker bin in (previous END bin, E]
    """
    if marker_type != ANY_MARKER:
        markers = markers[markers.marker == marker_type]
    marker_bins = markers.assign(bin=np.floor(markers.relative_hour).astype(int))

    rows = []
    for cid, events in y_df.groupby('case_admission_id'):
        event_bins = np.sort(events.relative_sample_date_hourly_cat.astype(int).unique())
        subj_preds = preds[preds.case_admission_id == cid].set_index('ts').y_prob
        subj_markers = marker_bins[marker_bins.case_admission_id == cid].bin.values
        previous = -1

        for E in event_bins:
            # an event in bin 0 cannot be predicted (no preceding timestep)
            if E < 1:
                previous = E
                continue

            window = subj_preds.loc[max(0, E - PREDICTION_HORIZON_H):E - 1]
            alerts = window.index[window >= threshold]
            A = float(alerts.min()) if len(alerts) else np.nan

            in_episode = subj_markers[(subj_markers > previous) & (subj_markers <= E)]
            C = float(in_episode.min()) if len(in_episode) else np.nan

            rows.append({'case_admission_id': cid, 'end_bin': E, 'alert_bin': A, 'concern_bin': C,
                         'alert_to_end_h': E - A, 'alert_to_concern_h': C - A, 'concern_to_end_h': E - C,
                         'timing': _classify(A, C).value})
            previous = E

    return pd.DataFrame(rows)


def summarise_lead_times(events: pd.DataFrame) -> dict:
    detected = events[events.timing != Timing.NOT_DETECTED.value]
    with_marker = detected[detected.timing != Timing.NO_MARKER.value]

    def q(s):
        return f'{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})' if len(s) else 'NA'

    summary = {'n_events': len(events), 'n_detected': len(detected),
               'pct_detected': 100 * len(detected) / len(events) if len(events) else np.nan,
               'n_events_without_marker_before_end': int(events.concern_bin.isna().sum()),
               'alert_to_end_h_median_iqr': q(detected.alert_to_end_h),
               'alert_to_concern_h_median_iqr': q(with_marker.alert_to_concern_h),
               'concern_to_end_h_median_iqr': q(events.concern_to_end_h.dropna())}
    for timing in Timing:
        n = int((events.timing == timing.value).sum())
        summary[f'n_{timing.value}'] = n
        if timing == Timing.NOT_DETECTED:
            continue
        summary[f'pct_of_detected_{timing.value}'] = 100 * n / len(detected) if len(detected) else np.nan
    return summary


# ---------------------------------------------------------------------------
# Step 4: pre-concern evaluation
# ---------------------------------------------------------------------------

def restrict_to_pre_concern(preds: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """Keep timesteps ts < first marker bin (patients without markers keep all)."""
    first = np.floor(markers.groupby('case_admission_id').relative_hour.min()).rename('first_marker_bin')
    df = preds.merge(first, left_on='case_admission_id', right_index=True, how='left')
    keep = df.first_marker_bin.isna() | (df.ts < df.first_marker_bin)
    return df[keep].drop(columns='first_marker_bin')


def _metrics_with_ci(preds: pd.DataFrame, thresholds: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Point estimates + patient-level bootstrap 95% CI."""
    groups = {cid: g for cid, g in preds.groupby('case_admission_id')}
    cids = np.array(list(groups))

    point = {name: _compute_metrics(preds.y_true.values, preds.y_prob.values, thr)
             for name, thr in thresholds.items()}

    samples = {name: [] for name in thresholds}
    for _ in range(N_BOOTSTRAP):
        bs = pd.concat([groups[c] for c in rng.choice(cids, len(cids), replace=True)])
        if bs.y_true.nunique() < 2:
            continue
        auroc = roc_auc_score(bs.y_true, bs.y_prob)
        for name, thr in thresholds.items():
            samples[name].append(_compute_metrics(bs.y_true.values, bs.y_prob.values, thr, auroc=auroc))

    rows = []
    for name, thr in thresholds.items():
        bs_df = pd.DataFrame(samples[name])
        for metric, value in point[name].items():
            rows.append({'threshold_name': name, 'threshold': thr, 'metric': metric, 'value': value,
                         'ci_low': bs_df[metric].quantile(0.025), 'ci_high': bs_df[metric].quantile(0.975)})
    return pd.DataFrame(rows)


def compare_full_vs_pre_concern(preds: pd.DataFrame, markers: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    pre = restrict_to_pre_concern(preds, markers)
    full_df = _metrics_with_ci(preds, thresholds, rng).assign(subset='full')
    pre_df = _metrics_with_ci(pre, thresholds, rng).assign(subset='pre_concern')
    counts = pd.DataFrame([
        {'subset': 'full', 'n_timepoints': len(preds), 'n_positive': int(preds.y_true.sum())},
        {'subset': 'pre_concern', 'n_timepoints': len(pre), 'n_positive': int(pre.y_true.sum())}])
    return pd.concat([full_df, pre_df]).merge(counts, on='subset')


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_lead_times(events: pd.DataFrame, path: str):
    with_marker = events.dropna(subset=['alert_bin', 'concern_bin'])
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(with_marker.alert_to_concern_h.min() - 0.5, with_marker.alert_to_concern_h.max() + 1.5)
    ax.hist(with_marker.alert_to_concern_h, bins=bins, color='#4C72B0', edgecolor='white')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Concern marker − first alert (h)\n(> 0: alert precedes concern)')
    ax.set_ylabel('END events')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(test_data_path: str, model_dir: str, raw_dir: str, features_path: str, output_dir: str):
    ensure_dir(output_dir)
    X, y_df = ch.load(test_data_path, weights_only=False)
    cids = set(X[:, 0, 0, 0])

    # 0. t0
    raw = load_raw_measurements(raw_dir, cids)
    t0_df = align_t0(raw, compute_t0(raw), load_preprocessed_bins(features_path),
                     y_df[['case_admission_id', 'first_sample_date']])
    t0_df.to_csv(os.path.join(output_dir, 't0_validation.csv'), index=False)
    valid = t0_df[t0_df.t0_source != T0Source.UNRESOLVED.value]
    valid_cids = set(valid.case_admission_id)
    print(f'Patients: {len(cids)}, t0 resolved: {len(valid_cids)}', t0_df.t0_source.value_counts().to_dict())
    stored = t0_df.dropna(subset=['stored_t0'])
    print(f'Stored t0 within aligned interval (+-1 min): '
          f'{(stored.stored_minus_aligned_min.abs() <= stored.interval_min / 2 + 1).mean():.1%} of {len(stored)}')

    # 1. markers
    markers = compute_markers(raw, valid[['case_admission_id', 't0']])
    markers = markers[markers.case_admission_id.isin(valid_cids)]
    markers.to_csv(os.path.join(output_dir, 'concern_markers.csv'), index=False)

    # 2. predictions
    preds = predict_hourly(X, y_df, model_dir)
    full_auroc = roc_auc_score(preds.y_true, preds.y_prob)
    print(f'Full test AUROC (all patients): {full_auroc:.4f}')
    preds = preds[preds.case_admission_id.isin(valid_cids)]
    y_valid = y_df[y_df.case_admission_id.isin(valid_cids)]

    # 3. lead times
    threshold_path = next(os.path.join(model_dir, f) for f in os.listdir(model_dir)
                          if f.startswith('threshold_tuning_results'))
    thresholds = load_thresholds(threshold_path)
    all_events, summaries = [], []
    for thr_name, thr in thresholds.items():
        for marker_type in [ANY_MARKER] + [m.value for m in Marker]:
            ev = event_lead_times(preds, y_valid, markers, thr, marker_type)
            all_events.append(ev.assign(threshold_name=thr_name, threshold=thr, marker_type=marker_type))
            summaries.append({'threshold_name': thr_name, 'threshold': thr, 'marker_type': marker_type,
                              **summarise_lead_times(ev)})
    events_df = pd.concat(all_events, ignore_index=True)
    events_df.to_csv(os.path.join(output_dir, 'per_event_lead_times.csv'), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(output_dir, 'lead_time_summary.csv'), index=False)

    primary = events_df[(events_df.threshold_name == PRIMARY_THRESHOLD) & (events_df.marker_type == ANY_MARKER)]
    plot_lead_times(primary, os.path.join(output_dir, 'alert_to_concern_lead_time.png'))

    # 4. pre-concern evaluation
    metrics = compare_full_vs_pre_concern(preds, markers, thresholds)
    metrics.to_csv(os.path.join(output_dir, 'pre_concern_vs_full_metrics.csv'), index=False)
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_data_path', required=True)
    parser.add_argument('-m', '--model_dir', required=True)
    parser.add_argument('-r', '--raw_dir', required=True)
    parser.add_argument('-f', '--features_path', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()
    run(args.test_data_path, args.model_dir, args.raw_dir, args.features_path, args.output_dir)
