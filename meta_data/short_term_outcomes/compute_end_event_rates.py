import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from prediction.short_term_outcome_prediction.timeseries_decomposition import decompose_and_label_timeseries

DATA_DIR = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047'
TRAIN_PATH = os.path.join(DATA_DIR, 'train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth')
TEST_PATH = os.path.join(DATA_DIR, 'test_data_early_neurological_deterioration_ts0.8_rs42_ns5.pth')

TARGET_TIME_TO_OUTCOME = 6


N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def _per_patient_stats(X, y_df):
    """Compute per-patient statistics needed for bootstrapping.

    Returns a DataFrame indexed by patient position (0..n_patients-1) with:
        n_timesteps, n_positive, n_events, has_event
    """
    cids = X[:, 0, 0, 0]

    idx_map, flat_labels = decompose_and_label_timeseries(
        X, y_df,
        target_time_to_outcome=TARGET_TIME_TO_OUTCOME,
        target_interval=True,
        restrict_to_first_event=False,
    )
    flat_labels = np.array(flat_labels)

    # Group timestep-level labels by patient index
    patient_n_ts = np.zeros(len(cids), dtype=int)
    patient_n_pos = np.zeros(len(cids), dtype=int)
    for i, (cid_idx, _ts) in enumerate(idx_map):
        patient_n_ts[cid_idx] += 1
        patient_n_pos[cid_idx] += int(flat_labels[i])

    # Events per patient from y_df
    event_cids = set(y_df.case_admission_id.values)
    patient_n_events = np.zeros(len(cids), dtype=int)
    patient_has_event = np.zeros(len(cids), dtype=int)
    for idx, cid in enumerate(cids):
        if cid in event_cids:
            patient_has_event[idx] = 1
            patient_n_events[idx] = len(y_df[y_df.case_admission_id == cid])

    return pd.DataFrame({
        'n_timesteps': patient_n_ts,
        'n_positive': patient_n_pos,
        'n_events': patient_n_events,
        'has_event': patient_has_event,
    })


def _metrics_from_patient_stats(ps):
    """Compute aggregate metrics from a (possibly resampled) patient stats DataFrame."""
    n_patients = len(ps)
    total_ts = ps['n_timesteps'].sum()
    n_positive = ps['n_positive'].sum()
    n_negative = total_ts - n_positive
    patients_with_event = ps['has_event'].sum()
    n_distinct_events = ps['n_events'].sum()

    return {
        'N patients': int(n_patients),
        'Total patient-hours': int(total_ts),
        'Patients with >= 1 END event': int(patients_with_event),
        'Distinct END events': int(n_distinct_events),
        'Positive timesteps (label=1)': int(n_positive),
        'Negative timesteps (label=0)': int(n_negative),
        'Proportion positive': n_positive / total_ts if total_ts else 0,
        'Proportion negative': n_negative / total_ts if total_ts else 0,
        'Event rate (events/patient-hour)': n_distinct_events / total_ts if total_ts else 0,
        'Positive timestep rate (pos_ts/patient-hour)': n_positive / total_ts if total_ts else 0,
    }


def compute_event_rates(X, y_df, dataset_name):
    """Compute END event rate metrics with 95% bootstrap CIs (patient-level resampling)."""
    ps = _per_patient_stats(X, y_df)

    # Point estimates
    point = _metrics_from_patient_stats(ps)
    point['Dataset'] = dataset_name

    # Bootstrap
    rng = np.random.RandomState(RANDOM_SEED)
    ci_metrics = [k for k in point if k != 'Dataset']
    boot_results = {k: [] for k in ci_metrics}

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(ps), size=len(ps), replace=True)
        boot_ps = ps.iloc[idx].reset_index(drop=True)
        boot_m = _metrics_from_patient_stats(boot_ps)
        for k in ci_metrics:
            boot_results[k].append(boot_m[k])

    for k in ci_metrics:
        vals = np.array(boot_results[k])
        point[f'{k} CI lower'] = np.percentile(vals, 2.5)
        point[f'{k} CI upper'] = np.percentile(vals, 97.5)

    return point


def main():
    # --- Derivation set (fold 0: train + val combined) ---
    print('Loading derivation data...')
    train_splits = torch.load(TRAIN_PATH, map_location='cpu')
    X_train, X_val, y_train, y_val = train_splits[0]
    X_deriv = np.concatenate([X_train, X_val], axis=0)
    y_deriv = pd.concat([y_train, y_val], ignore_index=True)
    del train_splits, X_train, X_val, y_train, y_val

    # --- Test set ---
    print('Loading test data...')
    X_test, y_test = torch.load(TEST_PATH, map_location='cpu')

    # --- Overall (derivation + test) ---
    X_all = np.concatenate([X_deriv, X_test], axis=0)
    y_all = pd.concat([y_deriv, y_test], ignore_index=True)

    print('Computing overall event rates...')
    overall_metrics = compute_event_rates(X_all, y_all, 'Overall')
    del X_all, y_all

    print('Computing derivation event rates...')
    deriv_metrics = compute_event_rates(X_deriv, y_deriv, 'Derivation')
    del X_deriv, y_deriv

    print('Computing test event rates...')
    test_metrics = compute_event_rates(X_test, y_test, 'Test')
    del X_test, y_test

    # --- Summary ---
    results = pd.DataFrame([overall_metrics, deriv_metrics, test_metrics])

    print('\n' + '=' * 80)
    print(f'END Event Rates Summary (95% CI from {N_BOOTSTRAP} bootstrap iterations)')
    print('=' * 80)

    def _fmt_ci(row, key, fmt='.4f'):
        lo = row[f'{key} CI lower']
        hi = row[f'{key} CI upper']
        return f'({lo:{fmt}} – {hi:{fmt}})'

    for _, row in results.iterrows():
        print(f"\n--- {row['Dataset']} ---")
        print(f"  N patients:                    {row['N patients']}")
        print(f"  Total patient-hours:           {row['Total patient-hours']}")
        print(f"  Patients with >= 1 END event:  {row['Patients with >= 1 END event']}")
        print(f"  Distinct END events:           {row['Distinct END events']}")
        print(f"  Positive timesteps (label=1):  {row['Positive timesteps (label=1)']}")
        print(f"  Negative timesteps (label=0):  {row['Negative timesteps (label=0)']}")
        print(f"  Proportion positive:           {row['Proportion positive']:.4f}  {_fmt_ci(row, 'Proportion positive')}")
        print(f"  Proportion negative:           {row['Proportion negative']:.4f}  {_fmt_ci(row, 'Proportion negative')}")
        print(f"  Event rate (events/pt-hour):   {row['Event rate (events/patient-hour)']:.6f}  {_fmt_ci(row, 'Event rate (events/patient-hour)', '.6f')}")
        print(f"  Pos timestep rate (pos/pt-hr): {row['Positive timestep rate (pos_ts/patient-hour)']:.6f}  {_fmt_ci(row, 'Positive timestep rate (pos_ts/patient-hour)', '.6f')}")

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'end_event_rates.csv')
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
