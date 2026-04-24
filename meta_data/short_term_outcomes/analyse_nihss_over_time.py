"""
Analyse NIHSS variation over time (72h), stratified by END vs non-END patients.

Plots delta NIHSS from baseline with 3 rows: Overall, Derivation, Test.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

DATA_DIR = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047'
TRAIN_PATH = os.path.join(DATA_DIR, 'train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth')
TEST_PATH = os.path.join(DATA_DIR, 'test_data_early_neurological_deterioration_ts0.8_rs42_ns5.pth')
NORM_PATH = os.path.join(DATA_DIR, 'logs_30012026_154047/normalisation_parameters.csv')

OUTPUT_DIR = '/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/best_xgb_final_model/auroc_over_time_plots'

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
NIHSS_FEATURE = 'median_NIHSS'


def load_norm_params(feature_name):
    """Load mean and std for a feature from normalisation parameters CSV."""
    df = pd.read_csv(NORM_PATH)
    row = df[df['variable'] == feature_name]
    if len(row) == 0:
        raise ValueError(f'Feature {feature_name} not found in normalisation parameters')
    return float(row['original_mean'].values[0]), float(row['original_std'].values[0])


def extract_nihss(X, norm_mean, norm_std):
    """Extract raw NIHSS values per patient per timestep.

    Returns: nihss array (n_patients, n_timesteps), case_admission_ids array
    """
    feature_names = X[0, 0, :, 2]
    nihss_idx = np.where(feature_names == NIHSS_FEATURE)[0]
    if len(nihss_idx) == 0:
        raise ValueError(f'{NIHSS_FEATURE} not found in features')
    nihss_idx = nihss_idx[0]

    nihss_norm = X[:, :, nihss_idx, -1].astype(float)
    nihss_raw = nihss_norm * norm_std + norm_mean

    cids = X[:, 0, 0, 0]  # case_admission_id per patient
    return nihss_raw, cids


def split_by_end(nihss, cids, y_df):
    """Split NIHSS array into END and non-END groups.

    Returns: nihss_end, nihss_nonend
    """
    end_cids = set(y_df['case_admission_id'].values)
    has_end = np.array([cid in end_cids for cid in cids])
    return nihss[has_end], nihss[~has_end]


def bootstrap_mean_ci(data, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Compute mean and 95% CI per timestep via patient-level bootstrap.

    Args:
        data: (n_patients, n_timesteps) array

    Returns: mean, ci_lower, ci_upper — each (n_timesteps,)
    """
    rng = np.random.RandomState(seed)
    n_patients, n_ts = data.shape
    point_mean = np.nanmean(data, axis=0)

    boot_means = np.zeros((n_boot, n_ts))
    for b in range(n_boot):
        idx = rng.choice(n_patients, size=n_patients, replace=True)
        boot_means[b] = np.nanmean(data[idx], axis=0)

    ci_lower = np.percentile(boot_means, 2.5, axis=0)
    ci_upper = np.percentile(boot_means, 97.5, axis=0)
    return point_mean, ci_lower, ci_upper


def _style_ax(ax, title, show_xlabel=True):
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('\u0394 NIHSS from baseline', fontsize=10)
    if show_xlabel:
        ax.set_xlabel('Hours since admission', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 71)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))


def _add_legend(ax):
    ax.legend(loc='upper right', frameon=False, fontsize=9)


def plot_nihss_delta(datasets, output_dir):
    """Delta from baseline (NIHSS change from hour 0)."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    hours = np.arange(72)

    for ax, (name, nihss_end, nihss_nonend, _y_df) in zip(axes, datasets):
        show_xlabel = (ax == axes[-1])

        delta_end = nihss_end - nihss_end[:, 0:1]
        delta_nonend = nihss_nonend - nihss_nonend[:, 0:1]

        mean_de, lo_de, hi_de = bootstrap_mean_ci(delta_end)
        mean_dne, lo_dne, hi_dne = bootstrap_mean_ci(delta_nonend)

        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')

        ax.fill_between(hours, lo_dne, hi_dne, alpha=0.2, color='tab:blue')
        ax.plot(hours, mean_dne, color='tab:blue', linewidth=1.5, label=f'Non-END (n={len(nihss_nonend)})')

        ax.fill_between(hours, lo_de, hi_de, alpha=0.2, color='tab:red')
        ax.plot(hours, mean_de, color='tab:red', linewidth=1.5, label=f'END (n={len(nihss_end)})')

        _style_ax(ax, name, show_xlabel)
        _add_legend(ax)

    fig.tight_layout()

    for ext in ('png', 'tiff'):
        fig.savefig(os.path.join(output_dir, f'nihss_over_time.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_summary(name, nihss_end, nihss_nonend):
    """Print summary statistics for a dataset."""
    print(f'\n--- {name} ---')
    print(f'  END patients:     {len(nihss_end)}')
    print(f'  Non-END patients: {len(nihss_nonend)}')
    print(f'  END   — baseline NIHSS: mean={np.nanmean(nihss_end[:, 0]):.2f}, '
          f'median={np.nanmedian(nihss_end[:, 0]):.2f}')
    print(f'  NoEND — baseline NIHSS: mean={np.nanmean(nihss_nonend[:, 0]):.2f}, '
          f'median={np.nanmedian(nihss_nonend[:, 0]):.2f}')
    print(f'  END   — overall NIHSS:  mean={np.nanmean(nihss_end):.2f}, '
          f'median={np.nanmedian(nihss_end):.2f}')
    print(f'  NoEND — overall NIHSS:  mean={np.nanmean(nihss_nonend):.2f}, '
          f'median={np.nanmedian(nihss_nonend):.2f}')


def main():
    norm_mean, norm_std = load_norm_params(NIHSS_FEATURE)
    print(f'Normalisation params for {NIHSS_FEATURE}: mean={norm_mean:.4f}, std={norm_std:.4f}')

    # --- Derivation set (fold 0: train + val) ---
    print('\nLoading derivation data...')
    train_splits = torch.load(TRAIN_PATH, map_location='cpu')
    X_train, X_val, y_train, y_val = train_splits[0]
    X_deriv = np.concatenate([X_train, X_val], axis=0)
    y_deriv = pd.concat([y_train, y_val], ignore_index=True)
    del train_splits, X_train, X_val, y_train, y_val

    nihss_deriv, cids_deriv = extract_nihss(X_deriv, norm_mean, norm_std)
    nihss_end_deriv, nihss_nonend_deriv = split_by_end(nihss_deriv, cids_deriv, y_deriv)
    del X_deriv

    # --- Test set ---
    print('Loading test data...')
    X_test, y_test = torch.load(TEST_PATH, map_location='cpu')

    nihss_test, cids_test = extract_nihss(X_test, norm_mean, norm_std)
    nihss_end_test, nihss_nonend_test = split_by_end(nihss_test, cids_test, y_test)
    del X_test

    # --- Overall (derivation + test) ---
    nihss_end_all = np.concatenate([nihss_end_deriv, nihss_end_test], axis=0)
    nihss_nonend_all = np.concatenate([nihss_nonend_deriv, nihss_nonend_test], axis=0)
    y_all = pd.concat([y_deriv, y_test], ignore_index=True)

    # --- Summary stats ---
    print('\n' + '=' * 60)
    print('NIHSS Summary Statistics (END vs non-END)')
    print('=' * 60)
    print_summary('Overall', nihss_end_all, nihss_nonend_all)
    print_summary('Derivation', nihss_end_deriv, nihss_nonend_deriv)
    print_summary('Test', nihss_end_test, nihss_nonend_test)

    # --- Plotting ---
    datasets = [
        ('Overall', nihss_end_all, nihss_nonend_all, y_all),
        ('Derivation', nihss_end_deriv, nihss_nonend_deriv, y_deriv),
        ('Test', nihss_end_test, nihss_nonend_test, y_test),
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('\nGenerating plot...')
    plot_nihss_delta(datasets, OUTPUT_DIR)
    print('  Plot saved.')

    print(f'\nPlot saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
