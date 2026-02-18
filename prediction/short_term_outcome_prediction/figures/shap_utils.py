"""Shared data loading, processing, and constants for SHAP figure scripts."""

import os
import pickle

import numpy as np
import pandas as pd

# ─── Prefix constants ────────────────────────────────────────────────────────
BASE_PREFIXES = ['avg_', 'min_', 'max_', 'std_', 'diff_']
LAG_PREFIXES = ['lag2_', 'lag3_']
ROLLING_PREFIXES = ['rolling_mean_', 'rolling_std_', 'rolling_trend_']
ALL_PREFIXES = BASE_PREFIXES + LAG_PREFIXES + ROLLING_PREFIXES

# Display names for aggregation types
AGG_DISPLAY_NAMES = {
    'raw': 'Raw value',
    'avg': 'Cumulative mean',
    'min': 'Cumulative min',
    'max': 'Cumulative max',
    'std': 'Cumulative std',
    'diff': 'Rate of change',
    'timestep_idx': 'Timestep index',
    'lag2': 'Lag t-2',
    'lag3': 'Lag t-3',
    'rolling_mean': 'Rolling mean (6h)',
    'rolling_std': 'Rolling std (6h)',
    'rolling_trend': 'Rolling trend (6h)',
}


# ─── Data loading ────────────────────────────────────────────────────────────

def load_shap_and_test_data(shap_path, test_data_path):
    """Load SHAP values and test data.

    Returns:
        shap_values: array (n_subjects, n_timesteps, n_features+1)
        X_test_raw: raw test tensor
        y_test_raw: raw test labels
    """
    with open(shap_path, 'rb') as f:
        original_shap_values = pickle.load(f)

    # Reshape from list of (n_subj, n_feat+1) per timestep to (n_subj, n_ts, n_feat+1)
    shap_values = np.array([original_shap_values[i]
                            for i in range(len(original_shap_values))]).swapaxes(0, 1)

    import torch as ch
    X_test_raw, y_test_raw = ch.load(test_data_path)

    return shap_values, X_test_raw, y_test_raw


def build_feature_names(X_test_raw, add_lag_features=False, add_rolling_features=False):
    """Build aggregated feature names from raw feature names in the test data."""
    raw_features = list(X_test_raw[0, 0, :, 2])

    names = list(raw_features)
    for prefix in BASE_PREFIXES:
        names += [f'{prefix}{f}' for f in raw_features]
    names += ['timestep_idx']

    if add_lag_features:
        for prefix in LAG_PREFIXES:
            names += [f'{prefix}{f}' for f in raw_features]
    if add_rolling_features:
        for prefix in ROLLING_PREFIXES:
            names += [f'{prefix}{f}' for f in raw_features]

    return names, raw_features


# ─── Time aggregation helpers ────────────────────────────────────────────────

def get_prefix(name):
    """Return the aggregation prefix key for a feature name."""
    for p in ROLLING_PREFIXES + BASE_PREFIXES + LAG_PREFIXES:
        if name.startswith(p):
            return p.rstrip('_')
    if name == 'timestep_idx':
        return 'timestep_idx'
    return 'raw'


def peak_shap_over_time(shap_values):
    """Select the signed SHAP value at the timestep with maximum |SHAP|.

    Unlike np.max (which biases toward positive values), this preserves the sign
    while selecting the most impactful timestep per subject per feature.

    Args:
        shap_values: (n_subjects, n_timesteps, n_features+1)
    Returns:
        peak_shap: (n_subjects, n_features+1) signed SHAP values
        idx_peak: (n_subjects, n_features+1) timestep indices of peak |SHAP|
    """
    idx_peak = np.argmax(np.abs(shap_values), axis=1)
    peak_shap = np.take_along_axis(
        shap_values, idx_peak[:, np.newaxis, :], axis=1).squeeze(axis=1)
    return peak_shap, idx_peak


def compute_mean_shap_with_feature_values(shap_values, X_test_raw, feature_names):
    """Compute mean-over-time SHAP values and mean feature values.

    Averages SHAP across all timesteps (rather than picking only the peak),
    capturing each feature's sustained contribution over the full observation window.

    Returns:
        DataFrame with columns: case_admission_id_idx, feature, shap_value, feature_value
    """
    # Mean SHAP across all timesteps
    mean_shap = np.mean(shap_values, axis=1)  # (n_subj, n_feat+1)

    all_names = feature_names + ['base_value']
    shap_df = pd.DataFrame(data=mean_shap, columns=all_names)
    shap_df.reset_index(inplace=True)
    shap_df.rename(columns={'index': 'case_admission_id_idx'}, inplace=True)
    shap_df = shap_df.melt(id_vars='case_admission_id_idx',
                            var_name='feature', value_name='shap_value')

    # Mean feature value across all timesteps (only for raw features)
    test_X_np = X_test_raw[:, :, :, -1].astype('float32')
    mean_feat_vals = np.mean(test_X_np, axis=1)  # (n_subj, n_raw_features)

    raw_features = list(X_test_raw[0, 0, :, 2])
    feat_val_df = pd.DataFrame(data=mean_feat_vals, columns=raw_features)
    feat_val_df.reset_index(inplace=True)
    feat_val_df.rename(columns={'index': 'case_admission_id_idx'}, inplace=True)
    feat_val_df = feat_val_df.melt(id_vars='case_admission_id_idx',
                                    var_name='feature', value_name='feature_value')

    merged = pd.merge(shap_df, feat_val_df,
                       on=['case_admission_id_idx', 'feature'], how='left')

    return merged


def identify_time_invariant_features(X_test_raw):
    """Return set of raw feature names that are constant across all timesteps for every subject."""
    test_X_np = X_test_raw[:, :, :, -1].astype('float32')
    raw_features = list(X_test_raw[0, 0, :, 2])
    std_over_time = np.std(test_X_np, axis=1)  # (n_subj, n_feat)
    is_invariant = np.all(std_over_time == 0, axis=0)
    return {f for f, inv in zip(raw_features, is_invariant) if inv}


# ─── Pooling / encoding helpers ──────────────────────────────────────────────

def pool_time_aggregated_prefixes(df):
    """Remove aggregation prefixes and sum SHAP/feature values per subject per base feature."""
    for prefix in ALL_PREFIXES:
        df['feature'] = df['feature'].str.replace(f'^{prefix}', '', regex=True)
    df = df.groupby(['case_admission_id_idx', 'feature']).sum().reset_index()
    return df


def reverse_categorical_encoding(df, cat_encoding_path):
    """Reverse categorical encoding: pool one-hot categories back to their parent variable."""
    cat_encoding_df = pd.read_csv(cat_encoding_path)
    for i in range(len(cat_encoding_df)):
        cat_basename = cat_encoding_df.sample_label[i].lower().replace(' ', '_')
        cat_item_list = (cat_encoding_df.other_categories[i]
                         .replace('[', '').replace(']', '').replace("'", '').split(', '))
        cat_item_list = [cat_basename + '_' + item.replace(' ', '_').lower()
                         for item in cat_item_list]
        for cat_item_idx, cat_item in enumerate(cat_item_list):
            df.loc[df.feature == cat_item, 'feature_value'] *= cat_item_idx + 1
            df.loc[df.feature == cat_item, 'feature'] = cat_encoding_df.sample_label[i]
            df = df.groupby(['case_admission_id_idx', 'feature']).sum().reset_index()

    cat_to_numerical_encoding = {
        'Prestroke disability (Rankin)': {0: 0, 1: 3, 2: 4, 3: 2, 4: 1, 5: 5},
        'categorical_onset_to_admission_time': {0: 2, 1: 1, 2: 0, 3: 3, 4: 5, 5: 4},
        'categorical_IVT': {0: 2, 1: 3, 2: 4, 3: 1, 4: 0},
        'categorical_IAT': {0: 1, 1: 2, 2: 3, 3: 0},
    }
    for cat_feature, cat_encoding in cat_to_numerical_encoding.items():
        df.loc[df.feature == cat_feature, 'feature_value'] = (
            df.loc[df.feature == cat_feature, 'feature_value'].map(cat_encoding))

    return df


def pool_hourly_split_features(df):
    """Pool hourly-split features (NIHSS, blood_pressure, etc.) into single features."""
    hourly_split_features = [
        'NIHSS', 'systolic_blood_pressure', 'diastolic_blood_pressure',
        'mean_blood_pressure', 'heart_rate', 'respiratory_rate',
        'temperature', 'oxygen_saturation',
    ]
    for feature in hourly_split_features:
        mask = df.feature.str.contains(feature)
        display_name = feature[0].upper() + feature[1:]
        display_name = display_name.replace('_', ' ')
        df.loc[mask, 'feature'] = display_name

    return df


def map_to_english_names(df, feature_names_path):
    """Map coded feature names to English names using correspondence file."""
    correspondence = pd.read_excel(feature_names_path)
    for feature in df.feature.unique():
        if feature in correspondence.feature_name.values:
            english = correspondence[correspondence.feature_name == feature].english_name.values[0]
            df.loc[df.feature == feature, 'feature'] = english
    return df


def select_top_features(df, n_top_features=10):
    """Select top N features by mean absolute SHAP value, excluding base_value."""
    df['absolute_shap_value'] = np.abs(df['shap_value'])
    top_features = (df.groupby('feature')['absolute_shap_value']
                    .mean()
                    .sort_values(ascending=False)
                    .head(n_top_features + 1)
                    .index.values)
    top_features = top_features[top_features != 'base_value'][:n_top_features]
    selected_df = df[df.feature.isin(top_features)].copy()
    return selected_df, top_features


# ─── Prefix stripping for feature pooling ────────────────────────────────────

HOURLY_PREFIXES = ['median_', 'min_', 'max_']
BP_PREFIXES = ['systolic_', 'diastolic_', 'mean_']


def strip_agg_prefix(name):
    """Strip the leading aggregation prefix to recover the raw feature name."""
    for p in ALL_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def strip_to_base_name(name):
    """Strip all prefixes (aggregation + hourly + BP) sequentially.

    Matches the logic in top_predictors_selection.py for pooling features
    across aggregation types, hourly sub-measurements, and BP types.
    """
    for p in ALL_PREFIXES + HOURLY_PREFIXES + BP_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
    return name


def select_top_predictors(shap_values, feature_names, n_top=10):
    """Select top features by total |SHAP| across all subjects and timesteps.

    Pools features by stripping all prefixes (aggregation, hourly, BP),
    then ranks by summed absolute SHAP.
    """
    total_abs_shap = np.abs(shap_values).sum(axis=(0, 1))
    all_names = feature_names + ['base_value']

    df = pd.DataFrame({'feature': all_names, 'abs_shap': total_abs_shap})
    df = df[~df.feature.isin(['timestep_idx', 'base_value'])]
    df['feature'] = df['feature'].apply(strip_to_base_name)
    df = df.groupby('feature')['abs_shap'].sum().reset_index()

    return (df.sort_values('abs_shap', ascending=False)
              .head(n_top)
              .feature.values)


def prepare_individual_obs_df(shap_values, X_test_raw, feature_names,
                              top_features, peak_per_subject=False):
    """Build per-observation SHAP DataFrame for selected features.

    Each row is one subject x timestep x aggregation variant.
    Features are pooled by base name (all aggregation/hourly/BP variants
    map to the same name). Feature values are the raw values from
    X_test_raw at each timestep.

    Args:
        peak_per_subject: if True, keep only the observation with the
            highest |SHAP| per subject per base feature (one dot per
            subject per feature).

    Returns DataFrame with columns: feature, shap_value, feature_value.
    """
    n_subj, n_ts, _ = shap_values.shape
    raw_features = list(X_test_raw[0, 0, :, 2])
    raw_feature_idx = {f: i for i, f in enumerate(raw_features)}
    test_X_np = X_test_raw[:, :, :, -1].astype('float32')

    all_names = feature_names + ['base_value']
    top_set = set(top_features)

    selected_indices = []
    display_names_list = []
    raw_idx_list = []

    for i, name in enumerate(all_names):
        if name in ('timestep_idx', 'base_value'):
            continue
        raw_name = strip_agg_prefix(name)
        display_name = strip_to_base_name(name)
        if display_name not in top_set:
            continue
        selected_indices.append(i)
        display_names_list.append(display_name)
        raw_idx_list.append(raw_feature_idx.get(raw_name))

    n_selected = len(selected_indices)

    # SHAP values for selected features: (n_subj, n_ts, n_selected) → flat
    shap_flat = shap_values[:, :, selected_indices].reshape(-1)

    # Tiled display names (same order repeats for each subject × timestep)
    feature_flat = np.tile(np.array(display_names_list), n_subj * n_ts)

    # Raw feature values at each (subject, timestep)
    feat_vals = np.full(n_subj * n_ts * n_selected, np.nan, dtype=np.float32)
    for j, raw_idx in enumerate(raw_idx_list):
        if raw_idx is not None:
            feat_vals[j::n_selected] = test_X_np[:, :, raw_idx].reshape(-1)

    # Subject index: each subject has n_ts * n_selected entries
    subject_flat = np.repeat(np.arange(n_subj), n_ts * n_selected)

    df = pd.DataFrame({
        'subject': subject_flat,
        'feature': feature_flat,
        'shap_value': shap_flat,
        'feature_value': feat_vals,
    })

    if peak_per_subject:
        df['abs_shap'] = np.abs(df['shap_value'])
        idx = df.groupby(['subject', 'feature'])['abs_shap'].idxmax()
        df = df.loc[idx].drop(columns=['abs_shap'])

    return df.drop(columns=['subject']).reset_index(drop=True)


def format_feature_name(name, feature_names_path=None, _correspondence=None):
    """Format a base feature name for display.

    Applies English name mapping if available, otherwise capitalises
    and replaces underscores with spaces.
    """
    if _correspondence is not None:
        if name in _correspondence.feature_name.values:
            return _correspondence[
                _correspondence.feature_name == name].english_name.values[0]
    elif feature_names_path is not None:
        correspondence = pd.read_excel(feature_names_path)
        if name in correspondence.feature_name.values:
            return correspondence[
                correspondence.feature_name == name].english_name.values[0]
    formatted = name.replace('_', ' ')
    if formatted and formatted[0].islower():
        formatted = formatted[0].upper() + formatted[1:]
    return formatted


# ─── High-level pipelines ───────────────────────────────────────────────────

def prepare_pooled_df(shap_path, test_data_path, cat_encoding_path,
                      feature_names_path, add_lag_features, add_rolling_features):
    """Full pipeline: load data → mean-over-time → pool prefixes → categories → English names.

    Returns the processed DataFrame (one row per subject x base feature) and metadata.
    """
    shap_values, X_test_raw, _ = load_shap_and_test_data(shap_path, test_data_path)
    feature_names, raw_features = build_feature_names(
        X_test_raw, add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features)

    df = compute_mean_shap_with_feature_values(shap_values, X_test_raw, feature_names)
    df = pool_time_aggregated_prefixes(df)
    df = reverse_categorical_encoding(df, cat_encoding_path)
    df = pool_hourly_split_features(df)
    df = map_to_english_names(df, feature_names_path)

    ti_features = identify_time_invariant_features(X_test_raw)

    return df, shap_values, X_test_raw, feature_names, ti_features


# ─── TI category helpers ────────────────────────────────────────────────────

def parse_cat_field(raw_str):
    """Parse a categorical encoding field like \"['a', 'b']\" into a list of strings."""
    return (raw_str.replace('[', '').replace(']', '')
            .replace("'", '').split(', '))


def get_ti_shap_by_base(shap_values, X_test_raw, feature_names):
    """Pool aggregation prefixes and filter to time-invariant base features.

    Returns:
        shap_ti: DataFrame (n_subjects x TI base features)
        ti_features: set of TI raw feature names
    """
    peak_shap, _ = peak_shap_over_time(shap_values)
    all_names = feature_names + ['base_value']
    shap_df = pd.DataFrame(peak_shap, columns=all_names).drop(columns=['base_value'])

    base_map = {}
    for col in shap_df.columns:
        base = col
        for p in ALL_PREFIXES:
            if col.startswith(p):
                base = col[len(p):]
                break
        base_map[col] = base
    renamed = shap_df.rename(columns=base_map)
    shap_by_base = renamed.T.groupby(renamed.columns).sum().T

    ti_features = identify_time_invariant_features(X_test_raw)
    ti_cols = [c for c in shap_by_base.columns if c in ti_features]

    return shap_by_base[ti_cols], ti_features


def prepare_ti_category_shap(shap_values, X_test_raw, feature_names,
                              cat_encoding_path, feature_names_path):
    """Compute mean SHAP per category level for time-invariant categorical features.

    Returns:
        cat_df: DataFrame with columns [parent, level, mean_shap]
    """
    shap_ti, ti_features = get_ti_shap_by_base(shap_values, X_test_raw, feature_names)

    cat_encoding_df = pd.read_csv(cat_encoding_path)
    correspondence = pd.read_excel(feature_names_path)

    def to_english(name):
        if name in correspondence.feature_name.values:
            return correspondence[correspondence.feature_name == name].english_name.values[0]
        return name

    records = []
    for i in range(len(cat_encoding_df)):
        parent = cat_encoding_df.sample_label.iloc[i]
        cat_basename = parent.lower().replace(' ', '_')
        all_levels = (parse_cat_field(cat_encoding_df.baseline_value.iloc[i])
                      + parse_cat_field(cat_encoding_df.other_categories.iloc[i]))

        parent_english = to_english(parent)
        for level in all_levels:
            one_hot = cat_basename + '_' + level.replace(' ', '_').lower()
            if one_hot in ti_features and one_hot in shap_ti.columns:
                records.append({
                    'parent': parent_english,
                    'level': level,
                    'mean_shap': shap_ti[one_hot].mean(),
                })

    return pd.DataFrame(records)
