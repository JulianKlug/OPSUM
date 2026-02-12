"""
Data loader for transforming OPSUM data to MIRA input format.

MIRA expects single-channel time series:
- input_ids: [batch, seq_len, 1] - historical values
- time_values: [batch, seq_len] - timestamps
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


def load_data_splits(data_path: str) -> List[Tuple]:
    """
    Load data splits from .pth file.

    Args:
        data_path: Path to the .pth file containing data splits

    Returns:
        List of tuples: (X_train, X_val, y_train_df, y_val_df)
    """
    splits = torch.load(data_path, map_location='cpu', weights_only=False)
    return splits


def get_feature_indices(X: np.ndarray) -> Tuple[int, int]:
    """
    Extract indices for min_NIHSS and max_NIHSS features.

    The feature names are stored in X[0, 0, :, -2].

    Args:
        X: Data array of shape (n_cases, n_time_steps, n_features, n_dims)

    Returns:
        Tuple of (min_NIHSS_idx, max_NIHSS_idx)
    """
    feature_names = X[0, 0, :, -2]

    max_nihss_idx = None
    min_nihss_idx = None

    for idx, name in enumerate(feature_names):
        if name == 'max_NIHSS':
            max_nihss_idx = idx
        elif name == 'min_NIHSS':
            min_nihss_idx = idx

    if max_nihss_idx is None:
        raise ValueError("max_NIHSS feature not found in data")
    if min_nihss_idx is None:
        raise ValueError("min_NIHSS feature not found in data")

    return min_nihss_idx, max_nihss_idx


def get_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on training data.

    Args:
        X_train: Training data of shape (n_cases, n_time_steps, n_features, n_dims)

    Returns:
        Fitted StandardScaler
    """
    # Extract feature values (last dimension contains the values)
    X_train_values = X_train[:, :, :, -1].astype('float32')

    # Reshape for scaler: (n_cases * n_time_steps, n_features)
    n_cases, n_time_steps, n_features = X_train_values.shape
    X_reshaped = X_train_values.reshape(-1, n_features)

    scaler = StandardScaler()
    scaler.fit(X_reshaped)

    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Apply StandardScaler to data.

    Args:
        X: Data of shape (n_cases, n_time_steps, n_features, n_dims)
        scaler: Fitted StandardScaler

    Returns:
        Scaled data of shape (n_cases, n_time_steps, n_features)
    """
    X_values = X[:, :, :, -1].astype('float32')
    n_cases, n_time_steps, n_features = X_values.shape

    X_reshaped = X_values.reshape(-1, n_features)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_cases, n_time_steps, n_features)

    return X_scaled


def extract_feature_timeseries(X_scaled: np.ndarray, feature_idx: int) -> np.ndarray:
    """
    Extract a single feature's time series from scaled data.

    Args:
        X_scaled: Scaled data of shape (n_cases, n_time_steps, n_features)
        feature_idx: Index of the feature to extract

    Returns:
        Feature time series of shape (n_cases, n_time_steps)
    """
    return X_scaled[:, :, feature_idx]


def prepare_mira_input(
    feature_timeseries: np.ndarray,
    up_to_timestep: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare MIRA input format from a single-feature time series.

    MIRA expects:
    - input_ids: [batch, seq_len, 1]
    - time_values: [batch, seq_len]

    Args:
        feature_timeseries: Feature values of shape (n_cases, n_time_steps)
        up_to_timestep: Include values from timestep 0 to up_to_timestep (inclusive)

    Returns:
        Tuple of (input_ids, time_values) tensors
    """
    # Extract values up to the given timestep
    values = feature_timeseries[:, :up_to_timestep + 1]
    n_cases, seq_len = values.shape

    # Create input_ids: [batch, seq_len, 1]
    input_ids = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    # Create time_values: [batch, seq_len]
    # Hourly timestamps: 0, 1, 2, ..., seq_len-1
    time_values = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(n_cases, -1)

    return input_ids, time_values


def load_normalisation_parameters(normalisation_path: str) -> pd.DataFrame:
    """
    Load normalisation parameters from CSV file.

    Args:
        normalisation_path: Path to normalisation_parameters.csv

    Returns:
        DataFrame with columns: variable, original_mean, original_std
    """
    return pd.read_csv(normalisation_path)


def load_outcome_data(outcome_path: str) -> pd.DataFrame:
    """
    Load outcome labels from CSV file.

    Args:
        outcome_path: Path to outcome CSV file

    Returns:
        DataFrame with columns: case_admission_id, relative_sample_date_hourly_cat, outcome_label
    """
    return pd.read_csv(outcome_path)


def get_validation_patient_ids(X_val: np.ndarray) -> np.ndarray:
    """
    Extract patient/case admission IDs from validation data.

    Args:
        X_val: Validation data of shape (n_cases, n_time_steps, n_features, n_dims)

    Returns:
        Array of case admission IDs
    """
    return X_val[:, 0, 0, 0]


def get_all_feature_names_and_indices(X: np.ndarray) -> dict:
    """
    Extract mapping of all feature names to their indices.

    Feature names are stored in X[0, 0, :, -2].

    Args:
        X: Data array of shape (n_cases, n_time_steps, n_features, n_dims)

    Returns:
        Dict mapping feature name (str) to index (int)
    """
    feature_names = X[0, 0, :, -2]
    return {str(name): idx for idx, name in enumerate(feature_names)}


def get_feature_index(X: np.ndarray, feature_name: str) -> int:
    """
    Get the index of a single feature by name.

    Args:
        X: Data array of shape (n_cases, n_time_steps, n_features, n_dims)
        feature_name: Name of the feature to find

    Returns:
        Index of the feature

    Raises:
        ValueError: If feature not found
    """
    feature_map = get_all_feature_names_and_indices(X)
    if feature_name not in feature_map:
        raise ValueError(f"Feature '{feature_name}' not found. Available: {list(feature_map.keys())}")
    return feature_map[feature_name]
