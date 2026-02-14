import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K


# define function for balanced training
def generate_balanced_arrays(X_train, y_train):
    while True:
        initial_positive = np.where(y_train == 1)[0].tolist()
        initial_negative = np.where(y_train == 0)[0].tolist()

        if len(initial_positive) < len(initial_negative):
            # if there are more negative samples than positive samples
            positive = initial_positive
            negative = np.random.choice(initial_negative, len(initial_positive), replace=False).tolist()
        else:
            # If there are more positive samples than negative samples, we need to downsample the positive samples
            positive = np.random.choice(initial_positive, len(initial_negative), replace=False).tolist()
            negative = initial_negative

        balance = np.concatenate((positive, negative), axis=0)
        np.random.shuffle(balance)
        input_ = X_train[balance]
        target = y_train[balance]
        yield input_, target


def filter_consecutive_numbers(lst):
    a = np.array(list(lst)).astype(int)
    if len(lst) < 2:
        return a
    consecutive_mask = np.concatenate(([False], (np.abs(a[1:] - a[:-1]) == 1)))
    result = a[np.logical_not(consecutive_mask)]
    if len(lst) == 2:
        return result
    next_to_consecutive_mask = np.concatenate(([False], (np.abs(result[1:] - result[:-1]) == 2)))
    result = result[np.logical_not(next_to_consecutive_mask)]
    return np.array(result)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def moving_time_average(a, n=3):
    """
    This function calculates the moving average over the last n elements of the array a.
    """
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:] / n


def _rolling_mean(features, w):
    """Rolling mean over last w timesteps. For t < w, uses all available timesteps."""
    n_samples, n_ts, n_feat = features.shape
    cumsum = np.cumsum(features, axis=1)
    rolling = np.empty_like(features)
    # For t < w: cumulative mean (use all available)
    for t in range(min(w, n_ts)):
        rolling[:, t, :] = cumsum[:, t, :] / (t + 1)
    # For t >= w: mean of last w values
    if w < n_ts:
        rolling[:, w:, :] = (cumsum[:, w:, :] - cumsum[:, :-w, :]) / w
    return rolling


def _rolling_std(features, rolling_mean, w):
    """Rolling std over last w timesteps. For t < w, uses all available timesteps."""
    n_samples, n_ts, n_feat = features.shape
    cumsum_sq = np.cumsum(features ** 2, axis=1)
    rolling_var = np.empty_like(features)
    for t in range(min(w, n_ts)):
        mean_sq = cumsum_sq[:, t, :] / (t + 1)
        rolling_var[:, t, :] = mean_sq - rolling_mean[:, t, :] ** 2
    if w < n_ts:
        mean_sq = (cumsum_sq[:, w:, :] - cumsum_sq[:, :-w, :]) / w
        rolling_var[:, w:, :] = mean_sq - rolling_mean[:, w:, :] ** 2
    return np.sqrt(np.maximum(rolling_var, 0))


def _rolling_trend(features, w):
    """Linear trend (slope) over last w timesteps via least-squares.
    For t < w, uses all available timesteps. Slope is normalized by window size."""
    n_samples, n_ts, n_feat = features.shape
    trend = np.zeros_like(features)
    # Precompute: slope = (sum(t*x) - n*mean_t*mean_x) / (sum(t^2) - n*mean_t^2)
    for t in range(1, n_ts):
        ww = min(t + 1, w)
        # indices 0..ww-1 within the window
        idx = np.arange(ww, dtype=features.dtype)
        sum_t = idx.sum()
        sum_t2 = (idx ** 2).sum()
        mean_t = sum_t / ww
        denom = sum_t2 - ww * mean_t ** 2
        if denom < 1e-10:
            continue
        # window of values: features[:, t-ww+1:t+1, :]
        window = features[:, t - ww + 1:t + 1, :]  # (n_samples, ww, n_feat)
        # sum(t * x) for each sample/feature
        sum_tx = np.einsum('j,ijk->ik', idx, window)
        mean_x = window.mean(axis=1)
        slope = (sum_tx - ww * mean_t * mean_x) / denom
        trend[:, t, :] = slope
    return trend


def aggregate_features_over_time(features, labels, moving_average=False, n=3,
                                 add_lag_features=False, add_rolling_features=False,
                                 rolling_window=6):
    """
    This function aggregates the features over time. Instead of having one row per case_admission_id and one column per feature over time,
    we have one row per case_admission_id and one column per feature aggregated over time (mean, min, max) for each timestep.
    The timesteps are then flattened along with the samples.

    :param features: a numpy array of shape (n_samples, n_time_steps, n_features)
    :param labels: a numpy array of shape (n_samples, 1)
    :param moving_average: if True, the moving average over the last n time steps is calculated
    :param n: the number of time steps for the moving average
    :param add_lag_features: if True, add lagged values at t-2 and t-3
    :param add_rolling_features: if True, add rolling window mean, std, and trend
    :param rolling_window: window size in timesteps for rolling features (default: 6 hours)
    """
    avg_features = np.cumsum(features, 1) / (np.arange(1, features.shape[1] + 1)[None, :, None])
    if moving_average:
        avg_features = np.append(avg_features[:, :n - 1], moving_time_average(features, n), axis=1)

    min_features = np.minimum.accumulate(features, 1)
    max_features = np.maximum.accumulate(features, 1)
    cumsum_sq = np.cumsum(features**2, 1)
    counts = np.arange(1, features.shape[1] + 1)[None, :, None]
    std_features = np.sqrt(np.maximum(cumsum_sq / counts - avg_features**2, 0))

    # Rate of change (first-order differences)
    diff_features = np.zeros_like(features)
    diff_features[:, 1:, :] = features[:, 1:, :] - features[:, :-1, :]

    # Timestep index (normalized to [0, 1])
    n_ts = features.shape[1]
    timestep_feature = np.arange(n_ts, dtype=features.dtype)[None, :, None]
    timestep_feature = np.broadcast_to(timestep_feature, (features.shape[0], n_ts, 1)).copy()
    timestep_feature = timestep_feature / max(n_ts - 1, 1)

    feature_list = [features, avg_features, min_features, max_features, std_features, diff_features, timestep_feature]

    if add_lag_features:
        # Lag-2: value at t-2 (zero-padded for t < 2)
        lag2 = np.zeros_like(features)
        lag2[:, 2:, :] = features[:, :-2, :]
        # Lag-3: value at t-3 (zero-padded for t < 3)
        lag3 = np.zeros_like(features)
        lag3[:, 3:, :] = features[:, :-3, :]
        feature_list.extend([lag2, lag3])

    if add_rolling_features:
        w = rolling_window
        roll_mean = _rolling_mean(features, w)
        roll_std = _rolling_std(features, roll_mean, w)
        roll_trend = _rolling_trend(features, w)
        feature_list.extend([roll_mean, roll_std, roll_trend])

    all_features = np.concatenate(feature_list, 2)
    all_features = all_features.reshape(-1, all_features.shape[-1])

    labels = labels[:, None].repeat(72, 1).ravel()

    return all_features, labels

def flatten(l):
    return [item for sublist in l for item in sublist]

def check_data(data):
    """
    Check if data contains nan or inf
    """
    if type(data) == np.ndarray:
        if np.isnan(data).any() or np.isinf(data).any():
            sys.exit('Data is corrupted!')
            return False
    elif type(data) == pd.DataFrame:
        if data.isnull().values.any() or data.isin([np.inf, -np.inf]).values.any():
            sys.exit('Data is corrupted!')
            return False
    else:
        return True


def ensure_dir(dirname: Path) -> None:
    """
    Create directory only if it does not exist yet.
    Throw an error otherwise.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def flatten(t):
    return [item for sublist in t for item in sublist]


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def ensure_tensor(x):
    if not isinstance(x, tf.Tensor):
        x = K.constant(x)
    return x

