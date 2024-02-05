import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

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


def aggregate_features_over_time(features, labels, moving_average=False, n=3):
    """
    This function aggregates the features over time. Instead of having one row per case_admission_id and one column per feature over time,
    we have one row per case_admission_id and one column per feature aggregated over time (mean, min, max) for each timestep.
    The timesteps are then flattened along with the samples.

    :param features: a numpy array of shape (n_samples, n_time_steps, n_features)
    :param labels: a numpy array of shape (n_samples, 1)
    :param moving_average: if True, the moving average over the last n time steps is calculated
    :param n: the number of time steps for the moving average
    """
    avg_features = np.cumsum(features, 1) / (np.arange(1, features.shape[1] + 1)[None, :, None])
    if moving_average:
        avg_features = np.append(avg_features[:, :n - 1], moving_time_average(features, n), axis=1)

    min_features = np.minimum.accumulate(features, 1)
    max_features = np.maximum.accumulate(features, 1)
    all_features = np.concatenate([features, avg_features, min_features, max_features], 2)
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
        x = tf.convert_to_tensor(x)
    return x