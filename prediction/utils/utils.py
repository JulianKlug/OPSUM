import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


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

        balance = np.concatenate((positive, negative), axis = 0)
        np.random.shuffle(balance)
        input_ = X_train[balance]
        target = y_train[balance]
        yield input_, target


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

