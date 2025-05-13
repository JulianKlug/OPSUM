import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold

from prediction.outcome_prediction.data_loading.data_formatting import features_to_numpy
from prediction.utils.utils import check_data, ensure_dir


def data_splits_to_file(features_path: str, labels_path:str, outcome:str, output_dir:str,
                      test_pids_path:str=None, train_pids_path:str=None,
                      test_size:float=0.2, seed=42, n_splits=5):
    """
    Splits the input data into training and testing sets based on patient IDs and a specified outcome. Saves the data
    splits to a file.

    Args:
        features_path (str): Path to the input features DataFrame.
        labels_path (str): Path to the outcome labels DataFrame.
        output_dir (str): Path to the output directory.
        outcome (str): The specific outcome label to consider for splitting.
        test_size (float): The proportion of the dataset to include in the test split.
        n_splits (int): Number of splits for cross-validation.
        seed (int): Random seed for reproducibility.

    If using predefined test and train patient ids:
        test_pids_path (str): Path to the test patient IDs file.
        test_pids_path (str): Path to the test patient IDs file.

    Returns:
        list: A list of tuples containing the split training and validation data.
        tuple: A tuple containing the test data.
    """

    ensure_dir(output_dir)
    # save all arguments given to function as json
    args = locals()
    # remove 'args' from args to avoid circular reference
    args = {k: v for k, v in args.items() if k != 'args'}

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    # load data
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    X['patient_id'] = X['case_admission_id'].apply(lambda x: x.split('_')[0])
    y['patient_id'] = y['case_admission_id'].apply(lambda x: x.split('_')[0])

    # test if data is corrupted
    check_data(X)

    # get splits per pid
    pid_train, pid_test, split_idx = generate_splits(X, y, outcome, test_size, n_splits, seed,
                                                        test_pids_path, train_pids_path)

    # save patient ids used for testing / training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.csv'), index=False)
    pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_test.csv'), index=False)

    columns_to_keep = ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value']

    # split off test data
    test_X = X[X.patient_id.isin(pid_test)]
    test_X = features_to_numpy(test_X, columns_to_keep)
    test_y = y[y.patient_id.isin(pid_test)]

    # split off train and validation data
    splits = []
    for fold_train_pidx, fold_val_pidx in split_idx:
        fold_X_train_df = X.loc[X.patient_id.isin(fold_train_pidx)]
        fold_y_train_df = y.loc[y.patient_id.isin(fold_train_pidx)]
        fold_X_val_df = X.loc[X.patient_id.isin(fold_val_pidx)]
        fold_y_val_df = y.loc[y.patient_id.isin(fold_val_pidx)]

        fold_X_train = features_to_numpy(fold_X_train_df, columns_to_keep)
        fold_X_val = features_to_numpy(fold_X_val_df, columns_to_keep)

        # timeseries are not yet split into subseries (this will be done at a later stage)
        splits.append((fold_X_train, fold_X_val, fold_y_train_df, fold_y_val_df))

    torch.save(splits, os.path.join(output_dir,
                                    f'train_data_splits_{outcome.replace(" ", "_")}_ts{1 - test_size}_rs{seed}_ns{n_splits}.pth'))
    torch.save((test_X, test_y), os.path.join(output_dir,
                                                f'test_data_{outcome.replace(" ", "_")}_ts{1 - test_size}_rs{seed}_ns{n_splits}.pth'))

    return splits, (test_X, test_y)


def generate_splits(X, y, outcome, test_size, n_splits, seed,
                    test_pids_path=None, train_pids_path=None):
    """
    Splits the input data into training and testing sets based on patient IDs and a specified outcome.

    Args:
        X: Input features DataFrame.
        y: Outcome labels DataFrame.
        outcome: The specific outcome label to consider for splitting.
        test_size (float): The proportion of the dataset to include in the test split.
        n_splits (int): Number of splits for cross-validation.
        seed (int): Random seed for reproducibility.

    If using predefined test and train patient ids:
        test_pids_path (str): Path to the test patient IDs file.
        test_pids_path (str): Path to the test patient IDs file.

    Returns:
        list: A list of tuples containing the split patient IDs for training and validation sets.

    Raises:
        ValueError: If the input data is corrupted or does not meet the required format.
    """

    X['patient_id'] = X['case_admission_id'].apply(lambda x: x.split('_')[0])
    y = y[y.outcome_label == outcome]
    y['patient_id'] = y['case_admission_id'].apply(lambda x: x.split('_')[0])

    """
    SPLITTING DATA
    Splitting is done by patient id (and not admission id) as in case of the rare multiple admissions per patient there
    would be a risk of data leakage otherwise split 'pid' in TRAIN and TEST pid = unique patient_id
    """
    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids = X.patient_id.unique()
    all_outcomes = [1 if pid in y.patient_id.values else 0 for pid in all_pids]

    # Using predefined test and train patient ids
    if test_pids_path is not None:
        pid_test = pd.read_csv(test_pids_path, dtype=str).patient_id.tolist()
        pid_train = pd.read_csv(train_pids_path, dtype=str).patient_id.tolist()

        y_pid_test = [1 if pid in y.patient_id.values else 0 for pid in pid_test]
        y_pid_train = [1 if pid in y.patient_id.values else 0 for pid in pid_train]

    else:
        pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids,
                                                                        all_outcomes,
                                                                        stratify=all_outcomes,
                                                                        test_size=test_size,
                                                                        random_state=seed)

    # define K fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # K-FOLD CROSS-VALIDATION
    splits_idx = []
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]
        splits_idx.append((fold_train_pidx, fold_val_pidx))

    return pid_train, pid_test, splits_idx


if __name__ == '__main__':
    '''
    Example usage:
    python data_splits_to_file.py -f '/Users/jk1/.../preprocessed_features_24022024_133425.csv' -l '/Users/jk1/.../preprocessed_outcomes_24022024_133425.csv' -o 'early_neurological_deterioration' -ptest '/Users/jk1/.../pid_test.tsv' -ptrain '/Users/jk1/.../pid_train.tsv' -od '/Users/jk1/.../train_data_splits' -ts 0.2 -s 42 -ns 5

    If pid_test and pid_train are given, the data is split according to these patient ids
    If pid_test and pid_train are not given, the data is split randomly  
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_path', type=str, required=True)
    parser.add_argument('-l', '--labels_path', type=str, required=True)
    parser.add_argument('-o', '--outcome', type=str, required=True)
    parser.add_argument('-ptest', '--pid_test_path', type=str, required=False, default=None)
    parser.add_argument('-ptrain', '--pid_train_path', type=str, required=False, default=None)
    parser.add_argument('-od', '--output_dir', type=str, required=False, default=None)

    # optional arguments
    parser.add_argument('-ts', '--test_size', type=float, required=False, default=0.2)
    parser.add_argument('-s', '--seed', type=int, required=False, default=42)
    parser.add_argument('-ns', '--n_splits', type=int, required=False, default=5)

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.features_path), f'{"_".join(args.outcome.split(" "))}_train_data_splits')
    else:
        output_dir = args.output_dir

    data_splits_to_file(features_path=args.features_path, labels_path=args.labels_path, outcome=args.outcome,
                        output_dir=output_dir, test_pids_path=args.pid_test_path, train_pids_path=args.pid_train_path,
                        test_size=args.test_size, seed=args.seed, n_splits=args.n_splits)