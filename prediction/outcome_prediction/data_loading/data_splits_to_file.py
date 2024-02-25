import numpy as np
import pandas as pd
import torch
import os
import json
from sklearn.model_selection import train_test_split, StratifiedKFold

from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy
from prediction.utils.utils import check_data, ensure_dir


def save_train_splits(features_path: str, labels_path:str, outcome:str, output_dir:str,
                      test_pids_path:str=None, train_pids_path:str=None,
                      test_size:float=0.2, seed=42, n_splits=5):
    ensure_dir(output_dir)
    # save all arguments given to function as json
    args = locals()
    # remove 'args' from args to avoid circular reference
    args = {k: v for k, v in args.items() if k != 'args'}

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    # test if data is corrupted
    check_data(X)

    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids_with_outcome = link_patient_id_to_outcome(y, outcome)

    # Using predefined test and train patient ids
    if test_pids_path is not None:
        pid_test = pd.read_csv(test_pids_path, sep='\t', dtype=str).patient_id.tolist()
        pid_train = pd.read_csv(train_pids_path, sep='\t', dtype=str).patient_id.tolist()

        # Reduce to pids actually present in the data
        pid_test = [pid for pid in pid_test if pid in all_pids_with_outcome.patient_id.tolist()]
        pid_train = [pid for pid in pid_train if pid in all_pids_with_outcome.patient_id.tolist()]

        y_pid_test = all_pids_with_outcome[all_pids_with_outcome.patient_id.isin(pid_test)].outcome.tolist()
        y_pid_train = all_pids_with_outcome[all_pids_with_outcome.patient_id.isin(pid_train)].outcome.tolist()

    else:
        """
        SPLITTING DATA
        Splitting is done by patient id (and not admission id) as in case of the rare multiple admissions per patient there
        would be a risk of data leakage otherwise split 'pid' in TRAIN and TEST pid = unique patient_id
        """
        pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                                        all_pids_with_outcome.outcome.tolist(),
                                                                        stratify=all_pids_with_outcome.outcome.tolist(),
                                                                        test_size=test_size,
                                                                        random_state=seed)

    # save patient ids used for testing / training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.tsv'),
        sep='\t', index=False)
    pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_test.tsv'),
        sep='\t', index=False)

    test_X = X[X.patient_id.isin(pid_test)]
    # Here test data is not needed anymore, but for reference should be loaded as such: test_y = y[y.patient_id.isin(pid_test)]

    # define K fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


    ### TRAIN MODEL USING K-FOLD CROSS-VALIDATION
    splits = []
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]

        fold_X_train_df = X.loc[X.patient_id.isin(fold_train_pidx)]
        fold_y_train_df = y.loc[y.patient_id.isin(fold_train_pidx)]
        fold_X_val_df = X.loc[X.patient_id.isin(fold_val_pidx)]
        fold_y_val_df = y.loc[y.patient_id.isin(fold_val_pidx)]

        fold_X_train = features_to_numpy(fold_X_train_df, ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
        fold_X_val = features_to_numpy(fold_X_val_df, ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])

        fold_y_train = np.array([fold_y_train_df[fold_y_train_df.case_admission_id == cid].outcome.values[0] for cid in fold_X_train[:, 0, 0, 0]]).astype('float32')
        fold_y_val = np.array([fold_y_val_df[fold_y_val_df.case_admission_id == cid].outcome.values[0] for cid in fold_X_val[:, 0, 0, 0]]).astype('float32')

        fold_X_train = fold_X_train[:, :, :, -1].astype('float32')
        fold_X_val = fold_X_val[:, :, :, -1].astype('float32')

        splits.append((fold_X_train, fold_X_val, fold_y_train, fold_y_val))

    torch.save(splits, os.path.join(output_dir, f'train_data_splits_{outcome.replace(" ", "_")}_ts{1-test_size}_rs{seed}_ns{n_splits}.pth'))
    return splits


if __name__ == '__main__':
    '''
    Example usage:
    python data_splits_to_file.py -f '/Users/jk1/.../preprocessed_features_24022024_133425.csv' -l '/Users/jk1/.../preprocessed_outcomes_24022024_133425.csv' -o '3M mRS 0-2' -ptest '/Users/jk1/.../pid_test.tsv' -ptrain '/Users/jk1/.../pid_train.tsv' -od '/Users/jk1/.../train_data_splits' -ts 0.2 -s 42 -ns 5
    
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

    save_train_splits(features_path=args.features_path, labels_path=args.labels_path, outcome=args.outcome,
                        output_dir=output_dir, test_pids_path=args.pid_test_path, train_pids_path=args.pid_train_path,
                        test_size=args.test_size, seed=args.seed, n_splits=args.n_splits)
