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
                      test_size:float=0.2, seed=42, n_splits=5):
    ensure_dir(output_dir)
    # save all arguments given to function as json
    args = locals()
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    # test if data is corrupted
    check_data(X)

    """
    SPLITTING DATA
    Splitting is done by patient id (and not admission id) as in case of the rare multiple admissions per patient there
    would be a risk of data leakage otherwise split 'pid' in TRAIN and TEST pid = unique patient_id
    """
    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids_with_outcome = link_patient_id_to_outcome(y, outcome)
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
    features_path = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/preprocessed_features_01012023_233050.csv'
    labels_path = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/preprocessed_outcomes_01012023_233050.csv'
    output_dir = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/data_splits'
    outcome = 'Death in hospital'
    save_train_splits(features_path, labels_path, outcome, output_dir)
