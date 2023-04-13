import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table, feature_order_verification
from prediction.utils.utils import check_data


def load_data(features_path, labels_path, outcome, test_size, n_splits, seed):
    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

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
    # Preprocess overall train data
    train_X_df = X[X.patient_id.isin(pid_train)]
    train_y_df = y[y.patient_id.isin(pid_train)]
    train_X_np = features_to_numpy(train_X_df,
                                   ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    train_y_np = np.array([train_y_df[train_y_df.case_admission_id == cid].outcome.values[0] for cid in
                           train_X_np[:, 0, 0, 0]]).astype('float32')
    train_X_np = train_X_np[:, :, :, -1].astype('float32')


    # Preprocess overall test data
    test_X_df = X[X.patient_id.isin(pid_test)]
    test_y_df = y[y.patient_id.isin(pid_test)]
    test_X_np = features_to_numpy(test_X_df,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    test_y_np = np.array([test_y_df[test_y_df.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')
    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    test_features_lookup_table = numpy_to_lookup_table(test_X_np)
    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')

    # Preprocess k-fold train/validation data
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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

    return (pid_train, pid_test), (train_X_np, train_y_np), (test_X_np, test_y_np), splits, test_features_lookup_table


def load_external_data(features_path:str, labels_path:str, outcome:str):
    # load the dataset
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

    # test if data is corrupted
    check_data(X)

    test_X_np = features_to_numpy(X,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])

    # ensure that the order of features (3rd dimension) is the one predefined for the model
    feature_order_verification(test_X_np)

    test_y_np = np.array([y[y.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')

    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    test_features_lookup_table = numpy_to_lookup_table(test_X_np)

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')

    return test_X_np, test_y_np, test_features_lookup_table
