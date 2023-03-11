import argparse
import json
import os
import numpy as np
import pandas as pd
import itertools
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from prediction.outcome_prediction.data_loading.data_formatting import format_to_linear_table, \
    format_to_2d_table_with_time, link_patient_id_to_outcome, features_to_numpy
from prediction.utils.scoring import precision, recall, matthews, specificity
from prediction.utils.utils import aggregrate_features_over_time

# define constants
n_splits = 5
n_epochs = 5000
seed = 42
test_size = 0.2


# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# define xgboost model function
def create_model(max_depth:int, learning_rate:float, n_estimators:int, feature_aggregation:bool, outcome:str, features_df_path:str,
                 outcomes_df_path:str, output_dir:str):
    np.random.seed(seed)
    optimal_model_df = pd.DataFrame()
    folds = []

    # load and format data
    if not feature_aggregation:
        X, y = format_to_linear_table(features_df_path, outcomes_df_path, outcome)
        X = np.asarray(X).astype('float32')

        # split into training and independent test sets
        X_model, X_test, y_model, y_test, ix_model, ix_test = train_test_split(X, y,
                                                                               range(X.shape[0]), test_size=test_size,
                                                                               random_state=seed)
        # split data further into training and validation sets for each CV
        for train_idx, val_idx in kfold.split(X_model, y_model):
            folds.append((train_idx, val_idx))
    else:
        X, y = format_to_2d_table_with_time(feature_df_path=features_df_path, outcome_df_path=outcomes_df_path,
                                            outcome=outcome)
        # Reduce every patient to a single outcome (to avoid duplicates)
        all_pids_with_outcome = link_patient_id_to_outcome(y, outcome)
        pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                                        all_pids_with_outcome.outcome.tolist(),
                                                                        stratify=all_pids_with_outcome.outcome.tolist(),
                                                                        test_size=test_size,
                                                                        random_state=seed)
        test_X_df = X[X.patient_id.isin(pid_test)]
        test_y_df = y[y.patient_id.isin(pid_test)]

        test_X_np = features_to_numpy(test_X_df,
                                      ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
        y_test = np.array([test_y_df[test_y_df.case_admission_id == cid].outcome.values[0] for cid in
                              test_X_np[:, 0, 0, 0]]).astype('float32')
        # Remove the case_admission_id, sample_label, and time_step_label columns from the data
        X_test = test_X_np[:, :, :, -1].astype('float32')
        X_test, y_test = aggregrate_features_over_time(X_test, y_test)

        for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
            folds.append((fold_pid_train_idx, fold_pid_val_idx))

    # run CV-folds
    i = 0
    for train_idx, val_idx in folds:
        i += 1
        if not feature_aggregation:
            X_train, y_train = X_model[train_idx], y_model[train_idx].ravel()
            X_val, y_val = X_model[val_idx], y_model[val_idx].ravel()

        if feature_aggregation:
            fold_train_pidx = np.array(pid_train)[train_idx]
            fold_val_pidx = np.array(pid_train)[val_idx]

            # split in TRAIN and VALIDATION sets
            fold_X_train_df = X.loc[X.patient_id.isin(fold_train_pidx)]
            fold_y_train_df = y.loc[y.patient_id.isin(fold_train_pidx)]
            fold_X_val_df = X.loc[X.patient_id.isin(fold_val_pidx)]
            fold_y_val_df = y.loc[y.patient_id.isin(fold_val_pidx)]

            # Transform dataframes to numpy arrays
            fold_X_train = features_to_numpy(fold_X_train_df,
                                             ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label',
                                              'value'])
            fold_X_val = features_to_numpy(fold_X_val_df,
                                           ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label',
                                            'value'])

            # collect outcomes for all admissions in train and validation sets
            fold_y_train = np.array(
                [fold_y_train_df[fold_y_train_df.case_admission_id == cid].outcome.values[0] for cid in
                 fold_X_train[:, 0, 0, 0]]).astype('float32')
            fold_y_val = np.array([fold_y_val_df[fold_y_val_df.case_admission_id == cid].outcome.values[0] for cid in
                                   fold_X_val[:, 0, 0, 0]]).astype('float32')

            # Remove the case_admission_id, sample_label, and time_step_label columns from the data
            fold_X_train = fold_X_train[:, :, :, -1].astype('float32')
            fold_X_val = fold_X_val[:, :, :, -1].astype('float32')

            # aggregate features over time so that one timepoint is one sample
            X_train, y_train = aggregrate_features_over_time(fold_X_train, fold_y_train)
            X_val, y_val = aggregrate_features_over_time(fold_X_val, fold_y_val)

        ### MODEL ARCHITECTURE ###
        xgb_model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators,
                                      reg_lambda=50, alpha=70)
        trained_xgb = xgb_model.fit(X_train, y_train, early_stopping_rounds = 50, eval_metric = ["auc"],
                                        eval_set = [(X_train, y_train), (X_val, y_val)])

        # define checkpoint
        if feature_aggregation:
            model_name = f'xgboost_{outcome}_d{max_depth}_nE{n_estimators}_agg'
        else:
            model_name = f'xgboost_{outcome}_d{max_depth}_nE{n_estimators}'
        filepath1 = os.path.join(output_dir, f'{model_name}_{i}.json')
        trained_xgb.save_model(filepath1)

        if feature_aggregation:
            # only keep prediction at last timepoint
            X_test = X_test.reshape(-1, 72, X_test.shape[-1])[:,-1,:].astype('float32')
            y_test = y_test.reshape(-1, 72)[:,-1].astype('float32')
            X_train = X_train.reshape(-1, 72, X_train.shape[-1])[:,-1,:].astype('float32')
            y_train = y_train.reshape(-1, 72)[:,-1].astype('float32')
            X_val = X_val.reshape(-1, 72, X_val.shape[-1])[:,-1,:].astype('float32')
            y_val = y_val.reshape(-1, 72)[:,-1].astype('float32')
        else:
            X_test = X_test[:, :, -1].astype('float32')
        model_y_test = trained_xgb.predict(X_test, iteration_range=(0, trained_xgb.best_iteration + 1)).reshape(len(y_test)).astype('float32')
        model_y_pred_test = np.where(model_y_test > 0.5, 1, 0).astype('float32')
        model_acc_test = accuracy_score(y_test, model_y_pred_test)
        model_precision_test = precision(y_test, model_y_pred_test.astype(float)).numpy()
        model_sn_test = recall(y_test, model_y_pred_test).numpy()
        model_auc_test = roc_auc_score(y_test, model_y_test)
        model_mcc_test = matthews_corrcoef(y_test, model_y_pred_test)
        model_sp_test = specificity(y_test, model_y_pred_test).numpy()

        model_y_val = trained_xgb.predict(X_val, iteration_range=(0, trained_xgb.best_iteration + 1)).reshape(len(y_val)).astype('float32')
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0).astype('float32')
        model_acc_val = accuracy_score(y_val, model_y_pred_val)
        model_precision_val = precision(y_val, model_y_pred_val.astype(float)).numpy()
        model_sn_val = recall(y_val, model_y_pred_val).numpy()
        model_auc_val = roc_auc_score(y_val, model_y_val)
        model_mcc_val = matthews_corrcoef(y_val, model_y_pred_val)
        model_sp_val = specificity(y_val, model_y_pred_val).numpy()

        model_y_train = trained_xgb.predict(X_train, iteration_range=(0, trained_xgb.best_iteration + 1)).reshape(len(y_train)).astype('float32')
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0).astype('float32')
        model_acc_train = accuracy_score(y_train, model_y_pred_train)
        model_precision_train = precision(y_train, model_y_pred_train.astype(float)).numpy()
        model_sn_train = recall(y_train, model_y_pred_train).numpy()
        model_auc_train = roc_auc_score(y_train, model_y_train)
        model_mcc_train = matthews_corrcoef(y_train, model_y_pred_train)
        model_sp_train = specificity(y_train, model_y_pred_train).numpy()

        # save model performance
        run_performance_df = pd.DataFrame(index=[0])
        run_performance_df['epoch'] = trained_xgb.best_iteration
        run_performance_df['CV'] = i
        run_performance_df['max_depth'] = max_depth
        run_performance_df['n_estimators'] = n_estimators
        run_performance_df['outcome'] = outcome
        run_performance_df['auc_train'] = model_auc_train
        run_performance_df['auc_val'] = model_auc_val
        run_performance_df['auc_test'] = model_auc_test
        run_performance_df['mcc_train'] = model_mcc_train
        run_performance_df['mcc_val'] = model_mcc_val
        run_performance_df['mcc_test'] = model_mcc_test
        run_performance_df['acc_train'] = model_acc_train
        run_performance_df['acc_val'] = model_acc_val
        run_performance_df['acc_test'] = model_acc_test
        run_performance_df['precision_train'] = model_precision_train
        run_performance_df['precision_val'] = model_precision_val
        run_performance_df['precision_test'] = model_precision_test
        run_performance_df['sn_train'] = model_sn_train
        run_performance_df['sn_val'] = model_sn_val
        run_performance_df['sn_test'] = model_sn_test
        run_performance_df['sp_train'] = model_sp_train
        run_performance_df['sp_val'] = model_sp_val
        run_performance_df['sp_test'] = model_sp_test

        optimal_model_df = optimal_model_df.append(run_performance_df)

    optimal_model_df.to_csv(os.path.join(output_dir, f'{model_name}.csv'), sep=',', index=False)


def get_model(args):
    create_model(max_depth=args['max_depth'], learning_rate=args['learning_rate'], n_estimators=args['n_estimators'],
                 feature_aggregation=args['feature_aggregation'],
                    outcome = args['outcome'], features_df_path = args['feature_df_path'],
                 outcomes_df_path = args['outcome_df_path'], output_dir = args['output_dir'])
    print('\nDONE: {}'.format(args))


if __name__=='__main__':
    import multiprocessing
    print('multiprocessing loaded')
    ### parse output in parallel

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature_df_path', type=str, help='path to input feature dataframe')
    parser.add_argument('-p', '--outcome_df_path', type=str, help='path to outcome dataframe')
    parser.add_argument('-o', '--outcome', type=str, help='selected outcome')
    parser.add_argument('-O', '--output_dir', type=str, help='Output directory')
    cli_args = parser.parse_args()

    # make parameter dictionary
    param_dict = {}
    param_dict['max_depth'] = [2, 4, 6]
    param_dict['n_estimators'] = [50, 100, 200]
    param_dict['learning_rate'] = [0.1, 0.001]
    param_dict['feature_aggregation'] = [True, False]
    param_dict['outcome'] = [cli_args.outcome]
    param_dict['feature_df_path'] = [cli_args.feature_df_path]
    param_dict['outcome_df_path'] = [cli_args.outcome_df_path]
    param_dict['output_dir'] = [cli_args.output_dir]

    # save parameters as json
    with open(os.path.join(cli_args.output_dir, 'xgboost_parameters.json'), 'w') as f:
        json.dump(param_dict, f)

    # create permutations
    keys, values = zip(*param_dict.items())
    all_args = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # ## run multiprocessing
    number_processes = 1
    pool = multiprocessing.Pool(number_processes)
    # # for args in all_args make model
    pool.map(get_model, all_args)