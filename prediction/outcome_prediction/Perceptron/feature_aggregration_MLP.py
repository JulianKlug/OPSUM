import argparse
import itertools
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy
from prediction.utils.scoring import precision, recall, specificity
from prediction.utils.utils import aggregrate_features_over_time

# define constants
n_splits = 5
n_epochs = 5000
seed = 42
test_size = 0.2

# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_model(hidden_layer_sizes:tuple, alpha:float, activation:str,
                 outcome:str, features_df_path:str, outcomes_df_path:str, output_dir:str):
    np.random.seed(seed)
    optimal_model_df = pd.DataFrame()

    model_name = f'feature_aggregration_MLP_{outcome}_{"_".join([str(h) for h in hidden_layer_sizes])}_{alpha}_{activation}'

    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_df_path, outcome_df_path=outcomes_df_path,
                                        outcome=outcome)

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

    test_X_df = X[X.patient_id.isin(pid_test)]
    test_y_df = y[y.patient_id.isin(pid_test)]

    test_X_np = features_to_numpy(test_X_df,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    test_y_np = np.array([test_y_df[test_y_df.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')
    X_test, y_test = aggregrate_features_over_time(test_X_np, test_y_np)
    # only keep prediction at last timepoint
    X_test = X_test.reshape(-1, 72, X_test.shape[-1])[:, -1, :].astype('float32')
    y_test = y_test.reshape(-1, 72)[:, -1].astype('float32')

    ### TRAIN MODEL USING K-FOLD CROSS-VALIDATION
    i = 0
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        i += 1

        # find indexes for train/val admissions
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]

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
        fold_y_train = np.array([fold_y_train_df[fold_y_train_df.case_admission_id == cid].outcome.values[0] for cid in
                                 fold_X_train[:, 0, 0, 0]]).astype('float32')
        fold_y_val = np.array([fold_y_val_df[fold_y_val_df.case_admission_id == cid].outcome.values[0] for cid in
                               fold_X_val[:, 0, 0, 0]]).astype('float32')


        # Remove the case_admission_id, sample_label, and time_step_label columns from the data
        fold_X_train = fold_X_train[:, :, :, -1].astype('float32')
        fold_X_val = fold_X_val[:, :, :, -1].astype('float32')

        # aggregate features over time so that one timepoint is one sample
        fold_X_train, fold_y_train = aggregrate_features_over_time(fold_X_train, fold_y_train)
        fold_X_val, fold_y_val = aggregrate_features_over_time(fold_X_val, fold_y_val)

        # Define the model
        mlp_model = MLPClassifier(hidden_layer_sizes, learning_rate='adaptive', alpha=alpha, activation=activation,
                                  max_iter=n_epochs)

        mlp_model.fit(fold_X_train, fold_y_train)

        # only keep prediction at last timepoint
        X_train = fold_X_train.reshape(-1, 72, fold_X_train.shape[-1])[:, -1, :].astype('float32')
        y_train = fold_y_train.reshape(-1, 72)[:, -1].astype('float32')
        X_val = fold_X_val.reshape(-1, 72, fold_X_val.shape[-1])[:, -1, :].astype('float32')
        y_val = fold_y_val.reshape(-1, 72)[:, -1].astype('float32')

        model_y_test = mlp_model.predict_proba(X_test)[:, 1].astype('float32')
        model_y_pred_test = np.where(model_y_test > 0.5, 1, 0).astype('float32')
        model_acc_test = accuracy_score(y_test, model_y_pred_test)
        model_precision_test = precision(y_test, model_y_pred_test.astype(float)).numpy()
        model_sn_test = recall(y_test, model_y_pred_test).numpy()
        model_auc_test = roc_auc_score(y_test, model_y_test)
        model_mcc_test = matthews_corrcoef(y_test, model_y_pred_test)
        model_sp_test = specificity(y_test, model_y_pred_test).numpy()

        model_y_val = mlp_model.predict_proba(X_val)[:, 1].astype('float32')
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0).astype('float32')
        model_acc_val = accuracy_score(y_val, model_y_pred_val)
        model_precision_val = precision(y_val, model_y_pred_val.astype(float)).numpy()
        model_sn_val = recall(y_val, model_y_pred_val).numpy()
        model_auc_val = roc_auc_score(y_val, model_y_val)
        model_mcc_val = matthews_corrcoef(y_val, model_y_pred_val)
        model_sp_val = specificity(y_val, model_y_pred_val).numpy()

        model_y_train = mlp_model.predict_proba(X_train)[:, 1].astype('float32')
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0).astype('float32')
        model_acc_train = accuracy_score(y_train, model_y_pred_train)
        model_precision_train = precision(y_train, model_y_pred_train.astype(float)).numpy()
        model_sn_train = recall(y_train, model_y_pred_train).numpy()
        model_auc_train = roc_auc_score(y_train, model_y_train)
        model_mcc_train = matthews_corrcoef(y_train, model_y_pred_train)
        model_sp_train = specificity(y_train, model_y_pred_train).numpy()

        # save model performance
        run_performance_df = pd.DataFrame(index=[0])
        run_performance_df['CV'] = i
        run_performance_df['hidden_layer_sizes'] = str(hidden_layer_sizes)
        run_performance_df['activation'] = activation
        run_performance_df['alpha'] = alpha
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

def evaluate_args(args):
    evaluate_model(hidden_layer_sizes=args['hidden_layer_sizes'], alpha = args['alpha'], activation = args['activation'],
                    outcome = args['outcome'], features_df_path = args['feature_df_path'],
                 outcomes_df_path = args['outcome_df_path'], output_dir = args['output_dir'])
    print('\nDONE: {}'.format(args))


if __name__=='__main__':
    # TODO: modify
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
    param_dict['layer_sizes'] = [(128, 128, 64), (128, 64, 32), (256,128,64))
    param_dict['alpha'] = [0.0001, 0.05, 0,1, 1, 12.5]
    param_dict['activation'] = ['relu', 'tanh']
    param_dict['feature_aggregation'] = [True]
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
    number_processes = 30
    pool = multiprocessing.Pool(number_processes)
    # # for args in all_args make model
    pool.map(getimport argparse
import itertools
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy
from prediction.utils.scoring import precision, recall, specificity
from prediction.utils.utils import aggregrate_features_over_time

# define constants
n_splits = 5
n_epochs = 5000
seed = 42
test_size = 0.2

# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_model(hidden_layer_sizes:tuple, alpha:float, activation:str,
                 outcome:str, features_df_path:str, outcomes_df_path:str, output_dir:str):
    np.random.seed(seed)
    optimal_model_df = pd.DataFrame()

    model_name = f'feature_aggregration_MLP_{outcome}_{"_".join([str(h) for h in hidden_layer_sizes])}_{alpha}_{activation}'

    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_df_path, outcome_df_path=outcomes_df_path,
                                        outcome=outcome)

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

    test_X_df = X[X.patient_id.isin(pid_test)]
    test_y_df = y[y.patient_id.isin(pid_test)]

    test_X_np = features_to_numpy(test_X_df,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    test_y_np = np.array([test_y_df[test_y_df.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')
    X_test, y_test = aggregrate_features_over_time(test_X_np, test_y_np)
    # only keep prediction at last timepoint
    X_test = X_test.reshape(-1, 72, X_test.shape[-1])[:, -1, :].astype('float32')
    y_test = y_test.reshape(-1, 72)[:, -1].astype('float32')

    ### TRAIN MODEL USING K-FOLD CROSS-VALIDATION
    i = 0
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        i += 1

        # find indexes for train/val admissions
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]

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
        fold_y_train = np.array([fold_y_train_df[fold_y_train_df.case_admission_id == cid].outcome.values[0] for cid in
                                 fold_X_train[:, 0, 0, 0]]).astype('float32')
        fold_y_val = np.array([fold_y_val_df[fold_y_val_df.case_admission_id == cid].outcome.values[0] for cid in
                               fold_X_val[:, 0, 0, 0]]).astype('float32')


        # Remove the case_admission_id, sample_label, and time_step_label columns from the data
        fold_X_train = fold_X_train[:, :, :, -1].astype('float32')
        fold_X_val = fold_X_val[:, :, :, -1].astype('float32')

        # aggregate features over time so that one timepoint is one sample
        fold_X_train, fold_y_train = aggregrate_features_over_time(fold_X_train, fold_y_train)
        fold_X_val, fold_y_val = aggregrate_features_over_time(fold_X_val, fold_y_val)

        # Define the model
        mlp_model = MLPClassifier(hidden_layer_sizes, learning_rate='adaptive', alpha=alpha, activation=activation,
                                  max_iter=n_epochs)

        mlp_model.fit(fold_X_train, fold_y_train)

        # only keep prediction at last timepoint
        X_train = fold_X_train.reshape(-1, 72, fold_X_train.shape[-1])[:, -1, :].astype('float32')
        y_train = fold_y_train.reshape(-1, 72)[:, -1].astype('float32')
        X_val = fold_X_val.reshape(-1, 72, fold_X_val.shape[-1])[:, -1, :].astype('float32')
        y_val = fold_y_val.reshape(-1, 72)[:, -1].astype('float32')

        model_y_test = mlp_model.predict_proba(X_test)[:, 1].astype('float32')
        model_y_pred_test = np.where(model_y_test > 0.5, 1, 0).astype('float32')
        model_acc_test = accuracy_score(y_test, model_y_pred_test)
        model_precision_test = precision(y_test, model_y_pred_test.astype(float)).numpy()
        model_sn_test = recall(y_test, model_y_pred_test).numpy()
        model_auc_test = roc_auc_score(y_test, model_y_test)
        model_mcc_test = matthews_corrcoef(y_test, model_y_pred_test)
        model_sp_test = specificity(y_test, model_y_pred_test).numpy()

        model_y_val = mlp_model.predict_proba(X_val)[:, 1].astype('float32')
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0).astype('float32')
        model_acc_val = accuracy_score(y_val, model_y_pred_val)
        model_precision_val = precision(y_val, model_y_pred_val.astype(float)).numpy()
        model_sn_val = recall(y_val, model_y_pred_val).numpy()
        model_auc_val = roc_auc_score(y_val, model_y_val)
        model_mcc_val = matthews_corrcoef(y_val, model_y_pred_val)
        model_sp_val = specificity(y_val, model_y_pred_val).numpy()

        model_y_train = mlp_model.predict_proba(X_train)[:, 1].astype('float32')
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0).astype('float32')
        model_acc_train = accuracy_score(y_train, model_y_pred_train)
        model_precision_train = precision(y_train, model_y_pred_train.astype(float)).numpy()
        model_sn_train = recall(y_train, model_y_pred_train).numpy()
        model_auc_train = roc_auc_score(y_train, model_y_train)
        model_mcc_train = matthews_corrcoef(y_train, model_y_pred_train)
        model_sp_train = specificity(y_train, model_y_pred_train).numpy()

        # save model performance
        run_performance_df = pd.DataFrame(index=[0])
        run_performance_df['CV'] = i
        run_performance_df['hidden_layer_sizes'] = str(hidden_layer_sizes)
        run_performance_df['activation'] = activation
        run_performance_df['alpha'] = alpha
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

def evaluate_args(args):
    evaluate_model(hidden_layer_sizes=args['hidden_layer_sizes'], alpha = args['alpha'], activation = args['activation'],
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
    param_dict['layer_sizes'] = [(128, 128, 64), (128, 64, 32), (256,128,64))
    param_dict['alpha'] = [0.0001, 0.05, 0,1, 1, 12.5]
    param_dict['activation'] = ['relu', 'tanh']
    param_dict['feature_aggregation'] = [True]
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
    pool.map(evaluate_args, all_args)



features_df_path = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/preprocessed_features_01012023_233050.csv'
outcomes_df_path = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/preprocessed_outcomes_01012023_233050.csv'
outcome = '3M mRS 0-2'
output_dir = '/Users/jk1/temp/opsum_prediction_output/linear_72h_perceptron/agregrated_mlp_test'


