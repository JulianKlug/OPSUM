import argparse
import json
import os
import numpy as np
import pandas as pd
import itertools
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_linear_table
from prediction.utils.scoring import precision, recall, matthews


# define constants
n_splits = 5
n_epochs = 5000
seed = 1234

# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# define xgboost model function
def create_model(max_depth:int, learning_rate:float, n_estimators:int, outcome:str, features_df_path:str,
                 outcomes_df_path:str, output_dir:str):
    np.random.seed(seed)
    optimal_model_df = pd.DataFrame()

    # load and format data
    X, y = format_to_linear_table(features_df_path, outcomes_df_path, outcome)
    X = np.asarray(X).astype('float32')

    # split into training and independent test sets
    X_model, X_test, y_model, y_test, ix_model, ix_test = train_test_split(X, y,
                                                                           range(X.shape[0]), test_size=0.15,
                                                                           random_state=seed)

    # run CV-folds
    i=0
    # split data further into training and validation sets for each CV
    for train_idx, val_idx in kfold.split(X_model, y_model):
        X_train, y_train = X_model[train_idx], y_model[train_idx].ravel()
        X_val, y_val = X_model[val_idx], y_model[val_idx].ravel()
        i += 1

        ### MODEL ARCHITECTURE ###
        xgb_model = xgb.XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        trained_xgb = xgb_model.fit(X_train, y_train, early_stopping_rounds = 50, eval_metric = ["auc"],
                                        eval_set = [(X_train, y_train), (X_val, y_val)])

        # define checkpoint
        filepath1 = os.path.join(output_dir, f'xgboost_{outcome}_d{max_depth}_nE{n_estimators}_{i}.json')
        trained_xgb.save_model(filepath1)

        model_y_test = trained_xgb.predict(X_test, iteration_range=(0, trained_xgb.best_iteration + 1)).reshape(len(y_test))
        model_y_pred_test = np.where(model_y_test > 0.5, 1, 0)
        model_acc_test = accuracy_score(y_test, model_y_pred_test)
        model_precision_test = precision(y_test, model_y_pred_test.astype(float)).numpy()
        model_recall_test = recall(y_test, model_y_pred_test).numpy()
        model_auc_test = roc_auc_score(y_test, model_y_test)
        model_mcc_test = matthews_corrcoef(y_test, model_y_pred_test)
        model_y_val = trained_xgb.predict(X_val, iteration_range=(0, trained_xgb.best_iteration + 1)).reshape(len(y_val))
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0)
        model_acc_val = accuracy_score(y_val, model_y_pred_val)
        model_precision_val = precision(y_val, model_y_pred_val.astype(float)).numpy()
        model_recall_val = recall(y_val, model_y_pred_val).numpy()
        model_auc_val = roc_auc_score(y_val, model_y_val)
        model_mcc_val = matthews_corrcoef(y_val, model_y_pred_val)
        model_y_train = trained_xgb.predict(X_train, iteration_range=(0, trained_xgb.best_iteration + 1)).reshape(len(y_train))
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0)
        model_acc_train = accuracy_score(y_train, model_y_pred_train)
        model_precision_train = precision(y_train, model_y_pred_train.astype(float)).numpy()
        model_recall_train = recall(y_train, model_y_pred_train).numpy()
        model_auc_train = roc_auc_score(y_train, model_y_train)
        model_mcc_train = matthews_corrcoef(y_train, model_y_pred_train)

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
        run_performance_df['recall_train'] = model_recall_train
        run_performance_df['recall_val'] = model_recall_val
        run_performance_df['recall_test'] = model_recall_test

        optimal_model_df = optimal_model_df.append(run_performance_df)

    optimal_model_df.to_csv(os.path.join(output_dir, f'xgboost_{outcome}_d{max_depth}_nE{n_estimators}.tsv'), sep='\t', index=False)


def get_model(args):
    create_model(max_depth=args['max_depth'], learning_rate=args['learning_rate'], n_estimators=args['n_estimators'],
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
    param_dict['learning_rate'] = [0.001]
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
    pool.map(get_model, all_args)