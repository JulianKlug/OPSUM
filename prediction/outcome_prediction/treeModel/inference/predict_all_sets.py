import glob
import os
import pickle

import xgboost as xgb
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
from sklearn.utils import resample
import numpy as np
import pandas as pd
from tqdm import tqdm

from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.utils.utils import aggregate_features_over_time


def evaluate_xgb(model, X, y, bootstrapped:bool=False):
    # calculate overall model prediction
    y_pred_test = model.predict_proba(X)[:, 1].astype('float32')

    # Bootstrapped testing
    roc_auc_scores = []
    matthews_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    neg_pred_value_scores = []

    bootstrapped_ground_truth = []
    bootstrapped_predictions = []
    bootstrapped_result_df = pd.DataFrame()
    if bootstrapped:
        n_iterations = 1000
        for i in tqdm(range(n_iterations)):
            X_bs, y_bs = resample(X, y, replace=True)
            # make predictions
            y_pred_bs = model.predict_proba(X_bs)[:, 1].astype('float32')
            y_pred_bs_binary = (y_pred_bs > 0.5).astype('int32')

            bootstrapped_ground_truth.append(y_bs)
            bootstrapped_predictions.append(y_pred_bs)

            # evaluate model
            roc_auc_bs = roc_auc_score(y_bs, y_pred_bs)
            roc_auc_scores.append(roc_auc_bs)
            matthews_bs = matthews_corrcoef(y_bs, y_pred_bs_binary)
            matthews_scores.append(matthews_bs)
            accuracy_bs = accuracy_score(y_bs, y_pred_bs_binary)
            accuracy_scores.append(accuracy_bs)
            precision_bs = precision_score(y_bs, y_pred_bs_binary)  # == PPV
            recall_bs = recall_score(y_bs, y_pred_bs_binary)  # == sensitivity
            precision_scores.append(precision_bs)
            recall_scores.append(recall_bs)

            mcm = multilabel_confusion_matrix(y_bs, y_pred_bs_binary)
            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]
            specificity_bs = tn / (tn + fp)
            specificity_scores.append(specificity_bs)
            neg_pred_value_bs = tn / (tn + fn)
            neg_pred_value_scores.append(neg_pred_value_bs)

        # get medians
        median_roc_auc = np.percentile(roc_auc_scores, 50)
        median_matthews = np.percentile(matthews_scores, 50)
        median_accuracy = np.percentile(accuracy_scores, 50)
        median_precision = np.percentile(precision_scores, 50)
        median_recall = np.percentile(recall_scores, 50)
        median_specificity = np.percentile(specificity_scores, 50)
        median_neg_pred_value = np.percentile(neg_pred_value_scores, 50)

        # get 95% interval
        alpha = 100 - 95
        lower_ci_roc_auc = np.percentile(roc_auc_scores, alpha / 2)
        upper_ci_roc_auc = np.percentile(roc_auc_scores, 100 - alpha / 2)
        lower_ci_matthews = np.percentile(matthews_scores, alpha / 2)
        upper_ci_matthews = np.percentile(matthews_scores, 100 - alpha / 2)
        lower_ci_accuracy = np.percentile(accuracy_scores, alpha / 2)
        upper_ci_accuracy = np.percentile(accuracy_scores, 100 - alpha / 2)
        lower_ci_precision = np.percentile(precision_scores, alpha / 2)
        upper_ci_precision = np.percentile(precision_scores, 100 - alpha / 2)
        lower_ci_recall = np.percentile(recall_scores, alpha / 2)
        upper_ci_recall = np.percentile(recall_scores, 100 - alpha / 2)
        lower_ci_specificity = np.percentile(specificity_scores, alpha / 2)
        upper_ci_specificity = np.percentile(specificity_scores, 100 - alpha / 2)
        lower_ci_neg_pred_value = np.percentile(neg_pred_value_scores, alpha / 2)
        upper_ci_neg_pred_value = np.percentile(neg_pred_value_scores, 100 - alpha / 2)

        bootstrapped_result_df = pd.DataFrame([{
            'auc_test': median_roc_auc,
            'auc_test_lower_ci': lower_ci_roc_auc,
            'auc_test_upper_ci': upper_ci_roc_auc,
            'matthews_test': median_matthews,
            'matthews_test_lower_ci': lower_ci_matthews,
            'matthews_test_upper_ci': upper_ci_matthews,
            'accuracy_test': median_accuracy,
            'accuracy_test_lower_ci': lower_ci_accuracy,
            'accuracy_test_upper_ci': upper_ci_accuracy,
            'precision_test': median_precision,
            'precision_test_lower_ci': lower_ci_precision,
            'precision_test_upper_ci': upper_ci_precision,
            'recall_test': median_recall,
            'recall_test_lower_ci': lower_ci_recall,
            'recall_test_upper_ci': upper_ci_recall,
            'specificity_test': median_specificity,
            'specificity_test_lower_ci': lower_ci_specificity,
            'specificity_test_upper_ci': upper_ci_specificity,
            'neg_pred_value_test': median_neg_pred_value,
            'neg_pred_value_test_lower_ci': lower_ci_neg_pred_value,
            'neg_pred_value_test_upper_ci': upper_ci_neg_pred_value,
        }], index=[0])

    return (bootstrapped_result_df, (bootstrapped_ground_truth, bootstrapped_predictions), (y, y_pred_test))



def predict_all_sets(outcome:str, model_weights_dir:str, features_df_path:str, outcomes_df_path:str, output_dir:str, config:dict,
                     test_size:float=0.2, n_splits:int=5, seed:int=42, bootstrapped:bool=False):

    if 'moving_average' in config.keys():
        config['moving_average'] = bool(config['moving_average'])
    else:
        config['moving_average'] = False


    # features_path, labels_path, outcome, test_size, n_splits, seed
    ((pid_train, pid_test), (train_X_np, train_y_np), (test_X_np, test_y_np),
        splits, test_features_lookup_table) = load_data(features_df_path, outcomes_df_path, outcome, test_size, n_splits, seed)

    train_results_df = pd.DataFrame()
    val_results_df = pd.DataFrame()
    for split_idx in tqdm(range(n_splits)):
        (fold_X_train, fold_X_val, fold_y_train, fold_y_val) = splits[split_idx]

        # Feature aggregration
        fold_X_train, fold_y_train = aggregate_features_over_time(fold_X_train, fold_y_train, moving_average=config['moving_average'])
        # only keep prediction at last timepoint
        fold_X_train = fold_X_train.reshape(-1, 72, fold_X_train.shape[-1])[:, -1, :].astype('float32')
        fold_y_train = fold_y_train.reshape(-1, 72)[:, -1].astype('float32')

        fold_X_val, fold_y_val = aggregate_features_over_time(fold_X_val, fold_y_val, moving_average=config['moving_average'])
        # only keep prediction at last timepoint
        fold_X_val = fold_X_val.reshape(-1, 72, fold_X_val.shape[-1])[:, -1, :].astype('float32')
        fold_y_val = fold_y_val.reshape(-1, 72)[:, -1].astype('float32')

        # find all model weights files
        model_weights_path = glob.glob(os.path.join(model_weights_dir, f'*cv{split_idx}.json'))[0]

        xgb_model = xgb.XGBClassifier(learning_rate=config['learning_rate'], max_depth=config['max_depth'],
                                      n_estimators=config['n_estimators'], reg_lambda=config['reg_lambda'],
                                      alpha=config['alpha'])

        xgb_model.load_model(model_weights_path)

        val_fold_result_df, val_fold_bootstrapped_gt_and_pred, val_fold_overall_gt_and_pred = evaluate_xgb(xgb_model, fold_X_val, fold_y_val,
                                                                                               bootstrapped=bootstrapped)
        train_fold_result_df, train_fold_bootstrapped_gt_and_pred, train_fold_overall_gt_and_pred = evaluate_xgb(xgb_model, fold_X_train, fold_y_train,
                                                                                                  bootstrapped=bootstrapped)

        val_fold_result_df['fold'] = split_idx
        train_fold_result_df['fold'] = split_idx

        val_results_df = pd.concat([val_results_df, val_fold_result_df])
        train_results_df = pd.concat([train_results_df, train_fold_result_df])

        pickle.dump(train_fold_overall_gt_and_pred, open(os.path.join(output_dir, f'train_gt_and_pred_cv_{split_idx}.pkl'), 'wb'))
        pickle.dump(val_fold_overall_gt_and_pred, open(os.path.join(output_dir, f'val_gt_and_pred_cv_{split_idx}.pkl'), 'wb'))

        if bootstrapped:
            pickle.dump(train_fold_bootstrapped_gt_and_pred, open(os.path.join(output_dir, f'train_bootstrapped_gt_and_pred_cv_{split_idx}.pkl'), 'wb'))
            pickle.dump(val_fold_bootstrapped_gt_and_pred, open(os.path.join(output_dir, f'val_bootstrapped_gt_and_pred_cv_{split_idx}.pkl'), 'wb'))

    # save results
    train_results_df.to_csv(os.path.join(output_dir, 'train_results.csv'), index=False)
    val_results_df.to_csv(os.path.join(output_dir, 'val_results.csv'), index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_df_path', type=str, help='path to input feature dataframe')
    parser.add_argument('-l', '--labels_df_path', type=str, help='path to outcome dataframe')
    parser.add_argument('-o', '--outcome', type=str, help='selected outcome')
    parser.add_argument('-w', '--model_weights_dir', type=str, help='path to model weights directory')
    parser.add_argument('-c', '--model_config_path', type=str, help='path to model config')
    parser.add_argument('-O', '--output_dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--bootstrapped', action='store_true', help='whether to use bootstrapped testing')

    cli_args = parser.parse_args()

    model_config = pd.read_csv(cli_args.model_config_path).squeeze().to_dict()

    predict_all_sets(outcome=cli_args.outcome, model_weights_dir=cli_args.model_weights_dir,
                     features_df_path=cli_args.features_df_path, outcomes_df_path=cli_args.labels_df_path,
                     output_dir=cli_args.output_dir, config=model_config, test_size=cli_args.test_size, n_splits=cli_args.n_splits,
                     seed=cli_args.seed, bootstrapped=cli_args.bootstrapped)

