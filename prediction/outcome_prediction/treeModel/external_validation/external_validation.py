import argparse
import shutil

import pandas as pd
import numpy as np
import os
import pickle
import xgboost as xgb
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix

from prediction.outcome_prediction.data_loading.data_loader import load_external_data
from prediction.utils.utils import save_json, aggregate_features_over_time


def external_validation(outcome:str, model_weights_path:str, external_features_df_path:str, external_outcomes_df_path:str, output_dir:str, config:dict):
    xgb_model = xgb.XGBClassifier(learning_rate=config['learning_rate'], max_depth=config['max_depth'],
                                  n_estimators=config['n_estimators'], reg_lambda=config['reg_lambda'],
                                  alpha=config['alpha'])

    xgb_model.load_model(model_weights_path)

    # copy model weights to output dir
    shutil.copy(model_weights_path, os.path.join(output_dir, 'model_weights.json'))

    # load external test data
    test_X_np, test_y_np, test_features_lookup_table = load_external_data(external_features_df_path, external_outcomes_df_path,
                                                                          outcome)

    X_test, y_test = aggregate_features_over_time(test_X_np, test_y_np, moving_average=config['moving_average'])
    # only keep prediction at last timepoint
    X_test = X_test.reshape(-1, 72, X_test.shape[-1])[:, -1, :].astype('float32')
    y_test = y_test.reshape(-1, 72)[:, -1].astype('float32')

    save_json(test_features_lookup_table,
              os.path.join(output_dir, 'test_lookup_dict.json'))

    # calculate overall model prediction
    y_pred_test = xgb_model.predict_proba(X_test)[:, 1].astype('float32')

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

    n_iterations = 1000
    for i in tqdm(range(n_iterations)):
        X_bs, y_bs = resample(X_test, y_test, replace=True)
        # make predictions
        y_pred_bs = xgb_model.predict_proba(X_bs)[:, 1].astype('float32')
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
        recall_bs = recall_score(y_bs, y_pred_bs_binary) # == sensitivity
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

    result_df = pd.DataFrame([{
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
        'outcome': outcome,
        'model_weights_path': output_dir
    }], index=[0])


    return result_df, (bootstrapped_ground_truth, bootstrapped_predictions), (y_test, y_pred_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ef', '--external_feature_df_path', type=str, help='path to input feature dataframe')
    parser.add_argument('-ep', '--external_outcome_df_path', type=str, help='path to outcome dataframe')
    parser.add_argument('-o', '--outcome', type=str, help='selected outcome')
    parser.add_argument('-O', '--output_dir', type=str, help='Output directory')
    parser.add_argument('--model_weights_path', required=True, type=str, help='path to model weights')
    parser.add_argument('--parameters_path', required=True, type=str, help='path to model parameters')

    cli_args = parser.parse_args()

    parameters_df = pd.read_csv(cli_args.parameters_path)
    config = parameters_df.squeeze().to_dict()
    # copy parameter_df to output directory
    shutil.copyfile(cli_args.parameters_path, os.path.join(cli_args.output_dir, 'parameters.csv'))

    if 'moving_average' in parameters_df:
        moving_average = parameters_df['moving_average'][0]
    else:
        moving_average = False

    result_df, bootstrapping_data, testing_data = external_validation(cli_args.outcome,
                                                                        cli_args.model_weights_path,
                                                                      cli_args.external_feature_df_path,
                                                                        cli_args.external_outcome_df_path,
                                                                        cli_args.output_dir,
                                                                      config)

    result_df.to_csv(os.path.join(cli_args.output_dir, 'external_validation_XGB_results.csv'), sep=',', index=False)

    # save bootstrapped ground truth and predictions
    pickle.dump(bootstrapping_data, open(os.path.join(cli_args.output_dir, 'bootstrapped_gt_and_pred.pkl'), 'wb'))
    pickle.dump(testing_data, open(os.path.join(cli_args.output_dir, 'external_validation_gt_and_pred.pkl'), 'wb'))