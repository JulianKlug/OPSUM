import argparse
import shutil

import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
from prediction.outcome_prediction.Perceptron.training.feature_aggregration_MLP import evaluate_model
from prediction.outcome_prediction.data_loading.data_loader import load_external_data
from prediction.utils.utils import aggregate_features_over_time, save_json


def external_validation(model_weights_dir:str, best_parameters_path:str, outcome:str, external_features_df_path:str,
                        external_outcomes_df_path:str, output_dir:str):
    best_parameters_df = pd.read_csv(best_parameters_path)

    # load external test data
    test_X_np, test_y_np, test_features_lookup_table = load_external_data(external_features_df_path,
                                                                          external_outcomes_df_path,
                                                                          outcome)

    X_test, y_test = aggregate_features_over_time(test_X_np, test_y_np)
    # only keep prediction at last timepoint
    X_test = X_test.reshape(-1, 72, X_test.shape[-1])[:, -1, :].astype('float32')
    y_test = y_test.reshape(-1, 72)[:, -1].astype('float32')

    save_json(test_features_lookup_table,
              os.path.join(output_dir, 'test_lookup_dict.json'))


    fold_model_paths = [path for path in os.listdir(model_weights_dir) if (path.endswith('.pkl')) and (path.startswith('feature_aggregration_MLP'))]
    overall_results_df = pd.DataFrame()
    for model_path in fold_model_paths:

        split_idx = model_path.split('.pkl')[0].split('cv')[1]
        trained_model = pickle.load(open(os.path.join(model_weights_dir, model_path), 'rb'))

        # calculate overall model prediction
        y_pred_test = trained_model.predict_proba(X_test)[:, 1].astype('float32')

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
            y_pred_bs = trained_model.predict_proba(X_bs)[:, 1].astype('float32')
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
            'fold': split_idx,
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
            'model_weights_path': model_path,
            'best_val_fold': int(best_parameters_df['CV'].values[0])
        }], index=[0])

        overall_results_df = pd.concat([overall_results_df, result_df])

        # save bootstrapped ground truth and predictions
        pickle.dump((bootstrapped_ground_truth, bootstrapped_predictions),
                    open(os.path.join(output_dir, f'fold_{split_idx}_bootstrapped_gt_and_pred.pkl'), 'wb'))
        pickle.dump((y_test, y_pred_test), open(os.path.join(output_dir, f'fold_{split_idx}_test_gt_and_pred.pkl'), 'wb'))

    overall_results_df.to_csv(os.path.join(output_dir, 'overall_results.csv'), sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature_df_path', type=str, help='path to input feature dataframe')
    parser.add_argument('-p', '--outcome_df_path', type=str, help='path to outcome dataframe')
    parser.add_argument('-o', '--outcome', type=str, help='selected outcome')
    parser.add_argument('-O', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-b', '--best_parameters_path', type=str, help='path to best parameters file')
    parser.add_argument('-w', '--model_weights_dir', type=str, help='directory containing model weights')
    cli_args = parser.parse_args()

    external_validation(cli_args.model_weights_dir, cli_args.best_parameters_path, cli_args.outcome, cli_args.feature_df_path, cli_args.outcome_df_path, cli_args.output_dir)

