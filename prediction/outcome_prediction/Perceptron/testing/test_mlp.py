import argparse
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
from prediction.outcome_prediction.Perceptron.training.feature_aggregration_MLP import evaluate_model


def test_model(hidden_layer_sizes:tuple, alpha:float, activation:str,
                 outcome:str, features_df_path:str, outcomes_df_path:str, output_dir:str, all_folds:bool=False):
    optimal_model_df, trained_models, test_dataset = evaluate_model(hidden_layer_sizes, alpha, activation,
                                                                    outcome, features_df_path, outcomes_df_path, output_dir, save_models=True)

    X_test, y_test = test_dataset

    best_cv_fold = int(optimal_model_df.sort_values(by='auc_val', ascending=False).iloc[0]['CV'])
    best_cv_fold_idx = best_cv_fold - 1

    if all_folds:
        fold_models = trained_models
    else:
        # Select model
        selected_model = trained_models[best_cv_fold_idx]
        fold_models = [selected_model]

    fold_results = []
    for model in fold_models:
        # calculate overall model prediction
        y_pred_test = model.predict_proba(X_test)[:, 1].astype('float32')

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

        result_df = pd.concat([result_df, best_parameters_df], axis=1)
        fold_results.append((result_df, (bootstrapped_ground_truth, bootstrapped_predictions), (y_test, y_pred_test)))

    return fold_results, best_cv_fold_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature_df_path', type=str, help='path to input feature dataframe')
    parser.add_argument('-p', '--outcome_df_path', type=str, help='path to outcome dataframe')
    parser.add_argument('-o', '--outcome', type=str, help='selected outcome')
    parser.add_argument('-O', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-b', '--best_parameters_path', type=str, help='path to best parameters file')
    parser.add_argument('-a', '--all_folds', type=bool, default=False, help='whether to test on all folds or just the best one')
    cli_args = parser.parse_args()

    best_parameters_df = pd.read_csv(cli_args.best_parameters_path)


    hidden_layer_sizes = [int(size) for size in best_parameters_df['hidden_layer_sizes'][0][1:-1].replace(' ', '').split(',')]
    fold_results, best_cv_fold_idx = test_model(hidden_layer_sizes, best_parameters_df['alpha'][0], best_parameters_df['activation'][0],
                                                                cli_args.outcome, cli_args.feature_df_path, cli_args.outcome_df_path, cli_args.output_dir,
                                                                cli_args.all_folds)

    # save results
    for fidx in range(len(fold_results)):
        result_df = fold_results[fidx][0]
        bootstrapping_data = fold_results[fidx][1]
        testing_data = fold_results[fidx][2]

        result_df.to_csv(os.path.join(cli_args.output_dir, f'test_mlp_cv_{fidx}_results.csv'), sep=',', index=False)

        # save bootstrapped ground truth and predictions
        pickle.dump(bootstrapping_data, open(os.path.join(cli_args.output_dir, f'bootstrapped_gt_and_pred_cv_{fidx}.pkl'), 'wb'))
        pickle.dump(testing_data, open(os.path.join(cli_args.output_dir, f'test_gt_and_pred_cv_{fidx}.pkl'), 'wb'))