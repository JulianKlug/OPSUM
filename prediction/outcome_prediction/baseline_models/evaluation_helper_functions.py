import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
from sklearn.utils import resample
import numpy as np

from prediction.utils.scoring import plot_roc_curve


def evaluate_method(method_name:str, data_df, ground_truth:str):
    # calculate overall model prediction
    y_pred_test = data_df[f'{method_name}_prob'].values
    y = data_df[ground_truth].values
    roc_auc_figure = plot_roc_curve(y, y_pred_test, title = f"ROC for {method_name})")

    # bootstrap predictions
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
    for i in range(n_iterations):
        selected_cids = resample(data_df.case_admission_id.values, replace=True)
        temp_df = data_df[(~data_df[f'{method_name}_prob'].isnull()) & (data_df.case_admission_id.isin(selected_cids))].copy()

        # ground truth
        y_bs = temp_df[ground_truth].values
        # predictions
        y_pred_bs = temp_df[f'{method_name}_prob'].values
        y_pred_bs_binary = temp_df[f'{method_name} good outcome pred'].values

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
    }], index=[0])

    return result_df, roc_auc_figure, (bootstrapped_ground_truth, bootstrapped_predictions), (y, y_pred_test)

