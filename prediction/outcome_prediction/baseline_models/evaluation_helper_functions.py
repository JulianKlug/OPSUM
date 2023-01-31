import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
from sklearn.utils import resample
import numpy as np

def plot_roc_curve(fpr, tpr, name:str):
    sns.lineplot(x=fpr, y=tpr, color='orange', label=name)
    sns.lineplot(x=[0, 1], y=[0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve of {name}')
    plt.legend()
    plt.show()


def evaluate_method(method_name:str, data_df, ground_truth:str):
    # bootstrap predictions
    roc_auc_scores = []
    matthews_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    n_iterations = 1000
    for i in range(n_iterations):
        selected_cids = resample(data_df.case_admission_id.values, replace=True)
        temp_df = data_df[(~data_df[f'{method_name}_prob'].isnull()) & (data_df.case_admission_id.isin(selected_cids))].copy()

        # ground truth
        y_bs = temp_df[ground_truth].values
        # predictions
        y_pred_bs = temp_df[f'{method_name}_prob'].values
        y_pred_bs_binary = temp_df[f'{method_name} good outcome pred'].values

        # evaluate model
        roc_auc_bs = roc_auc_score(y_bs, y_pred_bs)
        roc_auc_scores.append(roc_auc_bs)
        matthews_bs = matthews_corrcoef(y_bs, y_pred_bs_binary)
        matthews_scores.append(matthews_bs)
        accuracy_bs = accuracy_score(y_bs, y_pred_bs_binary)
        accuracy_scores.append(accuracy_bs)
        precision_bs = precision_score(y_bs, y_pred_bs_binary)
        recall_bs = recall_score(y_bs, y_pred_bs_binary)
        precision_scores.append(precision_bs)
        recall_scores.append(recall_bs)

    # get medians
    median_roc_auc = np.percentile(roc_auc_scores, 50)
    median_matthews = np.percentile(matthews_scores, 50)
    median_accuracy = np.percentile(accuracy_scores, 50)
    median_precision = np.percentile(precision_scores, 50)
    median_recall = np.percentile(recall_scores, 50)

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

    result_df = pd.DataFrame([{
        'ground_truth': ground_truth,
        'method_name': method_name,
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
    }], index=[0])

    # sklearn roc curve y_true, y_score
    method_fpr, method_tpr, method_thresholds = metrics.roc_curve(
                            temp_df[ground_truth],
                            temp_df[f'{method_name}_prob']
                                                                  )

    plot_roc_curve(method_fpr, method_tpr, method_name)

    return result_df

