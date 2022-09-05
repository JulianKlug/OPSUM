import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics




def plot_roc_curve(fpr, tpr, name:str):
    sns.lineplot(x=fpr, y=tpr, color='orange', label=name)
    sns.lineplot(x=[0, 1], y=[0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve of {name}')
    plt.legend()
    plt.show()


def evaluate_method(method_name:str, data_df, ground_truth:str):
    result_columns = ['ground truth', 'method', 'auc', 'accuracy', 'f1', 'precision', 'recall',
                      'fpr', 'tpr', 'roc_thresholds']
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    auc = tf.keras.metrics.AUC()
    accuracy = tf.keras.metrics.Accuracy()

    temp_df = data_df[~data_df[f'{method_name}_prob'].isnull()].copy()
    # keras auc  y_true, y_pred
    method_auc = auc(temp_df[ground_truth], temp_df[f'{method_name}_prob']).numpy()
    # keras accuracy y_true, y_pred
    method_acc = accuracy(temp_df[ground_truth], temp_df[f'{method_name} good outcome pred']).numpy()
    # sklearn f1 score y_true, y_pred
    method_f1 = metrics.f1_score(temp_df[ground_truth], temp_df[f'{method_name} good outcome pred'])
    # keras precision y_true, y_pred
    method_precision = precision(temp_df[ground_truth], temp_df[f'{method_name} good outcome pred']).numpy()
    # keras recall y_true, y_pred
    method_recall = recall(temp_df[ground_truth], temp_df[f'{method_name} good outcome pred']).numpy()
    # sklearn roc curve y_true, y_score
    method_fpr, method_tpr, method_thresholds = metrics.roc_curve(
                            temp_df[ground_truth],
                            temp_df[f'{method_name}_prob']
                                                                  )

    method_df = pd.DataFrame(
        [[ground_truth, method_name, method_auc, method_acc, method_f1, method_precision, method_recall,
         method_fpr, method_tpr, method_thresholds]],
        columns=result_columns)

    plot_roc_curve(method_fpr, method_tpr, method_name)

    return method_df

