from sklearn.preprocessing import StandardScaler

from prediction.outcome_prediction.Transformer.utils.utils import prepare_dataset, DictLogger
from torch.utils.data import DataLoader, TensorDataset
import torch as ch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
from sklearn.utils import resample

def test_transformer_model(trained_model, X_train, y_train, X_test, y_test, outcome, model_config, model_weights_path,
                           rescale=True, use_gpu=True):
    if use_gpu:
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'
    logger = DictLogger(0)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1000,
                         gradient_clip_val=model_config['grad_clip_value'], logger=logger)


    # calculate overall model predictions
    train_dataset, test_dataset = prepare_dataset((X_train, X_test, y_train, y_test),
                                                  balanced=model_config['balanced'],
                                                  rescale=rescale,
                                                  use_gpu=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=1024)
    y_pred_test = ch.sigmoid(trainer.predict(trained_model, test_loader)[0])[:, -1]

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

    # n_iterations = 1000
    n_iterations = 2
    for i in tqdm(range(n_iterations)):
        X_bs, y_bs = resample(X_test, y_test, replace=True)

        _, bs_test_dataset = prepare_dataset((X_train, X_bs, y_train, y_bs),
                                                        balanced=model_config['balanced'],
                                                        rescale=rescale,
                                                        use_gpu=use_gpu)
        bs_test_loader = DataLoader(bs_test_dataset, batch_size=1024)

        # make predictions
        y_pred_bs = ch.sigmoid(trainer.predict(trained_model, bs_test_loader)[0])[:, -1]
        y_pred_bs_binary = (y_pred_bs.numpy() > 0.5).astype('int32')

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
        'outcome': outcome,
        'model_weights_path': model_weights_path
    }], index=[0])

    result_df = result_df.append(pd.DataFrame(model_config, index=[0]))

    return result_df, (bootstrapped_ground_truth, bootstrapped_predictions), (y_test, y_pred_test)
