import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from prediction.mrs_outcome_prediction.LSTM.LSTM import lstm_generator
from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table
from prediction.utils.scoring import precision, recall, matthews, plot_roc_curve
from prediction.utils.utils import check_data, save_json, ensure_dir


def test_LSTM(X, y, model_weights_path, activation, batch, data, dropout, layers, masking, optimizer, outcome, units,
                n_time_steps, n_channels):
    model = lstm_generator(x_time_shape=n_time_steps, x_channels_shape=n_channels, masking=masking, n_units=units,
                           activation=activation, dropout=dropout, n_layers=layers)

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', precision, recall, matthews])

    model.load_weights(model_weights_path)

    # calculate overall model prediction
    y_pred_test = model.predict(X)
    roc_auc_figure = plot_roc_curve(y, y_pred_test, title = "ROC for LSTM model")

    # bootstrap predictions
    roc_auc_scores = []
    matthews_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    n_iterations = 1000
    for i in range(n_iterations):
        X_bs, y_bs = resample(X, y, replace=True)
        # make predictions
        y_pred_bs = model.predict(X_bs)
        y_pred_bs_binary = (y_pred_bs > 0.5).astype('int32')

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

        # TODO add variables:
        # PPV=positive predictive value. NPV=negative predictive value. LRP=likelihood ratio positive. LRN=likelihood ratio negative.

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
        'data': data,
        'activation': activation, 'dropout': dropout, 'units': units, 'optimizer': optimizer,
        'batch': batch,
        'layers': layers,
        'masking': masking,
        'outcome': outcome,
        'model_weights_path': model_weights_path
    }], index=[0])

    return result_df, roc_auc_figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
    parser.add_argument('--activation', required=True, type=str, help='activation function')
    parser.add_argument('--batch', required=True, type=str, help='batch size')
    parser.add_argument('--data', required=True, type=str, help='data to use')
    parser.add_argument('--dropout', required=True, type=float, help='dropout fraction')
    parser.add_argument('--layers', required=True, type=int, help='number of LSTM layers')
    parser.add_argument('--masking', required=True, type=bool, help='masking true/false')
    parser.add_argument('--optimizer', required=True, type=str, help='optimizer function')
    parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. 3M mRS 0-2)')
    parser.add_argument('--units', required=True, type=int, help='number of units in each LSTM layer')
    parser.add_argument('--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('--features_path', required=True, type=str, help='path to features')
    parser.add_argument('--labels_path', required=True, type=str, help='path to labels')
    parser.add_argument('--cv_fold', required=True, type=int, help='fold of cross-validation')
    parser.add_argument('--model_weights_dir', required=True, type=str, help='path to model weights')
    args = parser.parse_args()


    model_name = '_'.join([args.activation, str(args.batch),
                           args.data, str(args.dropout), str(args.layers),
                           str(args.masking), args.optimizer, args.outcome,
                           str(args.units), str(args.cv_fold)])
    model_weights_path = os.path.join(args.model_weights_dir, f'{model_name}.hdf5')
    output_dir = os.path.join(args.output_dir, f'test_LSTM_{model_name}')
    ensure_dir(output_dir)
    shutil.copy2(model_weights_path, output_dir)

    # define constants
    seed = 42
    test_size = 0.20

    # load the dataset
    outcome = args.outcome
    X, y = format_to_2d_table_with_time(feature_df_path=args.features_path, outcome_df_path=args.labels_path,
                                        outcome=outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    # test if data is corrupted
    check_data(X)

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

    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    save_json(numpy_to_lookup_table(test_X_np),
              os.path.join(output_dir, 'test_lookup_dict.json'))

    # save patient ids used for testing / training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.tsv'),
        sep='\t', index=False)
    pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_test.tsv'),
        sep='\t', index=False)

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')

    result_df, roc_auc_figure = test_LSTM(X=test_X_np, y=test_y_np, model_weights_path=model_weights_path,
                          activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
                          layers=args.layers, masking=args.masking, optimizer=args.optimizer,
                          outcome=args.outcome, units=args.units, n_time_steps=n_time_steps, n_channels=n_channels)

    roc_auc_figure.savefig(os.path.join(output_dir, 'roc_auc_curve.png'))

    result_df.to_csv(os.path.join(output_dir, 'test_LSTM_results.tsv'), sep='\t', index=False)
