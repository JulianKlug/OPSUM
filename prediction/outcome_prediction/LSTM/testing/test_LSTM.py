import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from prediction.outcome_prediction.LSTM.LSTM import lstm_generator
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table
from prediction.outcome_prediction.data_loading.data_loader import load_data, load_external_data
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
    specificity_scores = []
    neg_pred_value_scores = []

    bootstrapped_ground_truth = []
    bootstrapped_predictions = []

    n_iterations = 1000
    for i in tqdm(range(n_iterations)):
        X_bs, y_bs = resample(X, y, replace=True)
        # make predictions
        y_pred_bs = model.predict(X_bs)
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
        'data': data,
        'activation': activation, 'dropout': dropout, 'units': units, 'optimizer': optimizer,
        'batch': batch,
        'layers': layers,
        'masking': masking,
        'outcome': outcome,
        'model_weights_path': model_weights_path
    }], index=[0])

    return result_df, roc_auc_figure, (bootstrapped_ground_truth, bootstrapped_predictions), (y, y_pred_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
    parser.add_argument('-hy', '--hyperparameters_path', type=str, help='hyperparameters selection file', default='')
    parser.add_argument('--activation', type=str, help='activation function')
    parser.add_argument('--batch', type=str, help='batch size')
    parser.add_argument('--data', type=str, help='data to use')
    parser.add_argument('--dropout', type=float, help='dropout fraction')
    parser.add_argument('--layers', type=int, help='number of LSTM layers')
    parser.add_argument('--masking', type=bool, help='masking true/false')
    parser.add_argument('--optimizer', type=str, help='optimizer function')
    parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. 3M mRS 0-2)')
    parser.add_argument('--units', type=int, help='number of units in each LSTM layer')
    parser.add_argument('--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('--features_path', required=True, type=str, help='path to features')
    parser.add_argument('--labels_path', required=True, type=str, help='path to labels')
    parser.add_argument('--cv_fold', type=int, help='fold of cross-validation')
    parser.add_argument('-a', '--all_folds', type=bool, default=False, help='test all folds')
    parser.add_argument('--model_weights_dir', required=True, type=str, help='path to model weights')
    parser.add_argument('-ext', '--is_external_dataset', type=bool, default=False, help='is external dataset')
    args = parser.parse_args()

    if args.hyperparameters_path != '':
        hyperparameters_df = pd.read_csv(args.hyperparameters_path)
        hyperparameters_df = hyperparameters_df[hyperparameters_df['outcome'] == args.outcome]
        args.activation = hyperparameters_df['activation'].values[0]
        args.batch = hyperparameters_df['batch'].values[0]
        args.data = hyperparameters_df['data'].values[0]
        args.dropout = hyperparameters_df['dropout'].values[0]
        args.layers = hyperparameters_df['layers'].values[0]
        args.masking = hyperparameters_df['masking'].values[0]
        args.optimizer = hyperparameters_df['optimizer'].values[0]
        args.units = hyperparameters_df['units'].values[0]
        args.cv_fold = hyperparameters_df['best_fold'].values[0]
    else:
        # verify that all arguments are provided
        assert args.activation is not None, 'activation function or hyperparameters_path not provided'
        assert args.batch is not None, 'batch size or hyperparameters_path not provided'
        assert args.data is not None, 'data to use or hyperparameters_path not provided'
        assert args.dropout is not None, 'dropout fraction or hyperparameters_path not provided'
        assert args.layers is not None, 'number of LSTM layers or hyperparameters_path not provided'
        assert args.masking is not None, 'masking true/false or hyperparameters_path not provided'
        assert args.optimizer is not None, 'optimizer function or hyperparameters_path not provided'
        assert args.units is not None, 'number of units in each LSTM layer or hyperparameters_path not provided'
        assert args.cv_fold is not None, 'fold of cross-validation not provided'

    # define constants
    seed = 42
    test_size = 0.20
    n_splits = 5

    # test only best fold
    if not args.all_folds:
        cv_folds = [args.cv_fold]
    else:
        cv_folds = range(1, n_splits + 1)


    # load the dataset
    outcome = args.outcome

    if not args.is_external_dataset:
        # features_path, labels_path, outcome, test_size, n_splits, seed
        (pid_train, pid_test), (train_X_np, train_y_np), (test_X_np, test_y_np), splits, test_features_lookup_table = load_data(
            args.features_path, args.labels_path, outcome, test_size, n_splits, seed)

        # save patient ids used for testing / training
        pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
            os.path.join(args.output_dir, 'pid_train.tsv'),
            sep='\t', index=False)
        pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
            os.path.join(args.output_dir, 'pid_test.tsv'),
            sep='\t', index=False)

    else:
        test_X_np, test_y_np, test_features_lookup_table = load_external_data(
            args.features_path, args.labels_path, outcome)

    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    save_json(test_features_lookup_table,
              os.path.join(args.output_dir, 'test_lookup_dict.json'))

    n_time_steps = test_X_np.shape[1]
    n_channels = test_X_np.shape[-1]

    for cv_fold in cv_folds:
        model_name = '_'.join([args.activation, str(args.batch),
                               args.data, str(args.dropout), str(args.layers),
                               str(args.masking), args.optimizer, args.outcome,
                               str(args.units), str(cv_fold)])
        model_weights_path = os.path.join(args.model_weights_dir, f'{model_name}.hdf5')
        output_dir = os.path.join(args.output_dir, f'test_LSTM_{model_name}')
        ensure_dir(output_dir)
        shutil.copy2(model_weights_path, output_dir)

        result_df, roc_auc_figure, bootstrapping_data, testing_data = test_LSTM(X=test_X_np, y=test_y_np, model_weights_path=model_weights_path,
                              activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
                              layers=args.layers, masking=args.masking, optimizer=args.optimizer,
                              outcome=args.outcome, units=args.units, n_time_steps=n_time_steps, n_channels=n_channels)

        roc_auc_figure.savefig(os.path.join(output_dir, f'roc_auc_curve_fold_{cv_fold}.png'))

        result_df.to_csv(os.path.join(output_dir, f'test_LSTM_results_fold_{cv_fold}.tsv'), sep='\t', index=False)

        # save bootstrapped ground truth and predictions
        pickle.dump(bootstrapping_data, open(os.path.join(output_dir, f'bootstrapped_gt_and_pred_fold_{cv_fold}.pkl'), 'wb'))
        pickle.dump(testing_data, open(os.path.join(output_dir, f'test_gt_and_pred_fold_{cv_fold}.pkl'), 'wb'))
