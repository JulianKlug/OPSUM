import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time
from sklearn.model_selection import train_test_split
from prediction.mrs_outcome_prediction.data_loading.data_formatting import features_to_numpy, \
    link_patient_id_to_outcome, numpy_to_lookup_table
from prediction.utils.scoring import precision, recall, matthews
from prediction.mrs_outcome_prediction.LSTM.LSTM import lstm_generator


DEFAULT_CONFIG = {
    'outcome': '3M mRS 0-2',
    'masking': True,
    'units': 128,
    'activation' : 'sigmoid',
    'dropout' : 0.2,
    'layers' : 2,
    'optimizer' : 'RMSprop',
    'seed' : 42,
    'test_size' : 0.20,
}


def prediction_for_all_timesteps(data, model_weights_path:str, n_time_steps:int, n_channels:int, config:dict=DEFAULT_CONFIG):
    """
    Predicts the outcome for all timesteps for all patients in data.
    Args:
        data: numpy array of shape (n_patients, n_time_steps, n_features)
        model_weights_path: path to the model weights
        n_time_steps: total number of time steps
        n_channels: number of channels
        config: model configuration

    Returns:
        predictions: numpy array of shape (n_time_steps, n_patients)
    """

    subj_pred_over_ts = []

    for ts in tqdm(range(n_time_steps)):
        modified_time_steps = ts + 1

        model = lstm_generator(x_time_shape=modified_time_steps, x_channels_shape=n_channels, masking=config['masking'],
                               n_units=config['units'],
                               activation=config['activation'], dropout=config['dropout'], n_layers=config['layers'])

        model.compile(loss='binary_crossentropy', optimizer=config['optimizer'],
                      metrics=['accuracy', precision, recall, matthews])

        model.load_weights(model_weights_path)

        subj_X_with_first_n_ts = data[:, 0:modified_time_steps, :]

        y_pred = model.predict(subj_X_with_first_n_ts)
        subj_pred_over_ts.append(np.squeeze(y_pred))

    return np.array(subj_pred_over_ts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
    parser.add_argument('--activation', required=True, type=str, help='activation function')
    parser.add_argument('--test_size', required=False, type=float, help='test set size [0-1]', default=0.2)
    parser.add_argument('--seed', required=False, type=int, help='Seed', default=42)
    parser.add_argument('--dropout', required=True, type=float, help='dropout fraction')
    parser.add_argument('--layers', required=True, type=int, help='number of LSTM layers')
    parser.add_argument('--masking', required=True, type=bool, help='masking true/false')
    parser.add_argument('--optimizer', required=True, type=str, help='optimizer function')
    parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. 3M mRS 0-2)')
    parser.add_argument('--units', required=True, type=int, help='number of units in each LSTM layer')
    parser.add_argument('--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('--features_path', required=True, type=str, help='path to features')
    parser.add_argument('--labels_path', required=True, type=str, help='path to labels')
    parser.add_argument('--model_weights_path', required=True, type=str, help='path to model weights')
    args = parser.parse_args()

    config = {
    'outcome': args.outcome,
    'masking': args.masking,
    'units': args.units,
    'activation' : args.activation,
    'dropout' : args.dropout,
    'layers' : args.layers,
    'optimizer' : args.optimizer,
    'seed' : args.seed,
    'test_size' : args.test_size,
    }

    # load the dataset
    X, y = format_to_2d_table_with_time(feature_df_path=args.features_path, outcome_df_path=args.labels_path,
                                        outcome=args.outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]



    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids_with_outcome = link_patient_id_to_outcome(y, args.outcome)
    pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                                    all_pids_with_outcome.outcome.tolist(),
                                                                    stratify=all_pids_with_outcome.outcome.tolist(),
                                                                    test_size=args.test_size,
                                                                    random_state=args.seed)

    test_X_df = X[X.patient_id.isin(pid_test)]
    test_y_df = y[y.patient_id.isin(pid_test)]
    train_X_df = X[X.patient_id.isin(pid_train)]
    train_y_df = y[y.patient_id.isin(pid_train)]

    train_X_np = features_to_numpy(train_X_df,
                                   ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    test_X_np = features_to_numpy(test_X_df,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    train_y_np = np.array([train_y_df[train_y_df.case_admission_id == cid].outcome.values[0] for cid in
                           train_X_np[:, 0, 0, 0]]).astype('float32')
    test_y_np = np.array([test_y_df[test_y_df.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')

    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    test_features_lookup_table = numpy_to_lookup_table(test_X_np)
    train_features_lookup_table = numpy_to_lookup_table(train_X_np)

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')
    train_X_np = train_X_np[:, :, :, -1].astype('float32')

    predictions = prediction_for_all_timesteps(test_X_np, args.model_weights_path, n_time_steps, n_time_steps, config)

    # Save predictions as pickle
    with open(os.path.join(args.output_dir, 'predictions_over_timesteps.pkl'), 'wb') as f:
        pickle.dump(predictions, f)


