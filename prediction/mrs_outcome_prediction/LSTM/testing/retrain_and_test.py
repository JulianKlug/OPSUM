import argparse
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from prediction.mrs_outcome_prediction.LSTM.LSTM import lstm_generator
from prediction.mrs_outcome_prediction.LSTM.testing.test_LSTM import test_LSTM
from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table
from prediction.utils.scoring import precision, recall, matthews
from prediction.utils.utils import ensure_dir, check_data, save_json, generate_balanced_arrays


def retrain_LSTM(X, y, model_weights_path, activation, batch, data, dropout, layers, masking, optimizer, outcome, units,
                 n_time_steps, n_channels):
    # checkpoint
    save_checkpoint = True
    monitor_checkpoint = ['matthews', 'max']
    # early_stopping
    early_stopping = True
    monitor_early_stopping = ['matthews', 'max']
    patience = 200

    model = lstm_generator(x_time_shape=n_time_steps, x_channels_shape=n_channels, masking=masking, n_units=units,
                           activation=activation, dropout=dropout, n_layers=layers)

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', precision, recall, matthews])

    if batch == 'all':
        batch_size = X.shape[0]
    else:
        batch_size = int(batch)

    checkpoint = ModelCheckpoint(model_weights_path, monitor=monitor_checkpoint[0],
                                 verbose=0, save_best_only=True,
                                 mode=monitor_checkpoint[1])

    # define early stopping
    earlystopping = EarlyStopping(monitor=monitor_early_stopping[0], min_delta=0, patience=patience,
                                  verbose=0, mode=monitor_early_stopping[1])

    # define callbacks_list
    callbacks_list = []
    if early_stopping:
        callbacks_list.append(earlystopping)
    if save_checkpoint:
        callbacks_list.append(checkpoint)

    # TRAIN MODEL
    if data == "balanced":
        train_hist = model.fit_generator(generate_balanced_arrays(X, y),
                                         callbacks=callbacks_list,
                                         epochs=n_epochs,
                                         steps_per_epoch=1,
                                         verbose=0)

    elif data == "unchanged":
        train_hist = model.fit(X, y,
                               callbacks=callbacks_list,
                               epochs=n_epochs,
                               batch_size=batch_size,
                               verbose=0)

    # reload best weights
    model.load_weights(model_weights_path)

    # calculate model prediction classes
    y_pred_train = model.predict(X)

    # save train history
    model_history = pd.DataFrame.from_dict(train_hist.history)
    model_history['final_train_auc'] = roc_auc_score(y, y_pred_train)
    model_history['data'] = data
    model_history['activation'] = activation
    model_history['dropout'] = dropout
    model_history['units'] = units
    model_history['optimizer'] = optimizer
    model_history['batch'] = batch
    model_history['layers'] = layers
    model_history['masking'] = masking
    model_history['outcome'] = outcome
    model_history['epoch'] = n_epochs
    model_history.to_csv(os.path.join(os.path.dirname(model_weights_path), 'train_history.tsv'), index=False, sep='\t')

    return model




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
    args = parser.parse_args()

    model_name = '_'.join([args.activation, str(args.batch),
                           args.data, str(args.dropout), str(args.layers),
                           str(args.masking), args.optimizer, args.outcome,
                           str(args.units)])
    model_name = model_name.replace(' ', '-')
    output_dir = os.path.join(args.output_dir, f'retrain_and_test_LSTM_{model_name}')
    model_weights_path = os.path.join(output_dir, model_name + '.hdf5')
    ensure_dir(output_dir)

    # define constants
    seed = 42
    n_epochs = 1000
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
    save_json(numpy_to_lookup_table(test_X_np),
              os.path.join(output_dir, 'test_lookup_dict.json'))
    save_json(numpy_to_lookup_table(train_X_np),
                os.path.join(output_dir, 'train_lookup_dict.json'))

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')
    train_X_np = train_X_np[:, :, :, -1].astype('float32')

    retrain_LSTM(X=train_X_np, y=train_y_np, model_weights_path=model_weights_path,
              activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
              layers=args.layers, masking=args.masking, optimizer=args.optimizer,
              outcome=args.outcome, units=args.units, n_time_steps=n_time_steps, n_channels=n_channels)

    # test the model
    result_df = test_LSTM(X=test_X_np, y=test_y_np, model_weights_path=model_weights_path,
              activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
              layers=args.layers, masking=args.masking, optimizer=args.optimizer,
              outcome=args.outcome, units=args.units, n_time_steps=n_time_steps, n_channels=n_channels)

    result_df.to_csv(os.path.join(output_dir, 'test_LSTM_results.tsv'), sep='\t', index=False)
