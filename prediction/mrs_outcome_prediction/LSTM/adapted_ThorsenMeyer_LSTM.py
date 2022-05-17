#!/usr/bin/python
import os, sys, time, datetime, itertools, shutil, traceback
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Masking
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import argparse

# turn off warnings from Tensorflow
from prediction.mrs_outcome_prediction.LSTM.utils import initiate_log_files
from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from prediction.utils.scoring import precision, matthews, recall
from prediction.utils.utils import generate_balanced_arrays, check_data, ensure_dir, save_json

parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
# parser.add_argument('--date_string', required=True, type=str, help='datestring')
parser.add_argument('--activation', required=True, type=str, help='activation function')
parser.add_argument('--batch', required=True, type=str, help='batch size')
parser.add_argument('--data', required=True, type=str, help='data to use')
parser.add_argument('--dropout', required=True, type=float, help='dropout fraction')
parser.add_argument('--layers', required=True, type=int, help='number of LSTM layers')
parser.add_argument('--masking', required=True, type=bool, help='masking true/false')
parser.add_argument('--optimizer', required=True, type=str, help='optimizer function')
parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. 3M mRS 0-2)')
parser.add_argument('--units', required=True, type=int, help='number of units in each LSTM layer')
args = parser.parse_args()

output_dir = '/Users/jk1/temp/opsum_prediction_output/LSTM_72h'
# define constants
seed = 42
# n_splits = 5
n_splits = 3
# n_epochs = 1000
n_epochs = 1
test_size = 0.20
# checkpoint
save_checkpoint = True

monitor_checkpoint = ['val_matthews', 'max']
# early_stopping
early_stopping = True
monitor_early_stopping = ['val_matthews', 'max']
patience = 200

# load metadata file
# TODO fix paths
# metadata = pd.read_csv('<PATH_TO_METADATA>', sep='\t',
#                        usecols=['id_unique', 'pid', 'admdatetime', 'dead90'],
#                        parse_dates=['admdatetime'])

# load the dataset
# TODO fix paths
features_df_path = '/Users/jk1/temp/opsum_prepro_output/preprocessed_features_14052022_123333.csv'
outcome_df_path = '/Users/jk1/temp/opsum_prepro_output/preprocessed_outcomes_14052022_123333.csv'
outcome = '3M mRS 0-2'
X, y = format_to_2d_table_with_time(feature_df_path=features_df_path, outcome_df_path=outcome_df_path, outcome=outcome)

n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
n_channels = X.sample_label.unique().shape[0]

# dataset = np.load('<PATH_TO_DATASET>')

# # split data in X, y
# X = dataset[:, :, 2:-2]  # [id_unique, rel_time, ..., discharged, dead90]
# y = dataset[:, 0, -1]

# test if data is corrupted
check_data(X)

# # keep last admission per patient
# metadata_latest_adm = metadata.sort_values('admdatetime', ascending=False).drop_duplicates(['pid'])

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

test_X = X[X.patient_id.isin(pid_test)]
test_y = y[y.patient_id.isin(pid_test)]

# TODO check if this is necessary
# find case_admission_id for train/test admissions
# cid = case_admission_id
# train_cidx = y.loc[y.patient_id.isin(pid_train)].case_admission_id.tolist()
# test_cidx = y.loc[y.patient_id.isin(pid_test)].case_admission_id.tolist()

# train_cid_list_of_lists = [metadata.index[metadata['pid'] == x].tolist() for x in pid_train]
# train_idx = [item for sublist in train_idx_list_of_lists for item in sublist]
# test_idx_list_of_lists = [metadata.index[metadata['pid'] == x].tolist() for x in pid_test]
# test_idx = [item for sublist in test_idx_list_of_lists for item in sublist]

# split in TRAIN and TEST
# X_train = X.loc[X.case_admission_id.isin(train_cidx)]
# X_test = X.loc[X.case_admission_id.isin(test_cidx)]
# y_train = y.loc[y.case_admission_id.isin(train_cidx)]
# y_test = y.loc[y.case_admission_id.isin(test_cidx)]
# X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# define 'create_model'
def create_model(activation, batch, data, dropout, layers, masking, optimizer, outcome, units):
    ### MODEL ARCHITECTURE ###
    n_hidden = 1
    # input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    input_layer = Input(shape=(n_time_steps, n_channels))
    if masking:
        # masking layer
        masking_layer = Masking(mask_value=0.)(input_layer)
        if layers == 1:
            # add first LSTM layer
            lstm = LSTM(units, activation=activation, recurrent_dropout=dropout)(masking_layer)
        else:
            # add first LSTM layer
            lstm = LSTM(units, activation=activation, recurrent_dropout=dropout,
                        return_sequences=True)(masking_layer)
    else:
        if layers == 1:
            # add first LSTM layer
            lstm = LSTM(units, activation=activation, recurrent_dropout=dropout)(input_layer)
        else:
            lstm = LSTM(units, activation=activation, recurrent_dropout=dropout,
                        return_sequences=True)(input_layer)
    while n_hidden < layers:
        n_hidden += 1
        if n_hidden == layers:
            # add additional hidden layers
            lstm = LSTM(units, activation=activation, recurrent_dropout=dropout)(lstm)
        else:
            lstm = LSTM(units, activation=activation, recurrent_dropout=dropout,
                        return_sequences=True)(lstm)

    # add output layer
    output_layer = Dense(1, activation='sigmoid')(lstm)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', precision, recall, matthews])

    # define filepath for initial weights
    weights_dir = os.path.join(output_dir, 'best_weights')
    ensure_dir(weights_dir)
    initial_weights_path = os.path.join(weights_dir, '_'.join(['initial_weights', activation,
                                                                     str(batch), data, str(dropout),
                                                                     str(layers), str(masking),
                                                                     optimizer, outcome, str(units)]) + '.hdf5')
    model.save_weights(initial_weights_path)

    #################
    ### RUN MODEL ###
    #################

    i = 0
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        i += 1
        # load the initial weights
        model.load_weights(initial_weights_path)

        # find indexes for train/val admissions
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]

        # fold_pid_train = [pid_train[idx] for idx in fold_pid_train_idx]
        # train_idx_list_of_lists = [metadata.index[metadata['pid'] == x].tolist() for x in
        #                            fold_pid_train]
        # fold_train_idx = [item for sublist in train_idx_list_of_lists for item in sublist]

        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]

        # fold_pid_val = [pid_train[idx] for idx in fold_pid_val_idx]
        # val_idx_list_of_lists = [metadata.index[metadata['pid'] == x].tolist() for x in fold_pid_val]
        # fold_val_idx = [item for sublist in val_idx_list_of_lists for item in sublist]

        # split in TRAIN and VALIDATION sets
        fold_X_train_df = X.loc[X.patient_id.isin(fold_train_pidx)]
        fold_y_train_df = y.loc[y.patient_id.isin(fold_train_pidx)]
        fold_X_val_df = X.loc[X.patient_id.isin(fold_val_pidx)]
        fold_y_val_df = y.loc[y.patient_id.isin(fold_val_pidx)]

        # fold_X_train, fold_X_val, fold_y_train, fold_y_val = X[fold_train_idx], X[fold_val_idx], y[fold_train_idx], y[
        #     fold_val_idx]

        # TODO Remove this check (it is only for debugging purposes and slows down the code)
        # check that case_admission_ids are different in train, test, validation sets
        assert len(set(fold_X_train_df.case_admission_id.unique()).intersection(set(fold_X_val_df.case_admission_id.unique()))) == 0
        assert len(set(fold_X_train_df.case_admission_id.unique()).intersection(set(test_X.case_admission_id.unique()))) == 0

        # Transform dataframes to numpy arrays
        fold_X_train = features_to_numpy(fold_X_train_df, ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
        fold_X_val = features_to_numpy(fold_X_val_df, ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])

        # collect outcomes for all admissions in train and validation sets
        fold_y_train = np.array([fold_y_train_df[fold_y_train_df.case_admission_id == cid].outcome.values[0] for cid in fold_X_train[:, 0, 0, 0]]).astype('float32')
        fold_y_val = np.array([fold_y_val_df[fold_y_val_df.case_admission_id == cid].outcome.values[0] for cid in fold_X_val[:, 0, 0, 0]]).astype('float32')

        # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
        save_json(numpy_to_lookup_table(fold_X_train), os.path.join(output_dir, 'fold_train_lookup_dict_' + str(i) + '.json'))
        save_json(numpy_to_lookup_table(fold_X_val), os.path.join(output_dir, 'fold_val_lookup_dict_' + str(i) + '.json'))

        # Remove the case_admission_id, sample_label, and time_step_label columns from the data
        fold_X_train = fold_X_train[:, :, :, -1].astype('float32')
        fold_X_val = fold_X_val[:, :, :, -1].astype('float32')

        if batch == 'all':
            batch_size = fold_X_train.shape[0]
        else:
            batch_size = int(batch)

        # define checkpoint
        filepath = os.path.join(weights_dir, '_'.join([activation, str(batch),
                                                             data, str(dropout), str(layers),
                                                             str(masking), optimizer, outcome,
                                                             str(units), str(i)]) + '.hdf5')

        checkpoint = ModelCheckpoint(filepath, monitor=monitor_checkpoint[0],
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
            train_hist = model.fit_generator(generate_balanced_arrays(fold_X_train, fold_y_train),
                                             callbacks=callbacks_list,
                                             epochs=n_epochs,
                                             validation_data=[fold_X_val, fold_y_val],
                                             steps_per_epoch=1,
                                             verbose=0)

        elif data == "unchanged":
            train_hist = model.fit(fold_X_train, fold_y_train,
                                   callbacks=callbacks_list,
                                   epochs=n_epochs,
                                   validation_data=[fold_X_val, fold_y_val],
                                   batch_size=batch_size,
                                   verbose=0)

        try:
            # reload best weights
            model.load_weights(filepath)

            # calculate model prediction classes
            y_pred_train = model.predict(fold_X_train)
            y_pred_val = model.predict(fold_X_val)

            y_pred_train_binary = (y_pred_train > 0.5).astype('int32')
            y_pred_val_binary = (y_pred_val > 0.5).astype('int32')


            # append AUC score to existing file
            AUChistory = pd.DataFrame(columns=AUCheader)

            AUChistory = AUChistory.append(
                {'auc_train': roc_auc_score(fold_y_train, y_pred_train),
                 'auc_val': roc_auc_score(fold_y_val, y_pred_val),
                 'matthews_train': matthews_corrcoef(fold_y_train, y_pred_train_binary),
                 'matthews_val': matthews_corrcoef(fold_y_val, y_pred_val_binary),
                 'data': data,
                 'cv_num': i, 'activation': activation, 'dropout': dropout, 'units': units, 'optimizer': optimizer,
                 'batch': batch,
                 'layers': layers,
                 'masking': masking,
                 'outcome': outcome}, ignore_index=True)

            AUChistory.to_csv(os.path.join(output_dir,'AUC_history_gridsearch.tsv'), header=None, index=False, sep='\t', mode='a',
                              columns=AUCheader)

            # append other scores to existing file
            model_history = pd.DataFrame.from_dict(train_hist.history)
            model_history['data'] = data
            model_history['cv_num'] = i
            model_history['activation'] = activation
            model_history['dropout'] = dropout
            model_history['units'] = units
            model_history['optimizer'] = optimizer
            model_history['batch'] = batch
            model_history['layers'] = layers
            model_history['masking'] = masking
            model_history['outcome'] = outcome
            model_history.to_csv(os.path.join(output_dir,'CV_history_gridsearch.tsv'), header=None, index=True, sep='\t', mode='a',
                                 columns=CVheader)

        except:
            var = traceback.format_exc()
            errorDF = pd.DataFrame(columns=errorHeader)
            errorDF = errorDF.append({'error': var,
                                      'args': all_args}, ignore_index=True)
            errorDF.to_csv(os.path.join(output_dir,'error.log'), header=None, index=False,
                           sep='\t', mode='a', columns=errorHeader)


# TODO To be adapted
### RUN MODEL ###
# args = {'activation': 'sigmoid', 'batch': 'all', 'data': 'balanced', 'dropout': 0.0,
#         'layers': 1, 'masking': True, 'optimizer': 'RMSprop',
#         'outcome': '3M mRS 0-2', 'units': 1}


all_args = [args.activation, args.batch, args.data, args.dropout, args.layers, args.masking, args.optimizer,
            args.outcome, args.units]

# all_args = [args['activation'], args['batch'], args['data'], args['dropout'], args['layers'], args['masking'],
#             args['optimizer'], args['outcome'], args['units']]

initiate_log_files(output_dir)
AUCheader = list(pd.read_csv(os.path.join(output_dir, 'AUC_history_gridsearch.tsv'), sep='\t', nrows=1).columns.values)
CVheader = list(pd.read_csv(os.path.join(output_dir, 'CV_history_gridsearch.tsv'), sep='\t', nrows=1).columns.values)
progressHeader = list(pd.read_csv(os.path.join(output_dir, 'progress.log'), sep='\t', nrows=1).columns.values)
errorHeader = list(pd.read_csv(os.path.join(output_dir, 'error.log'), sep='\t', nrows=1).columns.values)

try:
    create_model(activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
                 layers=args.layers, masking=args.masking, optimizer=args.optimizer,
                 outcome=args.outcome, units=args.units)

    progressDF = pd.DataFrame(columns=progressHeader)
    progressDF = progressDF.append({'completed': all_args}, ignore_index=True)
    progressDF.to_csv(os.path.join(output_dir,'progress.log'), header=None, index=False,
                      sep='\t', mode='a', columns=progressHeader)

except:
    var = traceback.format_exc()
    errorDF = pd.DataFrame(columns=errorHeader)
    errorDF = errorDF.append({'error': var,
                              'args': all_args}, ignore_index=True)
    errorDF.to_csv(os.path.join(output_dir,'error.log'), header=None, index=False,
                   sep='\t', mode='a', columns=errorHeader)
