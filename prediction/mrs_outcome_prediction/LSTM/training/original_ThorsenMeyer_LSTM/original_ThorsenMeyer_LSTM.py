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

from prediction.utils.scoring import precision, matthews, recall
from prediction.utils.utils import generate_balanced_arrays

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--date_string', required=True, type=str, help='datestring' )
parser.add_argument('--activation', required=True, type=str, help='activation function' )
parser.add_argument('--batch', required=True, type=str, help='batch size' )
parser.add_argument('--data', required=True, type=str, help='data to use' )
parser.add_argument('--dropout', required=True, type=float, help='dropout fraction' )
parser.add_argument('--layers', required=True, type=int, help='number of LSTM layers' )
parser.add_argument('--masking', required=True, type=bool, help='masking true/false' )
parser.add_argument('--optimizer', required=True, type=str, help='optimizer function' )
parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. dead90)' )
parser.add_argument('--units', required=True, type=int, help='number of units in each LSTM layer' )
args = parser.parse_args()


from make_moab_jobs import param_dict as param_dict
output_dir = '<PATH_FOR_OUTPUT>'
# define constants
seed = 42
n_splits = 5
n_epochs = 1000
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
metadata = pd.read_csv('<PATH_TO_METADATA>', sep='\t',
 usecols = ['id_unique', 'ssn', 'admdatetime', 'dead90'],
 parse_dates = ['admdatetime'])
# turn off warnings from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# load the dataset
# TODO fix paths
dataset = np.load('<PATH_TO_DATASET>')

# split data in X, y
X = dataset[:, :, 2:-2] # [id_unique, rel_time, ..., discharged, dead90]
y = dataset[:, 0, -1]
# test if data is corrupted
if np.isnan(X).any() or np.isinf(X).any():
 sys.exit('Data is corrupted!')
# keep last admission per patient
metadata_latest_adm = metadata.sort_values('admdatetime', ascending=False).drop_duplicates(['ssn'])

# split 'ssn' in TRAIN and TEST
# ssn = encrypted social security number
ssn_train, ssn_test, y_ssn_train, y_ssn_test = train_test_split(metadata_latest_adm.ssn.tolist(),
 metadata_latest_adm.dead90.tolist(),
 stratify = metadata_latest_adm.dead90.tolist(),
 test_size=test_size,
 random_state=seed)

# find indexes for train/test admissions
train_idx_list_of_lists = [metadata.index[metadata['ssn'] == x].tolist() for x in ssn_train]
train_idx = [item for sublist in train_idx_list_of_lists for item in sublist]
test_idx_list_of_lists = [metadata.index[metadata['ssn'] == x].tolist() for x in ssn_test]
test_idx = [item for sublist in test_idx_list_of_lists for item in sublist]

# split in TRAIN and TEST
X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

# define 'create_model'
def create_model(activation, batch, data, dropout, layers, masking, optimizer, outcome, units):

    ### MODEL ARCHITECTURE ###
    n_hidden = 1
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
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
    initial_weights_path = output_dir + "/best_weights/" + '_'.join(['initial_weights', activation,
                                                                     str(batch), data, str(dropout),
                                                                     str(layers), str(masking),
                                                                     optimizer, outcome, str(units)]) + '.hdf5'
    model.save_weights(initial_weights_path)

    #################
    ### RUN MODEL ###
    #################

    i = 0
    for this_ssn_train_idx, this_ssn_val_idx in kfold.split(ssn_train, y_ssn_train):
        i += 1
        # load the initial weights
        model.load_weights(initial_weights_path)

        # find indexes for train/val admissions
        this_ssn_train = [ssn_train[idx] for idx in this_ssn_train_idx]
        train_idx_list_of_lists = [metadata.index[metadata['ssn'] == x].tolist() for x in
                                   this_ssn_train]
        this_train_idx = [item for sublist in train_idx_list_of_lists for item in sublist]

        this_ssn_val = [ssn_train[idx] for idx in this_ssn_val_idx]
        val_idx_list_of_lists = [metadata.index[metadata['ssn'] == x].tolist() for x in this_ssn_val]
        this_val_idx = [item for sublist in val_idx_list_of_lists for item in sublist]

        # split in TRAIN and VALIDATION sets
        this_X_train, this_X_val, this_y_train, this_y_val = X[this_train_idx], X[this_val_idx], y[this_train_idx], y[this_val_idx]

        if batch == 'all':
            batch_size = this_X_train.shape[0]
        else:
            batch_size = int(batch)

        # define checkpoint
        filepath = output_dir + "/best_weights/" + '_'.join([activation, str(batch),
                                                             data, str(dropout), str(layers),
                                                             str(masking), optimizer, outcome,
                                                             str(units), str(i)]) + '.hdf5'

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
            train_hist = model.fit_generator(generate_balanced_arrays(this_X_train, this_y_train),
                                             callbacks=callbacks_list,
                                             epochs=n_epochs,
                                             validation_data=[this_X_val, this_y_val],
                                             steps_per_epoch=1,
                                             verbose=0)

        elif data == "unchanged":
            train_hist = model.fit(this_X_train, this_y_train,
                                   callbacks=callbacks_list,
                                   epochs=n_epochs,
                                   validation_data=[this_X_val, this_y_val],
                                   batch_size=batch_size,
                                   verbose=0)

        try:
            # reload best weights
            model.load_weights(filepath)

            # calculate model prediction classes
            y_pred_train = model.predict(this_X_train)
            y_pred_val = model.predict(this_X_val)

            y_pred_train_binary = (y_pred_train > 0.5).astype('int32')
            y_pred_val_binary = (y_pred_val > 0.5).astype('int32')

            # append AUC score to existing file
            AUChistory = pd.DataFrame(columns=AUCheader)

            AUChistory = AUChistory.append(
                {'auc_train': roc_auc_score(this_y_train, y_pred_train), 'auc_val': roc_auc_score(this_y_val, y_pred_val),
                 'matthews_train': matthews_corrcoef(this_y_train, y_pred_train_binary),
                 'matthews_val': matthews_corrcoef(this_y_val, y_pred_val_binary),
                 'data': data,
                 'cv_num': i, 'activation': activation, 'dropout': dropout, 'units': units, 'optimizer': optimizer,
                 'batch': batch,
                 'layers': layers,
                 'masking': masking,
                 'outcome': outcome}, ignore_index=True)

            AUChistory.to_csv('AUC_history_gridsearch.tsv', header=None, index=False, sep='\t', mode='a', columns=AUCheader)

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
            model_history.to_csv('CV_history_gridsearch.tsv', header=None, index=True, sep='\t', mode='a', columns=CVheader)

        except:
            var = traceback.format_exc()
            errorDF = pd.DataFrame(columns=errorHeader)
            errorDF = errorDF.append({'error': var,
                                      'args': all_args}, ignore_index=True)
            errorDF.to_csv('error.log', header=None, index=False,
                           sep='\t', mode='a', columns=errorHeader)


### RUN MODEL ###
os.chdir(output_dir)

os.chdir(output_dir)
all_args = [args.activation, args.batch, args.data, args.dropout, args.layers, args.masking, args.optimizer, args.outcome, args.units]
AUCheader = list(pd.read_csv('AUC_history_gridsearch.tsv', sep='\t', nrows=1).columns.values)
CVheader = list(pd.read_csv('CV_history_gridsearch.tsv', sep='\t', nrows=1).columns.values)
progressHeader = list(pd.read_csv('progress.log', sep='\t', nrows=1).columns.values)
errorHeader = list(pd.read_csv('error.log', sep='\t', nrows=1).columns.values)

try:
    create_model(activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
    layers=args.layers, masking=args.masking, optimizer=args.optimizer,
    outcome=args.outcome, units=args.units)
    progressDF = pd.DataFrame(columns=progressHeader)
    progressDF = progressDF.append({'completed' : all_args}, ignore_index = True)
    progressDF.to_csv('progress.log', header=None, index=False,
    sep='\t', mode='a', columns = progressHeader)

except:
    var = traceback.format_exc()
    errorDF = pd.DataFrame(columns=errorHeader)
    errorDF = errorDF.append({'error': var,
    'args' : all_args}, ignore_index = True)
    errorDF.to_csv('error.log', header=None, index=False,
    sep='\t', mode='a', columns = errorHeader)










