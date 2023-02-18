#!/usr/bin/python
import json
import os
import traceback
import time
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import argparse

from tensorflow import keras

from prediction.outcome_prediction.transformer.transformer import transformer_generator
from prediction.utils.training_utils import WarmUpScheduler, initiate_log_files

# turn off warnings from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table, feature_order_verification
from prediction.utils.scoring import precision, matthews, recall
from prediction.utils.utils import generate_balanced_arrays, check_data, ensure_dir, save_json


def train_model(
        features_path: str, labels_path:str, output_dir:str, outcome:str,
        batch, data,
        atn_head_size, n_atn_heads, atn_feed_forward_dim, n_transformer_blocks, transformer_dropout,
        mlp_units, mlp_dropout,
        optimizer, learning_rate, weight_decay,
        test_size, seed, n_splits, n_epochs,
        save_checkpoint, monitor_checkpoint, early_stopping, monitor_early_stopping, patience_early_stopping,
        warmup, warmup_epochs,
        CVheader, errorHeader,
        verbose, use_tensorboard
):
    """
    Train a Transformer model on the given data.
    The model is trained using k-fold cross-validation.

    Model checkpoints and performance metrics are saved to disk.

    Arguments:
        features_path {str} -- Path to the features file.
        labels_path {str} -- Path to the labels file.
        output_dir {str} -- Path to the output directory.
        outcome {str} -- outcome to predict


        batch {int} -- batch size
        data {str} -- "balanced" or "unchanged", depending on whether the data should be balanced for training or not

        atn_head_size {int} -- size of the attention heads
        n_atn_heads {int} -- number of attention heads
        atn_feed_forward_dim {int} -- dimension of the feed-forward layer
        n_transformer_blocks {int} -- number of transformer blocks
        transformer_dropout {float} -- dropout rate for the transformer blocks
        mlp_units {list(int)} -- list of units for the MLP
        mlp_dropout {float} -- dropout rate for the MLP
        optimizer {str} -- optimizer to use
        learning_rate {float} -- learning rate
        weight_decay {float} -- weight decay

        test_size {float} -- proportion of data to use for testing
        seed {int} -- random seed
        n_splits {int} -- number of folds for cross-validation
        n_epochs {int} -- number of epochs

        save_checkpoint {bool} -- whether to save model checkpoints
        monitor_checkpoint {list(str)} -- metrics to monitor for model checkpoints
        early_stopping {bool} -- whether to use early stopping
        monitor_early_stopping {list(str)} -- metrics to monitor for early stopping
        patience_early_stopping {int} -- number of epochs to wait before early stopping
        warmup {bool} -- whether to use warmup
        warmup_steps {int} -- number of warmup steps (during warmup, the learning rate is linearly increased from 0 to the final learning rate)

        CVheader {str} -- header for the cross-validation log file
        errorHeader {str} -- header for the error log file

        verbose {int} -- verbosity level

    Returns: void
    """
    saved_args = locals().copy()
    saved_args.pop('CVheader')
    saved_args.pop('errorHeader')

    # save training parameters
    training_params_dir = os.path.join(output_dir, 'training_parameters')
    ensure_dir(training_params_dir)
    model_name = 'transformer_' + '_'.join([outcome, str(batch), str(data),
                                            str(atn_head_size), str(n_atn_heads), str(atn_feed_forward_dim),
                                            str(n_transformer_blocks), str(transformer_dropout),
                                            str(mlp_units), str(mlp_dropout),
                                            optimizer, str(learning_rate)])
    training_params_filename = f'params_{model_name}.json'
    with open(os.path.join(training_params_dir, training_params_filename), 'w') as fp:
        json.dump(saved_args, fp, indent=4)

    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
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

    test_X = X[X.patient_id.isin(pid_test)]
    # Here test data is not needed anymore, but for reference should be loaded as such: test_y = y[y.patient_id.isin(pid_test)]

    # define K fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ### DEFINE MODEL
    model = transformer_generator(
                input_shape=(n_time_steps, n_channels),
                n_output_dimensions=1,
                head_size=atn_head_size,
                num_heads=n_atn_heads,
                feed_forward_dim=atn_feed_forward_dim,
                num_transformer_blocks=n_transformer_blocks,
                mlp_units=mlp_units,
                transformer_dropout=transformer_dropout,
                mlp_dropout=mlp_dropout,
            )

    if optimizer == 'Adam':
        optimizer_function = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {optimizer} not implemented.')

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer_function,
        metrics=['accuracy', precision, recall, matthews, "sparse_categorical_accuracy"],
    )

    if verbose:
        model.summary()

    # define filepath for initial weights
    weights_dir = os.path.join(output_dir, 'best_weights')
    ensure_dir(weights_dir)
    initial_weights_path = os.path.join(weights_dir, f'initial_weights_{model_name}.hdf5')
    model.save_weights(initial_weights_path)

    # define header for AUC file
    AUCheader = list(
        pd.read_csv(os.path.join(output_dir, 'AUC_history_gridsearch.tsv'), sep='\t', nrows=1).columns.values)

    ### TRAIN MODEL USING K-FOLD CROSS-VALIDATION
    i = 0
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        i += 1
        # load the initial weights
        model.load_weights(initial_weights_path)

        # find indexes for train/val admissions
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]

        # split in TRAIN and VALIDATION sets
        fold_X_train_df = X.loc[X.patient_id.isin(fold_train_pidx)]
        fold_y_train_df = y.loc[y.patient_id.isin(fold_train_pidx)]
        fold_X_val_df = X.loc[X.patient_id.isin(fold_val_pidx)]
        fold_y_val_df = y.loc[y.patient_id.isin(fold_val_pidx)]

        # check that case_admission_ids are different in train, test, validation sets
        # Note: this is fairly slow
        assert len(set(fold_X_train_df.case_admission_id.unique()).intersection(set(fold_X_val_df.case_admission_id.unique()))) == 0
        assert len(set(fold_X_train_df.case_admission_id.unique()).intersection(set(test_X.case_admission_id.unique()))) == 0

        # Transform dataframes to numpy arrays
        fold_X_train = features_to_numpy(fold_X_train_df, ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
        fold_X_val = features_to_numpy(fold_X_val_df, ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])

        # ensure that the order of features (3rd dimension) is the one predefined for the model
        feature_order_verification(fold_X_train)

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
        filepath = os.path.join(weights_dir, f'{model_name}_{str(i)}.hdf5')

        checkpoint = ModelCheckpoint(filepath, monitor=monitor_checkpoint[0],
                                     verbose=0, save_best_only=True,
                                     mode=monitor_checkpoint[1])

        # define early stopping
        earlystopping = EarlyStopping(monitor=monitor_early_stopping[0], min_delta=0, patience=patience_early_stopping,
                                      verbose=0, mode=monitor_early_stopping[1], start_from_epoch=warmup_epochs)

        warmupscheduler = WarmUpScheduler(
            final_lr=learning_rate,
            warmup_steps=warmup_epochs,
            verbose=verbose
        )

        # define callbacks_list
        callbacks_list = []
        if early_stopping:
            callbacks_list.append(earlystopping)
        if save_checkpoint:
            callbacks_list.append(checkpoint)
        if warmup:
            callbacks_list.append(warmupscheduler)
        if use_tensorboard:
            tensorboard_dir = os.path.join(output_dir, 'tensorboard_logs', f'{model_name}_{str(i)}')
            ensure_dir(tensorboard_dir)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            callbacks_list.append(tensorboard_callback)

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
                    'outcome': outcome,
                    'batch': batch,
                    'data': data,
                    'atn_head_size': atn_head_size,
                    'n_atn_heads': n_atn_heads,
                    'atn_feed_forward_dim': atn_feed_forward_dim,
                    'n_transformer_blocks': n_transformer_blocks,
                    'transformer_dropout': transformer_dropout,
                    'mlp_units': mlp_units,
                    'mlp_dropout': mlp_dropout,
                    'optimizer': optimizer,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'warmup_epochs': warmup_epochs,
                    'cv_num': i
                }, ignore_index=True)
            AUChistory.to_csv(os.path.join(output_dir,'AUC_history_gridsearch.tsv'), header=None, index=False, sep='\t', mode='a',
                              columns=AUCheader)

            # append other scores to existing file
            model_history = pd.DataFrame.from_dict(train_hist.history)
            model_history['epoch'] = train_hist.epoch
            model_history['data'] = data
            model_history['cv_num'] = i
            model_history['batch'] = batch
            model_history['outcome'] = outcome
            model_history['max epochs'] = n_epochs
            model_history['optimizer'] = optimizer
            model_history['learning_rate'] = learning_rate
            model_history['warmup_epochs'] = warmup_epochs
            model_history['atn_head_size'] = atn_head_size
            model_history['n_atn_heads'] = n_atn_heads
            model_history['atn_feed_forward_dim'] = atn_feed_forward_dim
            model_history['n_transformer_blocks'] = n_transformer_blocks
            model_history['transformer_dropout'] = transformer_dropout
            model_history['mlp_units'] = str(mlp_units)
            model_history['mlp_dropout'] = mlp_dropout

            model_history.to_csv(os.path.join(output_dir,'CV_history_gridsearch.tsv'), header=None, index=False, sep='\t', mode='a',
                                 columns=CVheader)

        except Exception:
            var = traceback.format_exc()
            errorDF = pd.DataFrame(columns=errorHeader)
            errorDF = errorDF.append({'error': var,
                                      'args': saved_args.values()}, ignore_index=True)
            errorDF.to_csv(os.path.join(output_dir,'error.log'), header=None, index=False,
                           sep='\t', mode='a', columns=errorHeader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
    parser.add_argument('--date_string', required=True, type=str, help='datestring')
    parser.add_argument('--batch', required=True, type=str, help='batch size')
    parser.add_argument('--data', required=True, type=str, help='data to use')

    parser.add_argument('--atn_head_size', required=True, type=int, help='attention head size')
    parser.add_argument('--n_atn_heads', required=True, type=int, help='number of attention heads')
    parser.add_argument('--atn_feed_forward_dim', required=True, type=int, help='attention feed forward dimension')
    parser.add_argument('--n_transformer_blocks', required=True, type=int, help='number of transformer blocks')
    parser.add_argument('--transformer_dropout', required=True, type=float, help='transformer dropout')
    parser.add_argument('--mlp_units', required=True, action='append', type=int, help='mlp units')
    parser.add_argument('--mlp_dropout', required=True, type=float, help='mlp dropout')
    parser.add_argument('--optimizer', required=True, type=str, help='optimizer')
    parser.add_argument('--learning_rate', required=True, type=float, help='learning rate')
    parser.add_argument('--weight_decay', required=True, type=float, help='weight decay')

    parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. 3M mRS 0-2)')
    parser.add_argument('--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('--features_path', required=True, type=str, help='path to features')
    parser.add_argument('--labels_path', required=True, type=str, help='path to labels')

    parser.add_argument('--verbose', '-v', action='store_true', help='verbose', default=False)
    parser.add_argument('--use_tensorboard', '-tb', action='store_true', help='log onto tensorboard', default=False)
    args = parser.parse_args()

    start_time = time.time()

    # define constants
    output_dir = args.output_dir
    seed = 42
    n_splits = 5
    n_epochs = 3000
    test_size = 0.20
    # checkpoint
    save_checkpoint = True
    monitor_checkpoint = ['val_matthews', 'max']
    # early_stopping
    early_stopping = True
    monitor_early_stopping = ['val_matthews', 'max']
    patience = 400
    # warmup
    warmup = True
    warmup_epochs = 200
    if warmup:
        n_epochs = n_epochs + warmup_epochs

    ### RUN MODEL ###
    all_args = [args.batch, args.data, args.atn_head_size, args.n_atn_heads, args.atn_feed_forward_dim,
                args.n_transformer_blocks, args.transformer_dropout, args.mlp_units, args.mlp_dropout, args.optimizer,
                args.learning_rate, args.outcome]
    initiate_log_files(output_dir, param_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'parameter_space.json'))

    CVheader = list(
        pd.read_csv(os.path.join(output_dir, 'CV_history_gridsearch.tsv'), sep='\t', nrows=1).columns.values)
    progressHeader = list(pd.read_csv(os.path.join(output_dir, 'progress.log'), sep='\t', nrows=1).columns.values)
    errorHeader = list(pd.read_csv(os.path.join(output_dir, 'error.log'), sep='\t', nrows=1).columns.values)

    try:
        train_model(
            features_path=args.features_path, labels_path=args.labels_path, output_dir=output_dir, outcome=args.outcome,
            batch=args.batch, data=args.data,

            atn_head_size=args.atn_head_size, n_atn_heads=args.n_atn_heads, atn_feed_forward_dim=args.atn_feed_forward_dim,
            n_transformer_blocks=args.n_transformer_blocks, transformer_dropout=args.transformer_dropout,
            mlp_units=args.mlp_units, mlp_dropout=args.mlp_dropout,
            optimizer=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            test_size=test_size, seed=seed, n_splits=n_splits, n_epochs=n_epochs,
            save_checkpoint=save_checkpoint, monitor_checkpoint=monitor_checkpoint,
            early_stopping=early_stopping, monitor_early_stopping=monitor_early_stopping, patience_early_stopping=patience,
            warmup=warmup, warmup_epochs=warmup_epochs,
            CVheader=CVheader, errorHeader=errorHeader,
            verbose=args.verbose, use_tensorboard=args.use_tensorboard
        )

        progressDF = pd.DataFrame(columns=progressHeader)
        end_time = time.time()
        progressDF = progressDF.append({'completed': all_args, 'time_elapsed': (end_time - start_time) / 60 }, ignore_index=True)
        progressDF.to_csv(os.path.join(output_dir, 'progress.log'), header=None, index=False,
                          sep='\t', mode='a', columns=progressHeader)
        print('TRAINING COMPLETE')

    except Exception:
        var = traceback.format_exc()
        errorDF = pd.DataFrame(columns=errorHeader)
        errorDF = errorDF.append({'error': var,
                                  'args': all_args}, ignore_index=True)
        errorDF.to_csv(os.path.join(output_dir, 'error.log'), header=None, index=False,
                       sep='\t', mode='a', columns=errorHeader)
        print('ERROR WHILE TRAINING')
        print(f'Please see {os.path.join(output_dir, "error.log")} for details')

