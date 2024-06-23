import argparse
import os
import shutil
from tqdm import tqdm
import pandas as pd
import pickle

from prediction.outcome_prediction.LSTM.LSTM import lstm_generator
from prediction.outcome_prediction.LSTM.testing.test_LSTM import test_LSTM
from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.utils.utils import save_json, ensure_dir
from prediction.utils.scoring import precision, recall, matthews


def LSTM_inference(X, y, model_weights_path, activation, batch, data, dropout, layers, masking, optimizer, outcome, units,
              n_time_steps, n_channels):
    model = lstm_generator(x_time_shape=n_time_steps, x_channels_shape=n_channels, masking=masking, n_units=units,
                           activation=activation, dropout=dropout, n_layers=layers)

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', precision, recall, matthews])

    model.load_weights(model_weights_path)

    # calculate overall model prediction
    y_pred_test = model.predict(X)

    return (y, y_pred_test)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction of train & validation sets by LSTM model')
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


    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    save_json(test_features_lookup_table,
              os.path.join(args.output_dir, 'test_lookup_dict.json'))

    n_time_steps = test_X_np.shape[1]
    n_channels = test_X_np.shape[-1]

    for cv_fold in tqdm(cv_folds):
        model_name = '_'.join([args.activation, str(args.batch),
                               args.data, str(args.dropout), str(args.layers),
                               str(args.masking), args.optimizer, args.outcome,
                               str(args.units), str(cv_fold)])
        model_weights_path = os.path.join(args.model_weights_dir, f'{model_name}.hdf5')
        output_dir = os.path.join(args.output_dir, f'test_LSTM_{model_name}')
        ensure_dir(output_dir)
        shutil.copy2(model_weights_path, output_dir)

        (fold_X_train, fold_X_val, fold_y_train, fold_y_val) = splits[cv_fold - 1]

        train_pred_data = LSTM_inference(X=fold_X_train, y=fold_y_train, model_weights_path=model_weights_path,
                              activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
                              layers=args.layers, masking=args.masking, optimizer=args.optimizer,
                              outcome=args.outcome, units=args.units, n_time_steps=n_time_steps, n_channels=n_channels)

        val_pred_data = LSTM_inference(X=fold_X_val, y=fold_y_val, model_weights_path=model_weights_path,
                              activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
                              layers=args.layers, masking=args.masking, optimizer=args.optimizer,
                              outcome=args.outcome, units=args.units, n_time_steps=n_time_steps, n_channels=n_channels)

        # save ground truth and predictions
        pickle.dump(train_pred_data, open(os.path.join(output_dir, f'train_gt_and_pred_fold_{cv_fold}.pkl'), 'wb'))
        pickle.dump(val_pred_data, open(os.path.join(output_dir, f'val_gt_and_pred_fold_{cv_fold}.pkl'), 'wb'))

