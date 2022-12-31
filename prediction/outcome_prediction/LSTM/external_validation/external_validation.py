import argparse
import os
import shutil
import numpy as np
import pandas as pd
from modun.file_io import ensure_dir

from prediction.outcome_prediction.LSTM.testing.test_LSTM import test_LSTM
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    features_to_numpy, numpy_to_lookup_table, feature_order_verification
from prediction.utils.utils import check_data, save_json


def external_validation(model_weights_dir: str, features_path: str, labels_path: str, output_dir: str,
                        model_config: dict, outcome: str):
    model_name = '_'.join([model_config['activation'], str(model_config['batch']),
                           model_config['data'], str(model_config['dropout']), str(model_config['layers']),
                           str(model_config['masking']), model_config['optimizer'], outcome,
                           str(model_config['units']), str(model_config['cv_fold'])])

    model_weights_path = os.path.join(model_weights_dir, f'{model_name}.hdf5')
    output_dir = os.path.join(output_dir, f'test_LSTM_{model_name}')
    ensure_dir(output_dir)
    shutil.copy2(model_weights_path, output_dir)

    # load the dataset
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    # test if data is corrupted
    check_data(X)

    test_X_np = features_to_numpy(X,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])

    # ensure that the order of features (3rd dimension) is the one predefined for the model
    feature_order_verification(test_X_np)

    test_y_np = np.array([y[y.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')

    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    save_json(numpy_to_lookup_table(test_X_np),
              os.path.join(output_dir, 'test_lookup_dict.json'))

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')

    result_df, roc_auc_figure = test_LSTM(X=test_X_np, y=test_y_np, model_weights_path=model_weights_path,
                          activation=model_config['activation'], batch=model_config['batch'], data=model_config['data'],
                          dropout=model_config['dropout'],
                          layers=model_config['layers'], masking=model_config['masking'],
                          optimizer=model_config['optimizer'],
                          outcome=outcome, units=model_config['units'], n_time_steps=n_time_steps,
                          n_channels=n_channels)

    roc_auc_figure.savefig(os.path.join(output_dir, 'roc_auc_curve.png'))

    result_df.to_csv(os.path.join(output_dir, 'external_validation_LSTM_results.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome in external validation data')
    parser.add_argument('-t', '--target_outcome', required=True, type=str, help='outcome (ex. 3M Death)')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('-f', '--features_path', required=True, type=str, help='path to features')
    parser.add_argument('-l', '--labels_path', required=True, type=str, help='path to labels')
    parser.add_argument('-m', '--model_weights_path', required=True, type=str, help='path to model weights')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_weights_path).split('.hdf5')[0]

    model_config = {
        'activation': model_name.split('_')[0],
        'batch': model_name.split('_')[1],
        'data': model_name.split('_')[2],
        'dropout': float(model_name.split('_')[3]),
        'layers': int(model_name.split('_')[4]),
        'masking': model_name.split('_')[5],
        'optimizer': model_name.split('_')[6],
        'units': int(model_name.split('_')[8]),
        'cv_fold': int(model_name.split('_')[9])
    }

    external_validation(os.path.dirname(args.model_weights_path), args.features_path, args.labels_path, args.output_dir, model_config,
                        args.target_outcome)
