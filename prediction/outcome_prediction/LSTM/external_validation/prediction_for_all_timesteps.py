import argparse
import os
import pickle
import numpy as np
from modun.file_io import ensure_dir

from prediction.outcome_prediction.LSTM.testing.prediction_for_all_timesteps import prediction_for_all_timesteps
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    feature_order_verification
from prediction.outcome_prediction.data_loading.data_formatting import features_to_numpy, \
    numpy_to_lookup_table
from prediction.utils.utils import check_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
    parser.add_argument('-t', '--target_outcome', required=True, type=str, help='outcome (ex. 3M Death)')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('-f', '--features_path', required=True, type=str, help='path to features')
    parser.add_argument('-l', '--labels_path', required=True, type=str, help='path to labels')
    parser.add_argument('-m', '--model_weights_path', required=True, type=str, help='path to model weights')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_weights_path).split('.hdf5')[0]
    output_dir = os.path.join(args.output_dir, f'test_LSTM_{model_name}')
    ensure_dir(output_dir)

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

    # load the dataset
    X, y = format_to_2d_table_with_time(feature_df_path=args.features_path, outcome_df_path=args.labels_path,
                                        outcome=args.target_outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    # test if data is corrupted
    check_data(X)
    test_X_np = features_to_numpy(X,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    # ensure that the order of features (3rd dimension) is the one predefined for the model
    feature_order_verification(test_X_np)

    # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
    test_features_lookup_table = numpy_to_lookup_table(test_X_np)

    test_y_np = np.array([y[y.case_admission_id == cid].outcome.values[0] for cid in
                          test_X_np[:, 0, 0, 0]]).astype('float32')

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')


    predictions = prediction_for_all_timesteps(test_X_np, args.model_weights_path,
                                               n_time_steps=n_time_steps, n_channels=n_channels,
                                               config=model_config)

    # Save predictions as pickle
    with open(os.path.join(output_dir, 'predictions_over_timesteps.pkl'), 'wb') as f:
        pickle.dump(predictions, f)


