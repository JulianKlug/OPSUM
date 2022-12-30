import argparse
import os
import pickle
from sklearn.model_selection import train_test_split
from prediction.mrs_outcome_prediction.data_loading.data_formatting import features_to_numpy, \
    link_patient_id_to_outcome, feature_order_verification

from prediction.mrs_outcome_prediction.LSTM.testing.compute_shap_explanations_over_time import \
    compute_shap_explanations_over_time
from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time
from prediction.utils.utils import ensure_dir, check_data


def compute_shap_explanations_over_time_wrapper(model_weights_path:str, outcome,
                                                training_features_path:str, training_labels_path:str,
                                                validation_features_path:str, validation_labels_path:str,
                                                out_dir:str,
                                                config:dict):

    ## LOAD THE TRAINING DATASET
    derivation_X, derivation_y = format_to_2d_table_with_time(feature_df_path=training_features_path, outcome_df_path=training_labels_path,
                                        outcome=outcome)

    check_data(derivation_X)

    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids_with_outcome = link_patient_id_to_outcome(derivation_y, outcome)
    pid_train, pid_test, _, _ = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                    all_pids_with_outcome.outcome.tolist(),
                                                    stratify=all_pids_with_outcome.outcome.tolist(),
                                                    test_size=config['test_size'],
                                                    random_state=config['seed'])
    train_X_df = derivation_X[derivation_X.patient_id.isin(pid_train)]
    train_X_np = features_to_numpy(train_X_df,
                                   ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    feature_order_verification(train_X_np)
    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    train_X_np = train_X_np[:, :, :, -1].astype('float32')


    ## LOAD THE EXTERNAL VALIDATION DATASET
    test_X, _ = format_to_2d_table_with_time(feature_df_path=validation_features_path, outcome_df_path=validation_labels_path,
                                        outcome=outcome)

    # test if data is corrupted
    check_data(test_X)
    test_X_np = features_to_numpy(test_X,
                                  ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
    # ensure that the order of features (3rd dimension) is the one predefined for the model
    feature_order_verification(test_X_np)
    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')

    ## COMPUTE SHAP EXPLANATIONS
    n_time_steps = test_X.relative_sample_date_hourly_cat.max() + 1
    n_channels = test_X.sample_label.unique().shape[0]
    shap_values_over_ts = compute_shap_explanations_over_time(model_weights_path=model_weights_path, train_X_np=train_X_np, test_X_np=test_X_np,
                                                                n_time_steps=n_time_steps, n_channels=n_channels,
                                                                config=config)

    with open(os.path.join(out_dir, 'deep_explainer_shap_values_over_ts.pkl'),
              'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome in external validation data')
    parser.add_argument('-t', '--target_outcome', required=True, type=str, help='outcome (ex. 3M Death)')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='output directory')
    parser.add_argument('-tf', '--training_features_path', required=True, type=str, help='path to training features')
    parser.add_argument('-tl', '--training_labels_path', required=True, type=str, help='path to training labels')
    parser.add_argument('-vf', '--validation_features_path', required=True, type=str, help='path to validation features')
    parser.add_argument('-vl', '--validation_labels_path', required=True, type=str, help='path to validation labels')
    parser.add_argument('-m', '--model_weights_path', required=True, type=str, help='path to model weights')
    parser.add_argument('--test_size', required=False, type=float, help='test set size [0-1]', default=0.2)
    parser.add_argument('--seed', required=False, type=int, help='Seed', default=42)
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
        'cv_fold': int(model_name.split('_')[9]),
        'seed' : args.seed,
        'test_size' : args.test_size,
    }

    compute_shap_explanations_over_time_wrapper(args.model_weights_path, args.target_outcome,
                                                args.training_features_path, args.training_labels_path,
                                                args.validation_features_path, args.validation_labels_path,
                                                output_dir, model_config)


