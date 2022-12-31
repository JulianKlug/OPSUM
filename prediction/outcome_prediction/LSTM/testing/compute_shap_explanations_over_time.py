import argparse
import shap
import os

from prediction.outcome_prediction.LSTM.testing.shap_helper_functions import check_shap_version_compatibility
from prediction.utils.scoring import precision, recall, matthews
from prediction.outcome_prediction.LSTM.LSTM import lstm_generator
import numpy as np
import pickle
from tqdm import tqdm
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time

# Shap values require very specific versions
check_shap_version_compatibility()

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

def compute_shap_explanations_over_time_wrapper(model_weights_path:str, features_path:str, labels_path:str, out_dir:str, config:dict=DEFAULT_CONFIG):

    # load the dataset
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=config['outcome'])

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    from sklearn.model_selection import train_test_split
    from prediction.outcome_prediction.data_loading.data_formatting import features_to_numpy, \
        link_patient_id_to_outcome, numpy_to_lookup_table

    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids_with_outcome = link_patient_id_to_outcome(y, config['outcome'])
    pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                                    all_pids_with_outcome.outcome.tolist(),
                                                                    stratify=all_pids_with_outcome.outcome.tolist(),
                                                                    test_size=config['test_size'],
                                                                    random_state=config['seed'])

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

    shap_values_over_ts = compute_shap_explanations_over_time(model_weights_path=model_weights_path, train_X_np=train_X_np, test_X_np=test_X_np,
                                                                n_time_steps=n_time_steps, n_channels=n_channels,
                                                                config=config)

    with open(os.path.join(out_dir, '/deep_explainer_shap_values_over_ts.pkl'),
              'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)


def compute_shap_explanations_over_time(model_weights_path:str, train_X_np:np.ndarray, test_X_np:np.ndarray,
                                        n_time_steps:int, n_channels:int, config:dict=DEFAULT_CONFIG):
    shap_values_over_ts = []

    # Masking has to be overriden
    override_masking_value = False

    for ts in tqdm(range(n_time_steps)):
        modified_time_steps = ts + 1
        model = lstm_generator(x_time_shape=modified_time_steps, x_channels_shape=n_channels, masking=override_masking_value,
                               n_units=config['units'],
                               activation=config['activation'], dropout=config['dropout'], n_layers=config['layers'])

        model.compile(loss='binary_crossentropy', optimizer=config['optimizer'],
                      metrics=['accuracy', precision, recall, matthews])

        model.load_weights(model_weights_path)

        test_X_with_first_n_ts = test_X_np[:, 0:modified_time_steps, :]

        # Use the training data for deep explainer => can use fewer instances
        explainer = shap.DeepExplainer(model, train_X_np[:, 0:modified_time_steps, :])
        # explain the testing instances (can use fewer instances)
        # explaining each prediction requires 2 * background dataset size runs
        shap_values = explainer.shap_values(test_X_with_first_n_ts)

        shap_values_over_ts.append(shap_values)


    return shap_values_over_ts



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

    compute_shap_explanations_over_time_wrapper(model_weights_path=args.model_weights_path,
                                        features_path=args.features_path,
                                        labels_path=args.labels_path,
                                        out_dir=args.output_dir,
                                        config=config)


