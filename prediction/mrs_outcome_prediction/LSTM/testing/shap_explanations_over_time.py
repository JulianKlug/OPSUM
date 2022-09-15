import shap
import numpy as np
import pandas as pd
import os
from prediction.utils.scoring import precision, recall, matthews
from prediction.mrs_outcome_prediction.LSTM.LSTM import lstm_generator
import numpy as np
from random import randint
import pickle
from tqdm import tqdm

def compute_shap_explanations_over_time():
    model_weights_path = '/home/klug/data/opsum/models/sigmoid_all_unchanged_0.2_2_True_RMSprop_3M_mRS_0-2_128_4.hdf5'
    features_path = '/home/klug/data/opsum/72h_input_data/02092022_083046/preprocessed_features_02092022_083046.csv'
    labels_path = '/home/klug/data/opsum/72h_input_data/02092022_083046/preprocessed_outcomes_02092022_083046.csv'
    out_dir = '/home/klug/output/opsum/LSTM_72h_test_results/2022_09_07_1744'

    outcome = '3M mRS 0-2'
    masking = True
    units = 128
    activation = 'sigmoid'
    dropout = 0.2
    layers = 2
    optimizer = 'RMSprop'
    seed = 42
    test_size = 0.20
    override_masking_value = False

    from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time

    # load the dataset
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    from sklearn.model_selection import train_test_split
    from prediction.mrs_outcome_prediction.data_loading.data_formatting import features_to_numpy, \
        link_patient_id_to_outcome, numpy_to_lookup_table

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
    test_features_lookup_table = numpy_to_lookup_table(test_X_np)
    train_features_lookup_table = numpy_to_lookup_table(train_X_np)

    # Remove the case_admission_id, sample_label, and time_step_label columns from the data
    test_X_np = test_X_np[:, :, :, -1].astype('float32')
    train_X_np = train_X_np[:, :, :, -1].astype('float32')

    shap_values_over_ts = []

    for ts in tqdm(range(n_time_steps)):
        modified_time_steps = ts + 1
        model = lstm_generator(x_time_shape=modified_time_steps, x_channels_shape=n_channels, masking=override_masking_value,
                               n_units=units,
                               activation=activation, dropout=dropout, n_layers=layers)

        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy', precision, recall, matthews])

        model.load_weights(model_weights_path)

        test_X_with_first_n_ts = test_X_np[:, 0:modified_time_steps, :]

        # Use the training data for deep explainer => can use fewer instances
        explainer = shap.DeepExplainer(model, train_X_np[:, 0:modified_time_steps, :])
        # explain the testing instances (can use fewer instances)
        # explaining each prediction requires 2 * background dataset size runs
        shap_values = explainer.shap_values(test_X_with_first_n_ts)

        shap_values_over_ts.append(shap_values)

    with open(os.path.join(out_dir, '/deep_explainer_shap_values_over_ts.pkl'),
              'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)


