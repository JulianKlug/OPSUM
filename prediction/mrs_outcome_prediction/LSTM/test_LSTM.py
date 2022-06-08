import argparse

from sklearn.model_selection import train_test_split

from prediction.mrs_outcome_prediction.LSTM.LSTM import lstm_generator
from prediction.mrs_outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome
from prediction.utils.scoring import precision, recall, matthews
from prediction.utils.utils import check_data


def test_LSTM(X, y, activation, batch, data, dropout, layers, masking, optimizer, outcome, units):
    model = lstm_generator(x_time_shape=n_time_steps, x_channels_shape=n_channels, masking=masking, n_units=units,
                        activation=activation, dropout=dropout, n_layers=layers)

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', precision, recall, matthews])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model for predicting outcome')
    parser.add_argument('--date_string', required=True, type=str, help='datestring')
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

    # define constants
    output_dir = args.output_dir
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

    test_X = X[X.patient_id.isin(pid_test)]
    test_y = y[y.patient_id.isin(pid_test)]

    test_LSTM(X=test_X, y=test_y, activation=args.activation, batch=args.batch, data=args.data, dropout=args.dropout,
                     layers=args.layers, masking=args.masking, optimizer=args.optimizer,
                     outcome=args.outcome, units=args.units)
