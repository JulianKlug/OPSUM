import numpy as np
import pandas as pd


def binarize_to_int(y):
    if (y == 'yes') | (y == True):
        return 1
    elif (y == 'no') | (y == False):
        return 0
    else:
        return np.nan


def format_to_linear_table(feature_df_path: str, outcome_df_path: str, outcome: str) -> (np.array, np.array):
    """
    This function formats the data into a linear table with one row per case_admission_id, and one column per feature over time.
    The time dimension is thus disregarded.
    :param feature_df_path: path to the feature dataframe
    :param outcome_df_path: path to the outcome dataframe
    :param outcome: the outcome to be predicted
    :return: a tuple of the form (X, y) where X is a numpy array of shape (n_samples, n_features) and y is a numpy array of shape (n_samples, 1)
    """

    # Load the data
    features_df = pd.read_csv(feature_df_path)
    outcome_df = pd.read_csv(outcome_df_path)
    features_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # keep one row per case_admission_id
    pivoted_features = pd.pivot_table(features_df, index='case_admission_id', values=['value'],
                                      columns=['relative_sample_date_hourly_cat', 'sample_label'])
    pivoted_features.columns = [f'{col[2]}_hcat_{col[1]}' for col in pivoted_features.columns.values]
    pivoted_features_np = pivoted_features.reset_index().values

    X = pivoted_features_np[:, 1:]

    y = [outcome_df[outcome_df.case_admission_id == id][outcome].values[0]
         if len(outcome_df[outcome_df.case_admission_id == id][outcome].values) > 0
         else np.nan
         for id in pivoted_features_np[:, 0]]

    y = list(map(binarize_to_int, y))

    # find case_admission_ids where y is nan
    cid_with_no_outcome = pivoted_features_np[np.isnan(y), 0]
    print('Found {} case_admission_ids with no outcome. These will be excluded.'.format(len(cid_with_no_outcome)))

    # remove values in X and y where y is nan
    X = X[~np.isnan(y)]
    y = np.array(y)[~np.isnan(y)]

    return X, y


def format_to_2d_table_with_time(feature_df_path: str, outcome_df_path: str, outcome: str) -> (
pd.DataFrame, pd.DataFrame):
    """
    This function formats the data into a 2d table per case_admission_id and per time.
    :param feature_df_path:
    :param outcome_df_path:
    :param outcome:
    :return: X, y as tables with X-columns case_admission_id, patient_id, relative_sample_date_hourly_cat, sample_label, value and y-columns case_admission_id, patient_id, outcome
    """

    # Load the data
    features_df = pd.read_csv(feature_df_path)
    outcome_df = pd.read_csv(outcome_df_path)
    features_df.drop(['Unnamed: 0'], axis=1, inplace=True)

    # add a patient id column
    features_df['patient_id'] = features_df['case_admission_id'].apply(lambda x: x.split('_')[0][:-4])

    X = features_df.copy()

    y = pd.DataFrame(X['case_admission_id'].unique(), columns=['case_admission_id'])

    y['patient_id'] = y['case_admission_id'].apply(lambda x: x.split('_')[0][:-4])
    y['outcome'] = y.case_admission_id.apply(lambda x:
                                             outcome_df[outcome_df.case_admission_id == x][outcome].values[0]
                                             if len(outcome_df[outcome_df.case_admission_id == x][outcome].values) > 0
                                             else np.nan)

    y['outcome'] = y['outcome'].apply(binarize_to_int)

    # find case_admission_ids where y is nan
    cid_with_no_outcome = y[y.outcome.isna()]['case_admission_id'].unique()
    print('Found {} case_admission_ids with no outcome. These will be excluded.'.format(len(cid_with_no_outcome)))

    # remove values in X and y where y is nan
    X = X[~X.case_admission_id.isin(cid_with_no_outcome)]
    y = y[~y.case_admission_id.isin(cid_with_no_outcome)]

    return X, y


def link_patient_id_to_outcome(y: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    This function links patient_id to a single outcome
    - if selected outcome is '3M mRS 0-2', then the pid is linked to the worst outcome among all admissions
    :param outcome:
    :param y:
    :return: DataFrame with patient_id and outcome
    """
    all_pids = y[['patient_id', 'outcome']].copy()

    if outcome != '3M mRS 0-2':
        raise ValueError('Reduction to single outcome is not implemented for {}'.format(outcome))

    # replaces duplicated patient_ids with a single patient_id with minimum outcome
    duplicated_pids = all_pids[all_pids.duplicated(subset='patient_id', keep=False)].copy()
    reduced_pids = duplicated_pids.groupby('patient_id').min().reset_index()

    all_pids_no_duplicates = all_pids[~all_pids.duplicated(subset='patient_id', keep=False)].copy()
    all_pids_no_duplicates = all_pids_no_duplicates.append(reduced_pids)

    return all_pids_no_duplicates


def features_to_numpy(X: pd.DataFrame,
                      columns: list = ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value']
                      ) -> np.ndarray:
    """
    This function converts a pandas dataframe to a 4d numpy array with shape (n_cases, n_time_steps, n_sample_labels,
    n_features)
    :param columns:
    :param X:
    :return:
    """
    df = X[columns].copy()
    # pass through the lists to ensure that the order is correct
    gb_cid = [x for _, x in df.groupby('case_admission_id')]
    df_np = np.array([[x for _, x in gb_cid_x.groupby('relative_sample_date_hourly_cat')] for gb_cid_x in gb_cid])
    return df_np


def numpy_to_lookup_table(df_np: np.ndarray) -> dict:
    """
    This function expects a 4d numpy array with shape (n_cases, n_time_steps, n_sample_labels, n_features) where
    the features are case_admission_id, relative_sample_date_hourly_cat, sample_label, value
    :param df_np:
    :return:
    """
    return {
            'case_admission_id': {cid: i for i, cid in enumerate(df_np[:, 0, 0, 0])},
            'timestep': {t: i for i, t in enumerate(df_np[0, :, 0, 1])},
            'sample_label': {sl: i for i, sl in enumerate(df_np[0, 0, :, 2])}
        }

