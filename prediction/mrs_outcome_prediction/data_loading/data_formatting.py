import numpy as np
import pandas as pd


def binarize_to_int(y):
    if (y == 'yes') | (y == True):
        return 1
    elif (y == 'no') | (y == False):
        return 0
    else:
        return np.nan


def format_to_linear_table(feature_df_path:str, outcome_df_path: str, outcome:str) -> (np.array, np.array):
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


