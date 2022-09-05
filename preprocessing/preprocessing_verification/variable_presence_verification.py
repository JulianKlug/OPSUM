import os
from tqdm import tqdm
import pandas as pd


def assert_selected_variables_presence(df: pd.DataFrame, selected_variables: list) -> bool:
    """
    Asserts that all variables from the variable selection file are present in the dataframe.
    :param df: the dataframe to be checked
    :param selected_variables: the list of selected variables
    :return: None
    """
    missing_variables = []
    for variable in selected_variables:
        if (len([s for s in df.sample_label.unique() if variable in s]) == 0)\
                & (len([s for s in df.sample_label.unique() if variable.lower().replace(' ', '_') in s]) == 0):
            missing_variables.append(variable)

    if len(missing_variables) > 0:
        raise ValueError(f'The following variables are missing from the dataframe: {missing_variables}')

    return True


def variable_presence_verification(normalised_df: pd.DataFrame, desired_time_range:int=72) -> bool:

    # Verifying presence of all features
    all_variables_present = []
    selected_variables_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'variable_assembly/selected_variables.xlsx')
    selected_variables = pd.read_excel(selected_variables_path)['included']

    for cid in tqdm(normalised_df.case_admission_id.unique()):
        temp_cid_df = normalised_df[(normalised_df.case_admission_id == cid)]
        for time_bin in range(desired_time_range):
            all_variables_present.append(
                assert_selected_variables_presence(temp_cid_df[temp_cid_df.relative_sample_date_hourly_cat == time_bin],
                                                   selected_variables))
    assert all(all_variables_present), 'Not all selected variables are present in the final dataset'

    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_df_path', type=str, required=True)
    parser.add_argument('--desired_time_range', type=int, default=72)
    args = parser.parse_args()

    normalised_df = pd.read_csv(args.preprocessed_df_path)
    variable_presence_verification(normalised_df, args.desired_time_range)




