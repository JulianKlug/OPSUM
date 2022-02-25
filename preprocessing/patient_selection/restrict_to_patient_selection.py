import pandas as pd


def restrict_to_patient_selection(variable_df: pd.DataFrame, patient_selection_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Restricts a dataframe to only the patients that are in the patient selection file.
    :param variable_df:
    :param patient_selection_path:
    :param verbose:
    :return:
    """

    patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
    patient_selection_df['case_admission_id'] = patient_selection_df['patient_id'].astype(str) \
                                 + patient_selection_df['EDS_last_4_digits'].astype(str) \
                                 + '_' + pd.to_datetime(patient_selection_df['Arrival at hospital'], format='%Y%m%d').dt.strftime('%d%m%Y').astype(str)

    restricted_to_selection_df = variable_df[
        variable_df['case_admission_id'].isin(patient_selection_df['case_admission_id'])]

    if verbose:
        print('Number of patients after selection:', len(restricted_to_selection_df['case_admission_id'].unique()))
        print('Number of patients not selected:', len(variable_df['case_admission_id'].unique()) - len(restricted_to_selection_df['case_admission_id'].unique()))
        print('Number of patients from selection that were not found:', len(patient_selection_df['case_admission_id'].unique()) - len(restricted_to_selection_df['case_admission_id'].unique()))

    return restricted_to_selection_df