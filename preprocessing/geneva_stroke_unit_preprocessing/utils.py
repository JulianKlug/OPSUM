import pandas as pd
import numpy as np


def remove_french_accents_and_cedillas_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes(include=[np.object]).columns
    df[cols] = df[cols].apply(
        lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

    return df


def safe_conversion_to_numeric(df, column):
    remaining_non_numerical_values = df[pd.to_numeric(df[column], errors='coerce').isnull()][column].unique()
    if len(remaining_non_numerical_values) > 0:
        raise ValueError(f'Remaining non-numerical values: {remaining_non_numerical_values}')
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def restrict_variable_to_possible_ranges(df, variable_name, possible_value_ranges, verbose=False):
    """
    Restricts a variable to the possible ranges in the possible_value_ranges dataframe.
    """
    variable_range = possible_value_ranges[possible_value_ranges['variable_label'] == variable_name]
    variable_range = variable_range.iloc[0]
    clean_df = df.copy()
    clean_df[variable_name] = df[variable_name].apply(
        lambda x: np.nan if x < variable_range['Min'] or x > variable_range['Max'] else x)
    if verbose:
        print(f'Excluding {clean_df[variable_name].isna().sum()} observations because out of range')
    excluded_df = df[clean_df[variable_name].isna()]
    clean_df = clean_df.dropna(axis=0, subset=[variable_name])
    return clean_df, excluded_df


def create_ehr_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits)
    case_identification_column = df['patient_id'].astype(str) \
                                 + '_' + df['eds_end_4digit'].str.zfill(4).astype(str)
    return case_identification_column


def create_registry_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits)
    df = df.copy()
    if 'patient_id' not in df.columns:
        df['patient_id'] = df['Case ID'].apply(lambda x: x[8:-4]).astype(str)
    if 'EDS_last_4_digits' not in df.columns:
        df['EDS_last_4_digits'] = df['Case ID'].apply(lambda x: x[-4:]).astype(str)
    case_identification_column = df['patient_id'].astype(str) \
                                 + '_' + df['EDS_last_4_digits'].str.zfill(4).astype(str)
    return case_identification_column


def correct_overwritten_patient_id(df:pd.DataFrame, eds_df:pd.DataFrame) -> pd.DataFrame:
    """
    Correct for overwritten patient_id
    Overwritten patient_ids can be detected with / ehr_extraction_verification / detect_overwritten_patient_ids.py
    Gist: in some extractions, the patient_id is overwritten with the eds_final_patient_id

    Retrieve correct patient id by matching 'patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual'
    on scale_df side with 'eds_final_patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual' on eds_df

    Args:
        df: dataframe with overwritten patient_id
        eds_df: dataframe with correct patient_id used as reference
    Returns:
        df: dataframe with corrected patient_id
    """

    truncated_eds_df = eds_df[['patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual', 'eds_final_patient_id']]
    truncated_eds_df.drop_duplicates(inplace=True)
    df = pd.merge(df, truncated_eds_df,
                         left_on=['patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual'],
                         right_on=['eds_final_patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual'],
                         suffixes=('', '_eds'), how='left')

    df.drop(['patient_id', 'eds_final_patient_id_eds', 'nr'], axis=1, inplace=True)
    df.rename(columns={'patient_id_eds': 'patient_id'}, inplace=True)

    return df

