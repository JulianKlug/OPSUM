import pandas as pd


def preprocess_outcomes(outcome_data_path:str, verbose:bool=False) -> pd.DataFrame:
    """Preprocesses the outcome data.
    Patients with no dod are presumed to be alive at 3 months. (See: https://github.com/MIT-LCP/mimic-code/issues/185#issuecomment-285728687
    Target outcome variables:
        - 3M Death
        - Death in hospital

    Args:
        outcome_data_path (str): Path to the outcome data.
        verbose (bool, optional):

    Returns:
        pd.DataFrame: Preprocessed outcome data.
    """
    outcome_df = pd.read_csv(outcome_data_path)
    datatime_format = '%Y-%m-%d %H:%M:%S'

    # Preprocess In-hospital death
    outcome_df['Death in hospital'] = pd.to_datetime(outcome_df['dod'], format=datatime_format) <= pd.to_datetime(
        outcome_df['dischtime'], format=datatime_format)
    outcome_df['Death in hospital'] = outcome_df['Death in hospital'].astype(int)

    # Preprocess 3M Death (3M = 3 months after admission)
    outcome_df['3m_date'] = pd.to_datetime(outcome_df['admittime'], format=datatime_format) + pd.DateOffset(months=3)
    outcome_df['3M Death'] = pd.to_datetime(outcome_df['dod'], format=datatime_format) <= pd.to_datetime(
        outcome_df['3m_date'], format=datatime_format)
    outcome_df['3M Death'] = outcome_df['3M Death'].astype(int)

    # verify that all patients with in-hospital death are dead at 3M
    assert (outcome_df['Death in hospital'] == 1).all() == (outcome_df['3M Death'] == 1).all()

    outcome_df['case_admission_id'] = outcome_df['hadm_id'].astype(str) + '_' + outcome_df['icustay_id'].astype(str)

    outcome_df = outcome_df[['case_admission_id', 'Death in hospital', '3M Death']]

    return outcome_df




