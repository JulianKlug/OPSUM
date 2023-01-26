import pandas as pd
import os

def apply_further_exclusion_criteria(patient_selection: list, admission_table_path:str, log_dir:str) -> set:
    """
    Applies further exclusion criteria:
        - Exclude patients with time of death during surveillance period

    Returns a set of hadm_ids to exclude.

    Parameters
    ----------
    patient_selection : list
        list of patient case_admission_ids to apply exclusion criteria to

    admission_table_path : str
        path to admission table

    log_dir : str
        path to log directory

    Returns
    -------
    hadm_ids_to_exclude : set
    """

    # Exclude patients with time of death during surveillance period
    # (i.e. death in the ICU within 72 hours of admission)
    admission_table = pd.read_csv(admission_table_path)
    admission_table['case_admission_id'] = admission_table['hadm_id'].astype(str) + '_' + admission_table[
        'icustay_id'].astype(str)
    admission_table = admission_table[admission_table['case_admission_id'].isin(patient_selection)]
    date_format = '%Y-%m-%d %H:%M:%S'
    admission_table['admit_to_dod_under_72h'] = (pd.to_datetime(admission_table['admittime'],
                                                              format=date_format) - pd.to_datetime(
        admission_table['deathtime'], format=date_format)).dt.total_seconds() / 60 / 60 > -72
    admission_table['death_in_ICU'] = admission_table['deathtime'] < admission_table['outtime']
    admission_table['death_during_surveillance'] = admission_table['admit_to_dod_under_72h'] & admission_table[
        'admit_to_dod_under_72h']

    hadm_ids_to_exclude = set(admission_table[admission_table['death_during_surveillance']]['case_admission_id'])

    # save logs
    admission_table[admission_table['death_during_surveillance']][['case_admission_id', 'admittime', 'outtime', 'deathtime']]\
        .drop_duplicates().to_csv(os.path.join(log_dir, 'excluded_patients_death_during_surveillance.csv'), index=False)

    return hadm_ids_to_exclude