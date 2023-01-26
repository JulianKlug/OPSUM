import os
import pandas as pd

from preprocessing.mimic_preprocessing.admission_preprocessing.admission_preprocessing import preprocess_admission
from preprocessing.mimic_preprocessing.database_assembly.further_exclusion_criteria import \
    apply_further_exclusion_criteria
from preprocessing.mimic_preprocessing.lab_preprocessing.lab_preprocessing import preprocess_labs
from preprocessing.mimic_preprocessing.monitoring_preprocessing.monitoring_preprocessing import preprocess_monitoring


def assemble_variable_database(extracted_tables_path: str, admission_notes_data_path: str,
                               preproccessed_monitoring_data_path: str = '',
                                mimic_admission_nihss_db_path: str = '',
                                verbose: bool = False, log_dir:str = '') -> pd.DataFrame:
    """
    1. Restrict to patient selection (only use patients with extracted notes)
    2. Preprocess EHR and notes data
    3. Restrict to variable selection
    4. Assemble database from lab/scales/ventilation/vitals + data extracted from notes subparts
    :return: Dataframe with all features under sample_label, value, sample_date, source
    """

    target_columns = ['case_admission_id', 'sample_date', 'sample_label', 'value', 'source']
    date_format = '%Y-%m-%d %H:%M:%S'

    # Extract data from admission tables and from the preprocessed notes
    admission_table_path = os.path.join(extracted_tables_path, 'admission_df.csv')
    admission_data_df = preprocess_admission(admission_notes_data_path, admission_table_path, verbose=verbose)
    admission_data_df['case_admission_id'] = admission_data_df['hadm_id'].astype(str) + '_' + admission_data_df['icustay_id'].astype(str)
    admission_data_df.rename(columns={'admittime': 'sample_date'}, inplace=True)
    try:
        admission_data_df['sample_date'] = pd.to_datetime(admission_data_df['sample_date'], format=date_format)
    except ValueError:
        raise ValueError('Date format is not correct. Please check the date format in the admission data.')
    admission_data_df['source'] = 'notes'
    admission_data_df = admission_data_df[target_columns]

    # Apply further exclusion criteria
    hadm_ids_to_exclude = apply_further_exclusion_criteria(admission_data_df['case_admission_id'].unique(), admission_table_path, log_dir=log_dir)
    admission_data_df = admission_data_df[~admission_data_df['case_admission_id'].isin(hadm_ids_to_exclude)]

    # Deduce patient selection from admission data
    patient_selection = admission_data_df['case_admission_id'].unique()

    # Preprocess lab data
    lab_data_path = os.path.join(extracted_tables_path, 'lab_df.csv')
    lab_data_df = pd.read_csv(lab_data_path)
    lab_data_df['case_admission_id'] = lab_data_df['hadm_id'].astype(str) + '_' + lab_data_df['icustay_id'].astype(str)
    lab_data_df = lab_data_df[lab_data_df['case_admission_id'].isin(patient_selection)]
    lab_data_df = preprocess_labs(lab_data_df, log_dir=log_dir, verbose=verbose)
    lab_data_df.drop(columns=['value'], inplace=True)
    lab_data_df.rename(columns={'charttime': 'sample_date', 'valuenum': 'value', 'label': 'sample_label'}, inplace=True)
    lab_data_df['sample_date'] = pd.to_datetime(lab_data_df['sample_date'], format=date_format)
    lab_data_df['source'] = 'EHR'
    lab_data_df = lab_data_df[target_columns]

    # Preprocess monitoring data
    if preproccessed_monitoring_data_path != '':
        monitoring_data_df = pd.read_csv(preproccessed_monitoring_data_path)
    else:
        if mimic_admission_nihss_db_path == '':
            raise ValueError('Please provide a path to the MIMIC admission nihss database.')
        monitoring_df = pd.read_csv(os.path.join(extracted_tables_path, 'monitoring_df.csv'))
        monitoring_data_df = preprocess_monitoring(monitoring_df, mimic_admission_nihss_db_path, verbose)

    monitoring_data_df['case_admission_id'] = monitoring_data_df['hadm_id'].astype(int).astype(str) + '_' + monitoring_data_df['icustay_id'].astype(int).astype(str)
    monitoring_data_df = monitoring_data_df[monitoring_data_df['case_admission_id'].isin(patient_selection)]
    monitoring_data_df.drop(columns=['value'], inplace=True)
    monitoring_data_df.rename(columns={'charttime': 'sample_date', 'valuenum': 'value', 'label': 'sample_label'}, inplace=True)
    monitoring_data_df['sample_date'] = pd.to_datetime(monitoring_data_df['sample_date'], format=date_format)
    monitoring_data_df['source'] = 'EHR'
    monitoring_data_df = monitoring_data_df[target_columns]

    # Assemble database
    database_df = pd.concat([admission_data_df, lab_data_df, monitoring_data_df], axis=0)

    return database_df

