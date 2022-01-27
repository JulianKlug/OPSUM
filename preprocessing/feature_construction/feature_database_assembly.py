import pandas as pd
import os

from preprocessing.admission_params_preprocessing.admission_params_preprocessing import preprocess_admission_data
from preprocessing.lab_preprocessing.lab_preprocessing import preprocess_labs
from preprocessing.scales_preprocessing.scales_preprocessing import preprocess_scales
from preprocessing.ventilation_preprocessing.ventilation_preprocessing import preprocess_ventilation
from preprocessing.vitals_preprocessing.vitals_preprocessing import preprocess_vitals


def load_data_from_main_dir(data_path:str, file_start:str) -> pd.DataFrame:
    files = [pd.read_csv(os.path.join(data_path, f), delimiter=';', encoding='utf-8',
                         dtype=str)
                    for f in os.listdir(data_path)
                    if f.startswith(file_start)]
    return pd.concat(files, ignore_index=True)


def assemble_feature_database(raw_data_path:str, admission_data_path:str, patient_selection_path:str, verbose:bool=False,
                              use_admission_data:bool=True) -> pd.DataFrame:
    """
    Assemble database from lab/scales/ventilation/vitals subparts
    :return: Dataframe with all features under sample_label, value, sample_date
    """

    patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
    patient_selection_df['case_admission_id'] = patient_selection_df['patient_id'].astype(str) \
                                 + patient_selection_df['EDS_last_4_digits'].astype(str) \
                                 + '_' + pd.to_datetime(patient_selection_df['Arrival at hospital'], format='%Y%m%d').dt.strftime('%d%m%Y').astype(str)
    # TODO restrict to patient selection

    # load eds data
    eds_df = pd.read_csv(os.path.join(raw_data_path, 'eds_j1.csv'), delimiter=';', encoding='utf-8',
                         dtype=str)

    # Load and preprocess lab data
    lab_file_start = 'labo'
    lab_df = load_data_from_main_dir(raw_data_path, lab_file_start)
    preprocessed_lab_df = preprocess_labs(lab_df, verbose=verbose)
    preprocessed_lab_df = preprocessed_lab_df[['case_admission_id','sample_date','dosage_label','value']]
    preprocessed_lab_df.rename(columns={'dosage_label': 'sample_label'}, inplace=True)

    # Load and preprocess scales data
    scales_file_start = 'scale'
    scales_df = load_data_from_main_dir(raw_data_path, scales_file_start)
    scales_df = preprocess_scales(scales_df, eds_df, verbose=verbose)
    scales_df = scales_df[['scale','event_date','score','case_admission_id']]
    scales_df.rename(columns={'scale': 'sample_label', 'score':'value', 'event_date':'sample_date'}, inplace=True)

    # Load and preprocess ventilation data
    ventilation_file_start = 'ventilation'
    ventilation_df = load_data_from_main_dir(raw_data_path, ventilation_file_start)
    fio2_df, spo2_df = preprocess_ventilation(ventilation_df, verbose=verbose)
    fio2_df = fio2_df[['case_admission_id', 'FIO2', 'datetime']]
    fio2_df['sample_label'] = 'FIO2'
    fio2_df.rename(columns={'FIO2': 'value', 'datetime':'sample_date'}, inplace=True)
    spo2_df = spo2_df[['case_admission_id', 'spo2', 'datetime']]
    spo2_df['sample_label'] = 'oxygen_saturation'
    spo2_df.rename(columns={'spo2': 'value', 'datetime':'sample_date'}, inplace=True)

    # Load and preprocess vitals data
    vitals_file_start = 'patientvalue'
    vitals_df = load_data_from_main_dir(raw_data_path, vitals_file_start)
    vitals_df = preprocess_vitals(vitals_df, verbose=verbose)
    vitals_df = vitals_df[['case_admission_id','datetime','vital_value','vital_name']]
    vitals_df.rename(columns={'vital_name': 'sample_label', 'vital_value':'value', 'datetime':'sample_date'}, inplace=True)


    # # Load and preprocess admission data
    # if use_admission_data:
        # admission_data_files = [file for file in os.listdir(admission_data_path) if file.startswith('SSR_cases_of_2018_')]
        # admission_data_tables = [pd.read_excel(os.path.join(admission_data_path, file), skiprows=[0, 1, 2, 3, 4, 5, 7]) for file in admission_data_files]
        # admission_data_df = pd.concat(admission_data_tables)
        # admission_data_df = preprocess_admission_data(admission_data_df, patient_selection_df, verbose=verbose)

    # Assemble feature database
    feature_database = pd.concat([preprocessed_lab_df, scales_df, fio2_df, spo2_df, vitals_df], ignore_index=True)

    # retain only case_admission_ids that are in patient_selection_df

    return feature_database

# data_path = '/Users/jk1/stroke_datasets/stroke_unit_dataset/per_value/Extraction_20211110'
# admission_data_path = '/Users/jk1/OneDrive - unige.ch/stroke_research/geneva_stroke_unit_dataset/data/stroke_registry/'
# patient_selection_path = '/Users/jk1/temp/opsum_extration_output/high_frequency_data_patient_selection.csv'
# assemble_feature_database(data_path, admission_data_path, patient_selection_path)