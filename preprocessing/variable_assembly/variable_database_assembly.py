import pandas as pd
import os

from preprocessing.patient_selection.filter_ehr_patients import filter_ehr_patients
from preprocessing.patient_selection.restrict_to_patient_selection import restrict_to_patient_selection
from preprocessing.stroke_registry_params_preprocessing.admission_params_preprocessing import preprocess_admission_data
from preprocessing.lab_preprocessing.lab_preprocessing import preprocess_labs
from preprocessing.scales_preprocessing.scales_preprocessing import preprocess_scales
from preprocessing.stroke_registry_params_preprocessing.timing_params_preprocessing import preprocess_timing_params
from preprocessing.stroke_registry_params_preprocessing.treatment_params_preprocessing import \
    treatment_params_preprocessing
from preprocessing.stroke_registry_params_preprocessing.utils import set_sample_date
from preprocessing.utils import create_registry_case_identification_column
from preprocessing.variable_assembly.variable_selection import restrict_to_selected_variables
from preprocessing.ventilation_preprocessing.ventilation_preprocessing import preprocess_ventilation
from preprocessing.vitals_preprocessing.vitals_preprocessing import preprocess_vitals


def load_data_from_main_dir(data_path: str, file_start: str) -> pd.DataFrame:
    files = [pd.read_csv(os.path.join(data_path, f), delimiter=';', encoding='utf-8',
                         dtype=str)
             for f in os.listdir(data_path)
             if f.startswith(file_start)]
    return pd.concat(files, ignore_index=True)


def assemble_variable_database(raw_data_path: str, stroke_registry_data_path: str,
                               patient_selection_path: str, verbose: bool = False,
                               use_stroke_registry_data: bool = True,
                               log_dir:str = '') -> pd.DataFrame:
    """
    1. Restrict to patient selection (done after preprocessing for EHR data and before processing for stroke registry data)
    2. Preprocess EHR and stroke registry data
    3. Restrict to variable selection
    4. Assemble database from lab/scales/ventilation/vitals + stroke registry subparts
    :return: Dataframe with all features under sample_label, value, sample_date, source
    """
    # load eds data
    eds_df = pd.read_csv(os.path.join(raw_data_path, 'eds_j1.csv'), delimiter=';', encoding='utf-8',
                         dtype=str)
    eds_df = filter_ehr_patients(eds_df, patient_selection_path)

    # Load and preprocess lab data
    lab_file_start = 'labo'
    lab_df = load_data_from_main_dir(raw_data_path, lab_file_start)
    lab_df = filter_ehr_patients(lab_df, patient_selection_path)
    preprocessed_lab_df = preprocess_labs(lab_df, verbose=verbose, log_dir=log_dir)
    preprocessed_lab_df = preprocessed_lab_df[['case_admission_id', 'sample_date', 'dosage_label', 'value']]
    preprocessed_lab_df.rename(columns={'dosage_label': 'sample_label'}, inplace=True)
    preprocessed_lab_df['source'] = 'EHR'

    # Load and preprocess scales data
    scales_file_start = 'scale'
    scales_df = load_data_from_main_dir(raw_data_path, scales_file_start)
    scales_df = filter_ehr_patients(scales_df, patient_selection_path)
    scales_df = preprocess_scales(scales_df, eds_df, verbose=verbose)
    scales_df = scales_df[['scale', 'event_date', 'score', 'case_admission_id']]
    scales_df.rename(columns={'scale': 'sample_label', 'score': 'value', 'event_date': 'sample_date'}, inplace=True)
    scales_df['source'] = 'EHR'

    # Load and preprocess ventilation data
    ventilation_file_start = 'ventilation'
    ventilation_df = load_data_from_main_dir(raw_data_path, ventilation_file_start)
    ventilation_df = filter_ehr_patients(ventilation_df, patient_selection_path)
    fio2_df, spo2_df = preprocess_ventilation(ventilation_df, eds_df, verbose=verbose)
    fio2_df = fio2_df[['case_admission_id', 'FIO2', 'datetime']]
    fio2_df['sample_label'] = 'FIO2'
    fio2_df.rename(columns={'FIO2': 'value', 'datetime': 'sample_date'}, inplace=True)
    fio2_df['source'] = 'EHR'
    spo2_df = spo2_df[['case_admission_id', 'spo2', 'datetime']]
    spo2_df['sample_label'] = 'oxygen_saturation'
    spo2_df.rename(columns={'spo2': 'value', 'datetime': 'sample_date'}, inplace=True)
    spo2_df['source'] = 'EHR'

    # Load and preprocess vitals data
    vitals_file_start = 'patientvalue'
    vitals_df = load_data_from_main_dir(raw_data_path, vitals_file_start)
    vitals_df = filter_ehr_patients(vitals_df, patient_selection_path)
    vitals_df = preprocess_vitals(vitals_df, verbose=verbose)
    vitals_df = vitals_df[['case_admission_id', 'datetime', 'vital_value', 'vital_name']]
    vitals_df.rename(columns={'vital_name': 'sample_label', 'vital_value': 'value', 'datetime': 'sample_date'},
                     inplace=True)
    vitals_df['source'] = 'EHR'

    # Assemble feature database
    feature_database = pd.concat([preprocessed_lab_df, scales_df, fio2_df, spo2_df, vitals_df], ignore_index=True)
    feature_database = restrict_to_patient_selection(feature_database, patient_selection_path, verbose=verbose,
                                                     restrict_to_event_period=True)

    # Load and preprocess admission data from stroke registry
    if use_stroke_registry_data:
        if verbose:
            print('Preprocessing stroke registry_data')
        # Load stroke registry data and restrict to patient selection
        stroke_registry_df = pd.read_excel(stroke_registry_data_path)
        stroke_registry_df['patient_id'] = stroke_registry_df['Case ID'].apply(lambda x: x[8:-4])
        stroke_registry_df['EDS_last_4_digits'] = stroke_registry_df['Case ID'].apply(lambda x: x[-4:])
        stroke_registry_df['case_admission_id'] = create_registry_case_identification_column(stroke_registry_df)

        # set sample date to stroke onset or arrival at hospital, whichever is later
        stroke_registry_df = set_sample_date(stroke_registry_df)

        restricted_stroke_registry_df = restrict_to_patient_selection(stroke_registry_df, patient_selection_path,
                                                                      verbose=verbose, restrict_to_event_period=False)

        admission_data_df = preprocess_admission_data(restricted_stroke_registry_df, verbose=verbose)
        admission_data_df['source'] = 'stroke_registry'

        timings_df = preprocess_timing_params(restricted_stroke_registry_df)
        timings_df['source'] = 'stroke_registry'

        treatment_data_df = treatment_params_preprocessing(restricted_stroke_registry_df)
        treatment_data_df['source'] = 'stroke_registry'

        selected_stroke_registry_data_df = pd.concat([admission_data_df, timings_df, treatment_data_df],
                                                     ignore_index=True)

        # Only keep case_admissions that are in the EHR data and in the stroke registry data (intersection)
        # (FIO2 are excluded from this count, as it is inferred from when missing)
        ehr_cid_before_restriction_to_registry = feature_database[feature_database.sample_label != 'FIO2']['case_admission_id'].unique()
        intersection_ehr_registry = set(ehr_cid_before_restriction_to_registry)\
            .intersection(set(selected_stroke_registry_data_df['case_admission_id'].unique()))
        # 1. Restrict EHR to intersection (EHR /intersect/ registry)
        feature_database = feature_database[feature_database['case_admission_id'].isin(intersection_ehr_registry)]
        print('Number of cases from EHR data (after restriction to patient selection) not found in registry:',
              len(ehr_cid_before_restriction_to_registry) - len(feature_database[feature_database.sample_label != 'FIO2']['case_admission_id'].unique()))
        # 2. Restrict registry to intersection (EHR /intersect/ registry)
        n_registry_cid_before_restriction_to_ehr = len(selected_stroke_registry_data_df['case_admission_id'].unique())
        selected_stroke_registry_data_df = selected_stroke_registry_data_df[
            selected_stroke_registry_data_df['case_admission_id'].isin(intersection_ehr_registry)]
        print('Number of cases from registry not found in EHR data:', n_registry_cid_before_restriction_to_ehr - len(
            selected_stroke_registry_data_df['case_admission_id'].unique()))

        feature_database = pd.concat([feature_database, selected_stroke_registry_data_df], ignore_index=True)

    # Restrict to variable selection
    variable_selection_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selected_variables.xlsx')
    feature_database = restrict_to_selected_variables(feature_database, variable_selection_path)

    if log_dir != '':
        # save cids of patients in selection and that are not included in the feature_database
        patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
        cids_in_selection = set(create_registry_case_identification_column(patient_selection_df).unique())
        cids_in_feature_database = set(feature_database['case_admission_id'].unique())
        cids_not_found_in_feature_database = cids_in_selection.difference(cids_in_feature_database)
        cids_in_database_not_found_in_selection = cids_in_feature_database.difference(cids_in_selection)
        assert len(cids_in_database_not_found_in_selection) == 0, 'Unselected patients found in database'
        cids_not_found_in_feature_database_df = pd.DataFrame(cids_not_found_in_feature_database, columns=['case_admission_id'])
        cids_not_found_in_feature_database_df.to_csv(os.path.join(log_dir, 'missing_cids_from_feature_database.tsv'), sep='\t', index=False)

    return feature_database
