from preprocessing.geneva_stroke_unit_preprocessing.variable_assembly.variable_database_assembly import assemble_feature_database
import pandas as pd

def test_for_absence_of_data(data_path, admission_data_path, patient_selection_path):
    print('Checking for patients with no data')
    feature_database = assemble_feature_database(data_path, admission_data_path, patient_selection_path,
                                                 use_admission_data=False)

    patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
    patient_selection_df['case_admission_id'] = patient_selection_df['patient_id'].astype(str) \
                                                + patient_selection_df['EDS_last_4_digits'].astype(str) \
                                                + '_' + pd.to_datetime(patient_selection_df['Arrival at hospital'],
                                                                       format='%Y%m%d').dt.strftime('%d%m%Y').astype(
    str)
    restricted_to_registry_df = feature_database[
        feature_database['case_admission_id'].isin(patient_selection_df['case_admission_id'])]

    case_admission_ids_with_no_data = (set(patient_selection_df['case_admission_id'].unique()) - set(
        restricted_to_registry_df['case_admission_id'].unique()))
    patients_with_no_data = patient_selection_df[
        patient_selection_df['case_admission_id'].isin(case_admission_ids_with_no_data)]

    print('Patients with no data:', len(patients_with_no_data),patients_with_no_data)
    assert len(patients_with_no_data) == 0

