import pandas as pd

from preprocessing.geneva_stroke_unit_preprocessing.utils import create_ehr_case_identification_column


def test_registry_and_extraction_start_date(registry_patient_selection_path, eds_j1_path):
    print('Testing if eds start date and admission date from registry correspond.')

    registry_data_df = pd.read_csv(registry_patient_selection_path)

    registry_data_df['case_admission_id'] = registry_data_df['patient_id'].astype(str) \
                                            + registry_data_df['EDS_last_4_digits'].astype(str) \
                                            + '_' + pd.to_datetime(registry_data_df['Arrival at hospital'],
                                                                   format='%Y%m%d').dt.strftime('%d%m%Y').astype(str)

    eds_df = pd.read_csv(eds_j1_path, delimiter=';')

    eds_df['case_admission_id'] = create_ehr_case_identification_column(eds_df)

    restricted_eds_df = eds_df[eds_df['case_admission_id'].isin(registry_data_df['case_admission_id'])]

    joined_df = pd.merge(registry_data_df, restricted_eds_df, on=['case_admission_id'], how='left')

    joined_df['registry_arrival_to_date_from_days'] = (
                pd.to_datetime(joined_df['date_from'], format='%d.%m.%Y %H:%M') - pd.to_datetime(
            joined_df['Arrival at hospital'], format='%Y%m%d')).dt.days

    differing_starting_date_df = joined_df[
        (joined_df['registry_arrival_to_date_from_days'] > 0) | (joined_df['registry_arrival_to_date_from_days'] < -10)]

    assert differing_starting_date_df.shape[0] == 0

    print('Test passed.')