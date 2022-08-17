import pandas as pd


def filter_ehr_patients(df, patient_selection_path: str = '') -> pd.DataFrame:
    """
    Filter out:
    - patients with missmatch in patient_id (from registry) and eds_final_patient_id (from EHR extraction)
        - Goal: avoid using data from another patient that was mistakenly selected during EHR extraction
        - Exception: patient that have been matched by manually completed EDS or ID (patient_id_manual, eds_manual)

    Arguments:
        df {pd.DataFrame} -- Dataframe with EHR data
        patient_selection_path {str} -- Path to patient selection file
    Returns:
        pd.DataFrame -- Filtered EHR dataframe
    """

    # If available merge with information obtained from patient selection file
    if patient_selection_path != '':
        patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
        patient_selection_df.rename(columns={'EDS_last_4_digits': 'eds_end_4digit', 'manual_eds': 'eds_manual',
                                             'manual_patient_id': 'patient_id_manual'}, inplace=True)
        patient_selection_df = patient_selection_df[['patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual']]
        patient_selection_df.drop_duplicates(inplace=True)
        merged_df = pd.merge(df, patient_selection_df, on=['patient_id', 'eds_end_4digit'], how='left',
                             suffixes=('', '_y'))

        # Fill missing manually completed EDS and patient_id with values from patient selection file
        merged_df.patient_id_manual.fillna(merged_df.patient_id_manual_y, inplace=True)
        merged_df.eds_manual.fillna(merged_df.eds_manual_y, inplace=True)
        merged_df.drop(columns=['eds_manual_y', 'patient_id_manual_y'], inplace=True)
        df = merged_df
    else:
        # Print a warning if patient selection file is not available
        print('WARNING: Patient selection file not available')

    # drop selected rows
    filtered_df = df.drop(
        df[(df['patient_id'] != df['eds_final_patient_id'])
            & ((df.eds_manual.isna()) | (df.match_by != '0 = eds manual'))
            & (df['patient_id_manual'] != df['eds_final_patient_id'])
                                                ].index)

    return filtered_df
