import pandas as pd


def filter_ehr_patients(df) -> pd.DataFrame:
    """
    Filter out:
    - patients with missmatch in patient_id (from registry) and eds_final_patient_id (from EHR extraction)
        - Goal: avoid using data from another patient that was mistakenly selected during EHR extraction
        - Exception: patient that have been matched by manually completed EDS or ID (patient_id_manual, eds_manual)

    Arguments:
        df {pd.DataFrame} -- Dataframe with EHR data
    Returns:
        pd.DataFrame -- Filtered EHR dataframe
    """

    # Manual patient id should not be used for matching (manual EDS is always used in this case)
    if df[(~df.patient_id_manual.isna() & (df.match_by != '0 = eds manual'))]:
        raise ValueError('patient_id_manual might be used for matching. Please check data.')


    filtered_df = df.drop(
                    df[(df['patient_id'] != df['eds_final_patient_id'])
                    & ((df.eds_manual.isna()) | (df.match_by != '0 = eds manual'))]
                    .index)

    return filtered_df
