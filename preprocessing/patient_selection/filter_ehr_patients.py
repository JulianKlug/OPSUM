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

    filtered_df = df.drop(
                    df[(df['patient_id'] != df['eds_final_patient_id'])
                    & ((df.eds_manual.isna()) | (df.match_by != '0 = eds manual'))]
                    .index)

    # TODO: update match_by strategies with new extraction
    # TODO: check if patient_id_manual is used for matching

    return filtered_df
