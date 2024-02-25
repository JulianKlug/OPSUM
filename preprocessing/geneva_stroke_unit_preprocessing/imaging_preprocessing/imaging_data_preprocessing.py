import pandas as pd
from preprocessing.geneva_stroke_unit_preprocessing.patient_selection.restrict_to_patient_selection import \
    restrict_to_patient_selection


def preprocess_imaging_data(imaging_data_path: str, patient_selection_path: str):
    """
    Preprocesses pre-extracted perfusion imaging data.

    Included variables:
    - T10 (Tmax > 10s)
    - T8 (Tmax > 8s)
    - T6 (Tmax > 6s)
    - T4 (Tmax > 4s)
    - CBF (CBF < 30%)

    Args:
        imaging_data_path (str): The path to the imaging data file.
        patient_selection_path (str): The path to the patient selection file.

    Returns:
        pandas.DataFrame: The preprocessed imaging data with selected variables melted.
    """
    imaging_data_df = pd.read_excel(imaging_data_path)
    imaging_data_df = restrict_to_patient_selection(imaging_data_df, patient_selection_path,
                                                    restrict_to_event_period=False)

    target_datatime_format = '%d.%m.%Y %H:%M'
    imaging_data_df['imaging_full_date'] = imaging_data_df['1st brain imaging date'].astype(str) + ' ' + \
                                           imaging_data_df['1st brain imaging time'].astype(str)
    imaging_data_df['sample_date'] = pd.to_datetime(imaging_data_df['imaging_full_date'], format='%Y%m%d %H:%M')
    # convert to target format
    imaging_data_df['sample_date'] = imaging_data_df['sample_date'].dt.strftime(target_datatime_format)

    # Variables to include: 'T10' (Tmax > 10), 'T8' (Tmax > 8), 'T6' (Tmax > 6), 'T4' (Tmax > 4), 'CBF' (CBF < 30%)
    # - CTP_artefacted (presence of artefacts in CTP) --> not kept for now, because to little data
    selected_imaging_data_df = imaging_data_df[['case_admission_id', 'sample_date', 'T10', 'T8', 'T6', 'T4', 'CBF']]

    return selected_imaging_data_df.melt(id_vars=['case_admission_id', 'sample_date'],
                                         value_vars=['T10', 'T8', 'T6', 'T4', 'CBF'], var_name='sample_label',
                                         value_name='value')
