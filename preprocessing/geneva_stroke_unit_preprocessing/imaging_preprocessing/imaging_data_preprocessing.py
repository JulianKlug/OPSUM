import pandas as pd
from preprocessing.geneva_stroke_unit_preprocessing.patient_selection.restrict_to_patient_selection import \
    restrict_to_patient_selection


def preprocess_imaging_data(imaging_data_path: str, patient_selection_path: str, restricted_stroke_registry_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Preprocesses pre-extracted perfusion imaging data.

    Args:
        imaging_data_path (str): The path to the imaging data file.
        patient_selection_path (str): The path to the patient selection file.
        restricted_stroke_registry_df (pandas.DataFrame, optional): DataFrame containing restricted stroke registry data (with preprocessed admission imaging data)
            Defaults to None.
    Returns:
        pandas.DataFrame: The preprocessed imaging data with selected variables melted.
    """
    imaging_data_df = pd.read_csv(imaging_data_path)
    imaging_data_df = restrict_to_patient_selection(imaging_data_df, patient_selection_path,
                                                    restrict_to_event_period=False)

    target_datatime_format = '%d.%m.%Y %H:%M'
    imaging_data_df['imaging_full_date'] = imaging_data_df['acquisition_date'].astype(str) + ' ' + \
                                           imaging_data_df['acquisition_time'].astype(str)
    # convert to datetime - format is either '%Y%m%d %H%M%S' or '%Y%m%d %H:%M'
    imaging_data_df['sample_date'] = pd.to_datetime(imaging_data_df['imaging_full_date'], format='%Y%m%d %H%M%S', errors='coerce')
    # if there are NaT values, try the other format
    imaging_data_df.loc[imaging_data_df['sample_date'].isna(), 'sample_date'] = pd.to_datetime(
        imaging_data_df.loc[imaging_data_df['sample_date'].isna(), 'imaging_full_date'], format='%Y%m%d %H:%M', errors='coerce')
    
    # raise an error if there are still NaT values - show problematic rows
    if imaging_data_df['sample_date'].isna().sum() > 0:
        raise ValueError(f"There are NaT values in 'sample_date' after trying both date formats. Problematic rows:\n{imaging_data_df[imaging_data_df['sample_date'].isna()][['imaging_full_date']]}")

    # convert to target format
    imaging_data_df['sample_date'] = imaging_data_df['sample_date'].dt.strftime(target_datatime_format)

    # drop original columns
    imaging_data_df = imaging_data_df.drop(columns=['acquisition_date', 'acquisition_time', 'imaging_full_date', 'patient_id'])

    # rename parameter_name to sample_label
    imaging_data_df = imaging_data_df.rename(columns={'parameter_name': 'sample_label'})

    # add source column
    imaging_data_df['source'] = 'EHR'

    # add analysed admission imaging data from stroke registry if provided
    if restricted_stroke_registry_df is not None:
        registry_imaging_df = restricted_stroke_registry_df[['case_admission_id', 'Acute perf. imaging result',
                                                              '1st vascular imaging result',
                                                              '1st brain imaging date',
                                                              '1st brain imaging time', 
                                                              'sample_date']]
        # manually encode the imaging results
        registry_imaging_df['hypoperfusion_with_mismatch'] = registry_imaging_df['Acute perf. imaging result'].apply(
            lambda x: 1 if x == 'Focal hypoperfusion with mismatch' else (0 if pd.notna(x) else None))
        registry_imaging_df['hypoperfusion_without_mismatch'] = registry_imaging_df['Acute perf. imaging result'].apply(
            lambda x: 1 if x == 'Focal hypoperfusion without mismatch' else (0 if pd.notna(x) else None))
        
        registry_imaging_df['vascular_occlusion'] = registry_imaging_df['1st vascular imaging result'].apply(
            lambda x: 1 if x == 'Occlusion in suspected ischemic territory' else (0 if pd.notna(x) else None))
        registry_imaging_df['vascular_stenosis_over_50p'] = registry_imaging_df['1st vascular imaging result'].apply(
            lambda x: 1 if x == 'Stenosis 50-99% in suspected ischemic territory' else (0 if pd.notna(x) else None))
        
        # create sample_date from registry imaging date and time
        # if '.0' is present in date or time, remove it
        registry_imaging_df['1st brain imaging date'] = registry_imaging_df['1st brain imaging date'].astype(str).str.replace('.0', '', regex=False)
        registry_imaging_df['1st brain imaging time'] = registry_imaging_df['1st brain imaging time'].astype(str).str.replace('.0', '', regex=False)

        registry_imaging_df['imaging_full_date'] = registry_imaging_df['1st brain imaging date'].astype(str) + ' ' + \
                                                   registry_imaging_df['1st brain imaging time'].astype(str)
        registry_imaging_df['registry_imaging_sample_date'] = pd.to_datetime(registry_imaging_df['imaging_full_date'], format='%Y%m%d %H:%M', errors='coerce')
        # convert to target format
        registry_imaging_df['registry_imaging_sample_date'] = registry_imaging_df['registry_imaging_sample_date'].dt.strftime(target_datatime_format)
        # replace 'NaT' strings with actual NaN
        registry_imaging_df['registry_imaging_sample_date'] = registry_imaging_df['registry_imaging_sample_date'].replace('NaT', pd.NA)

        # fillnan in registry_imaging_sample_date with sample_date
        registry_imaging_df['registry_imaging_sample_date'] = registry_imaging_df['registry_imaging_sample_date'].fillna(registry_imaging_df['sample_date'])
        
        value_vars = ['hypoperfusion_with_mismatch', 'hypoperfusion_without_mismatch',
                           'vascular_occlusion', 'vascular_stenosis_over_50p']
        
        # drop original columns
        registry_imaging_df = registry_imaging_df.drop(columns=['Acute perf. imaging result', '1st vascular imaging result', 'sample_date'])

        registry_imaging_df = registry_imaging_df.melt(id_vars=['case_admission_id', 'registry_imaging_sample_date'],
                                                       value_vars=value_vars, var_name='sample_label',
                                                       value_name='value')
        # rename registry_imaging_sample_date to sample_date
        registry_imaging_df = registry_imaging_df.rename(columns={'registry_imaging_sample_date': 'sample_date'})

        # add source column
        registry_imaging_df['source'] = 'stroke_registry'

        # concatenate imaging data with registry imaging data
        imaging_data_df = pd.concat([imaging_data_df, registry_imaging_df], ignore_index=True)

    
    # drop rows in where 'value' is nan
    imaging_data_df = imaging_data_df.dropna(subset=['value'])

    return imaging_data_df
