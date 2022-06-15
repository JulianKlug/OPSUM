import pandas as pd

from preprocessing.utils import create_registry_case_identification_column


def restrict_to_patient_selection(variable_df: pd.DataFrame, patient_selection_path: str,
                                  restrict_to_event_period:bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Restricts a dataframe to only the patients that are in the patient selection file and with sampling date corresponding to the event period.
    *Exclusion criteria for cases start date of EHR sampling* (if restrict_to_event_period is True):
    - EHR sampling start date needs to at most 10 days before stroke onset (so that 14 days periods includes 72h of stroke monitoring) [when stroke onset is not available, arrival date from registry is used]
    - EHR sampling start date should be at most 7 days after reference date in registry (stroke onset or arrival date, whichever is later)

    *Exclusion criteria for individual samples*
    - Samples occurring before the day of stroke onset should be excluded
    :param variable_df:
    :param patient_selection_path:
    :param restrict_to_event_period:
    :param verbose:
    :return:
    """

    patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
    patient_selection_df['case_admission_id'] = create_registry_case_identification_column(patient_selection_df)

    restricted_to_selection_df = variable_df[
        variable_df['case_admission_id'].isin(patient_selection_df['case_admission_id'])]

    if verbose:
        print('Number of patients after selection:', len(restricted_to_selection_df['case_admission_id'].unique()))
        print('Number of patients not selected:', len(variable_df['case_admission_id'].unique()) - len(restricted_to_selection_df['case_admission_id'].unique()))
        print('Number of patients from selection that were not found:', len(patient_selection_df['case_admission_id'].unique()) - len(restricted_to_selection_df['case_admission_id'].unique()))

    if not restrict_to_event_period:
        return restricted_to_selection_df
    else:
        # restrict case_admissions to those sampled within bounds of event [event date -10 days, event date + 7 days]
        datatime_format = '%d.%m.%Y %H:%M'
        # find first sample_date for each case_admission id
        temp_df = restricted_to_selection_df.copy()
        temp_df['sample_date_dt'] = pd.to_datetime(temp_df['sample_date'], format=datatime_format)
        first_sample_date = temp_df.groupby('case_admission_id').sample_date_dt.min()
        temp_df.drop(columns=['sample_date_dt'], inplace=True)
        first_sample_date = first_sample_date.reset_index(level=0)
        first_sample_date.rename(columns={'sample_date_dt': 'first_sample_date'}, inplace=True)
        first_sample_date = first_sample_date.merge(patient_selection_df, on='case_admission_id', how='left')
        # LOWER BOUND: Applying lower bound of EHR sampling
        # set stroke onset date as reference (or Arrival date if no stroke onset date is available)
        first_sample_date['event_start_date_reference'] = first_sample_date['Stroke onset date'].fillna(
            first_sample_date['Arrival at hospital'])
        first_sample_date['registry_onset_to_first_sample_date_days'] = (
                pd.to_datetime(first_sample_date['first_sample_date'], format=datatime_format) - pd.to_datetime(
            first_sample_date['event_start_date_reference'], format='%Y%m%d')).dt.days
        cid_sampled_too_early = first_sample_date[first_sample_date['registry_onset_to_first_sample_date_days'] < -10][
            'case_admission_id'].unique()

        # UPPER BOUND: Applying upper bound of EHR sampling
        # set end of reference period to stroke onset or arrival at hospital, whichever is later
        first_sample_date['delta_onset_arrival'] = (
                pd.to_datetime(first_sample_date['Stroke onset date'], format='%Y%m%d') - pd.to_datetime(
            first_sample_date['Arrival at hospital'], format='%Y%m%d')).dt.total_seconds()
        first_sample_date['sampling_start_upper_bound_reference'] = first_sample_date \
            .apply(lambda x: x['Stroke onset date'] if x['delta_onset_arrival'] > 0 else x['Arrival at hospital'], axis=1)
        first_sample_date['registry_upper_bound_to_first_sample_date_days'] = (
                pd.to_datetime(first_sample_date['first_sample_date'], format=datatime_format) - pd.to_datetime(
            first_sample_date['sampling_start_upper_bound_reference'], format='%Y%m%d')).dt.days
        cid_sampled_too_late = first_sample_date[first_sample_date['registry_upper_bound_to_first_sample_date_days'] > 7][
            'case_admission_id'].unique()

        # drop cid from temp_df if in cid_sampled_too_early or cid_sampled_too_late
        temp_df = temp_df[~temp_df['case_admission_id'].isin(cid_sampled_too_early)]
        temp_df = temp_df[~temp_df['case_admission_id'].isin(cid_sampled_too_late)]

        if verbose:
            print(f'Dropping {len(cid_sampled_too_early)} cases due to EHR sampling start date too early')
            print(f'Dropping {len(cid_sampled_too_late)} cases due to EHR sampling start date too late')
            print('Number of patients after selection:', len(temp_df['case_admission_id'].unique()))

        # Samples occurring before stroke onset should be excluded
        initial_columns = temp_df.columns
        # create duplicate of Arrival at hospital to avoid confusion with stroke registry data
        patient_selection_df['arrival_at_hospital_date'] = patient_selection_df['Arrival at hospital']
        temp_df = temp_df.merge(patient_selection_df[['case_admission_id', 'Stroke onset date', 'arrival_at_hospital_date']],
                                on='case_admission_id', how='left')
        temp_df['event_start_date_reference'] = temp_df['Stroke onset date'].fillna(temp_df['arrival_at_hospital_date'])
        temp_df['delta_sample_date_stroke_onset'] = (
                pd.to_datetime(temp_df['sample_date'], format=datatime_format) - pd.to_datetime(
            temp_df['event_start_date_reference'], format='%Y%m%d')).dt.days
        # drop rows with delta_sample_date_stroke_onset < 0
        temp_df = temp_df[temp_df['delta_sample_date_stroke_onset'] >= 0]
        temp_df = temp_df[initial_columns]

        return temp_df

