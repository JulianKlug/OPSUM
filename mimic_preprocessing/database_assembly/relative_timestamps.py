import os
import numpy as np
import pandas as pd


def transform_to_relative_timestamps(df: pd.DataFrame, drop_old_columns: bool = True,
                                     restrict_to_time_range: bool = False, desired_time_range: int = 72,
                                     enforce_min_time_range: bool = False, min_time_range: int = 12,
                                     log_dir: str = '') -> pd.DataFrame:
    """
    Transform the datetime column to relative timestamps in hours from first measurement.
    - Restriction to time range can be applied (all data after time limit is dropped)
    - Enforcement of minimum time range can be applied (all cases with a sampling range < min_time_range are dropped)

    :param df: Dataframe with sample_date column
    :param drop_old_columns: Drop columns with old timestamps
    :param restrict_to_time_range: bool, if true, restrict to time range
    :param desired_time_range: int, if restrict_to_time_range is True, restrict to this upper limit in hours, default is 72 hours
    :param enforce_min_time_range: bool, if true, enforce minimum time range
    :param min_time_range: int, if enforce_min_time_range is True, enforce minimum time range to this value in hours, default is 12 hours
    :param log_dir: str, path to log directory
    :return: Dataframe with relative timestamps
    """
    datatime_format = '%Y-%m-%d %H:%M:%S'

    df['sample_date'] = pd.to_datetime(df['sample_date'], format=datatime_format)

    # SET REFERENCE time point 0 for each case_admission id
    # - default reference as start: first sample date of EHR

    # Find first sample date of EHR
    first_ehr_sample_date = df[(df.source == 'EHR')] \
        .groupby('case_admission_id').sample_date.min().reset_index(level=0)
    first_ehr_sample_date.rename(columns={'sample_date': 'first_ehr_sample_date'}, inplace=True)

    # Find first sample date of admission notes
    first_notes_sample_date = df[df.source == 'notes'].groupby(
        'case_admission_id').sample_date.min().reset_index(level=0)
    first_notes_sample_date.rename(columns={'sample_date': 'first_notes_sample_date'}, inplace=True)

    merged_first_sample_dates_df = first_ehr_sample_date.merge(first_notes_sample_date, on='case_admission_id')

    merged_first_sample_dates_df['delta_first_sample_date_h'] = (
                        merged_first_sample_dates_df[
                            'first_ehr_sample_date']
                        - merged_first_sample_dates_df[
                            'first_notes_sample_date']) / np.timedelta64(1, 'h')

    # verify that notes are not too distant from EHR
    if any(merged_first_sample_dates_df['delta_first_sample_date_h'].abs() > 24):
        raise Exception('Data extracted from admission notes should always occur before EHR!')

    merged_first_sample_dates_df['reference_first_sample_date'] = merged_first_sample_dates_df['first_ehr_sample_date']
    merged_first_sample_dates_df = merged_first_sample_dates_df[['case_admission_id', 'reference_first_sample_date']]

    # TRANSFORM TO RELATIVE TIMESTAMPS
    df['case_admission_id'] = df['case_admission_id'].astype(str)
    merged_first_sample_dates_df['case_admission_id'] = merged_first_sample_dates_df['case_admission_id'].astype(str)
    merged_first_sample_dates_df.rename(columns={'reference_first_sample_date': 'first_sample_date'}, inplace=True)
    df = df.merge(merged_first_sample_dates_df, on='case_admission_id')
    df['relative_sample_date'] = (pd.to_datetime(df['sample_date'], format=datatime_format)
                                  - pd.to_datetime(df['first_sample_date'], format=datatime_format)) \
                                 / np.timedelta64(1, 'h')

    # ensure notes data at timepoint 0
    df.loc[df.source == 'notes', 'relative_sample_date'] = 0.0

    # FILTER OUT UNWARRANTED DATA
    # exclude samples with relative_sample_date < 0
    df = df[df['relative_sample_date'] >= 0]

    if restrict_to_time_range:
        df = df[df['relative_sample_date'] <= desired_time_range]

    if enforce_min_time_range:
        max_sampling_dates = df[df.source != 'notes'].groupby(
            'case_admission_id').relative_sample_date.max().reset_index()
        cid_with_short_range = max_sampling_dates[
            max_sampling_dates.relative_sample_date < min_time_range].case_admission_id.unique()
        print(f'Excluding {len(cid_with_short_range)} cases with a sampling range < {min_time_range} hours')
        df = df[~df['case_admission_id'].isin(cid_with_short_range)]
        if log_dir != '':
            pd.DataFrame(cid_with_short_range, columns=['case_admission_id']) \
                .to_csv(os.path.join(log_dir, 'excluded_patients_with_sampling_range_too_short.tsv'), index=False)

    if drop_old_columns:
        df.drop(['sample_date', 'first_sample_date'], axis=1, inplace=True)

    return df
