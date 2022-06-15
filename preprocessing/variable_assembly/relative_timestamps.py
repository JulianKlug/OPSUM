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
    datatime_format = '%d.%m.%Y %H:%M'
    df['sample_date'] = pd.to_datetime(df['sample_date'], format=datatime_format)

    # SET REFRENCE time point 0 for each case_admission id
    # - default reference as start: first sample date of EHR
    # -- Exclude registry data from first sample date estimation
    #    Rationale:
    #       - exact registry data is more error-prone & EHR data should be filtered at this stage)
    #       - this allows for the case in which patient is admitted a few days after initial evaluation (limited to +7 days
    #          in restrict_to_patient_selection) and avoids a gap between registry and EHR data
    # -- FIO2 should not be used as reference, as its timing is inferred from registry data
    # - when first sample date of EHR is more than 1 day before first sample date of stroke registry:
    # -- if first sample of NIHSS is before first sample of stroke registry -> reference is first sample of EHR
    # -- if first sample of NIHSS is after first sample of stroke registry -> reference is first sample of stroke registry
    #     -> remove samples occurring before reference

    # Find first sample date of EHR
    first_ehr_sample_date = df[(df.source == 'EHR') & (df.sample_label != 'FIO2')] \
        .groupby('case_admission_id').sample_date.min().reset_index(level=0)
    first_ehr_sample_date.rename(columns={'sample_date': 'first_ehr_sample_date'}, inplace=True)

    # Find first sample date of stroke registry
    first_registry_sample_date = df[df.source == 'stroke_registry'].groupby(
        'case_admission_id').sample_date.min().reset_index(level=0)
    first_registry_sample_date.rename(columns={'sample_date': 'first_registry_sample_date'}, inplace=True)

    # Find first sample date of NIHSS for each case_admission id, if NIHSS is not available, use nan
    first_NIHSS_sample_date = df[(df.sample_label == 'NIHSS') & (df.source == 'EHR')].groupby(
        'case_admission_id').sample_date.min().reset_index(level=0)
    # for all cases with missing NIHSS data, use nan
    missing_NIHSS_sample_date = pd.DataFrame(set(df.case_admission_id.unique())
                                             .difference(set(first_NIHSS_sample_date.case_admission_id.unique())),
                                             columns=['case_admission_id'])
    missing_NIHSS_sample_date['sample_date'] = np.nan
    first_NIHSS_sample_date = first_NIHSS_sample_date.append(missing_NIHSS_sample_date)
    first_NIHSS_sample_date.rename(columns={'sample_date': 'first_NIHSS_sample_date'}, inplace=True)

    merged_first_sample_dates_df = first_ehr_sample_date.merge(first_registry_sample_date, on='case_admission_id')
    merged_first_sample_dates_df = merged_first_sample_dates_df.merge(first_NIHSS_sample_date, on='case_admission_id')

    merged_first_sample_dates_df['delta_first_sample_date_h'] = (
                        merged_first_sample_dates_df[
                            'first_ehr_sample_date']
                        - merged_first_sample_dates_df[
                            'first_registry_sample_date']) / np.timedelta64(
        1, 'h')

    merged_first_sample_dates_df['delta_first_NIHSS_to_registry_start_date_h'] = (
             merged_first_sample_dates_df[
                 'first_NIHSS_sample_date']
             - merged_first_sample_dates_df[
                 'first_registry_sample_date']) / np.timedelta64(
        1, 'h')

    def determine_reference_time_point(row):
        # default is first sample date of EHR
        # (NB: samples occurring before stroke onset should be dropped by now)
        if row['delta_first_sample_date_h'] > -24:
            return row['first_ehr_sample_date']
        # except in cases where first sample date of EHR is more than 1 day before first sample date of stroke registry
        # in that case, reference is first sample date of stroke registry
        elif (row['delta_first_NIHSS_to_registry_start_date_h'] > 0) \
                and (~np.isnan(row['delta_first_NIHSS_to_registry_start_date_h'])):
            return row['first_registry_sample_date']
        # except if first sample date of NIHSS is before first sample date of stroke registry, then it is likely that
        # the patient was admitted and evaluated before the registry admission date (and NIHSS is done after other date in the EHR)
        # if NIHSS sample date is nan, use first sample date of EHR
        else:
            # ALTERNATIVE would be first sample date of EHR again
            return row['first_ehr_sample_date']

    merged_first_sample_dates_df['reference_first_sample_date'] = merged_first_sample_dates_df.apply(
        lambda row: determine_reference_time_point(row), axis=1)

    merged_first_sample_dates_df = merged_first_sample_dates_df[['case_admission_id', 'reference_first_sample_date']]

    # TRANSFORM TO RELATIVE TIMESTAMPS
    df['case_admission_id'] = df['case_admission_id'].astype(str)
    merged_first_sample_dates_df['case_admission_id'] = merged_first_sample_dates_df['case_admission_id'].astype(str)
    merged_first_sample_dates_df.rename(columns={'reference_first_sample_date': 'first_sample_date'}, inplace=True)
    df = df.merge(merged_first_sample_dates_df, on='case_admission_id')
    df['relative_sample_date'] = (pd.to_datetime(df['sample_date'], format=datatime_format)
                                  - pd.to_datetime(df['first_sample_date'], format=datatime_format)) / np.timedelta64(1,
                                                                                                                      'h')

    # ensure stroke registry data & inferred FIO2 is inserted in to timepoint 0
    df.loc[df.source == 'stroke_registry', 'relative_sample_date'] = 0.0
    df.loc[(df.sample_label == 'FIO2') & (df.relative_sample_date < 0), 'relative_sample_date'] = 0.0

    # FILTER OUT UNWARRANTED DATA
    # exclude samples with relative_sample_date < 0
    df = df[df['relative_sample_date'] >= 0]

    if enforce_min_time_range:
        max_sampling_dates = df[df.source != 'stroke_registry'].groupby(
            'case_admission_id').relative_sample_date.max().reset_index()
        cid_with_short_range = max_sampling_dates[
            max_sampling_dates.relative_sample_date < min_time_range].case_admission_id.unique()
        print(f'Excluding {len(cid_with_short_range)} cases with a sampling range < {min_time_range} hours')
        df = df[~df['case_admission_id'].isin(cid_with_short_range)]
        if log_dir != '':
            pd.DataFrame(cid_with_short_range, columns=['case_admission_id']) \
                .to_csv(os.path.join(log_dir, 'excluded_patients_with_sampling_range_too_short.tsv'), index=False,
                        sep='\t')

    if drop_old_columns:
        df.drop(['sample_date', 'first_sample_date'], axis=1, inplace=True)

    if restrict_to_time_range:
        df = df[df['relative_sample_date'] <= desired_time_range]

    return df
