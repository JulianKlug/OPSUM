import pandas as pd


def transform_to_relative_timestamps(df: pd.DataFrame, drop_old_columns:bool = True, restrict_to_time_range: bool = False,
                                     desired_time_range:int = 72) -> pd.DataFrame:
    """
    Transform the datetime column to relative timestamps in hours from first measurement.
    """
    datatime_format = '%d.%m.%Y %H:%M'

    # find first sample_date for each case_admission id
    df['sample_date'] = pd.to_datetime(df['sample_date'], format=datatime_format)
    first_sample_date = df.groupby('case_admission_id').sample_date.min()
    first_sample_date = first_sample_date.reset_index(level=0)

    df['case_admission_id'] = df['case_admission_id'].astype(str)
    first_sample_date['case_admission_id'] = first_sample_date['case_admission_id'].astype(str)
    first_sample_date.rename(columns={'sample_date': 'first_sample_date'}, inplace=True)
    df = df.merge(first_sample_date, on='case_admission_id')
    df['relative_sample_date'] = (pd.to_datetime(df['sample_date'], format=datatime_format)
                                  - pd.to_datetime(df['first_sample_date'], format=datatime_format))\
                                     .dt.total_seconds() / (60 * 60)
    if drop_old_columns:
        df.drop(['sample_date', 'first_sample_date'], axis=1, inplace=True)


    if restrict_to_time_range:
        df = df[df['relative_sample_date'] <= desired_time_range]

    return df