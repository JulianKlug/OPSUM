import pandas as pd
import numpy as np

VARIABLES_TO_DOWN_SAMPLE = [
    'NIHSS',
    'oxygen_saturation',
    'systolic_blood_pressure',
    'diastolic_blood_pressure',
    'mean_blood_pressure',
    'heart_rate',
    'respiratory_rate'
]


def resample_to_hourly_features(df: pd.DataFrame, verbose=True,
                                variables_to_down_sample: list = VARIABLES_TO_DOWN_SAMPLE) -> pd.DataFrame:
    """
    Resample the dataframe to hourly values.
    Variables in variables_to_down_sample are down sampled to hourly values (median, max, min). For all other variables, if more than one sample per hour is present, take the median
    In this process only the following columns are kept: 'case_admission_id', 'relative_sample_date_hourly_cat','sample_label','source','value'
    :param df: dataframe to resample (must be purely numerical, ie. categorical must have been encoded first)
    :param verbose: if True, print the intermediate steps
    :param variables_to_down_sample: list of variables to down sample (default: VARIABLES_TO_DOWN_SAMPLE)
    :return: resampled dataframe
    """

    df['relative_sample_date_hourly_cat'] = np.floor(df['relative_sample_date'])

    if verbose:
        print('The following variables will be downsampled to median, max, min per hour:')
        print(variables_to_down_sample)

    # resampling demands keeping only minimal columns
    columns_to_keep = [
        'case_admission_id',
        'relative_sample_date_hourly_cat',
        'sample_label',
        'source',
        'value'
    ]
    resampled_df = df[columns_to_keep].copy()

    if verbose:
        print('The following columns will be disregarded:')
        print(df.columns.difference(columns_to_keep))

    for variable in variables_to_down_sample:
        if verbose:
            print(f"Downsampling: {variable}")
        # extract median
        median_variable_df = df[
            df.sample_label == variable].groupby([
            'case_admission_id',
            'relative_sample_date_hourly_cat'
        ])['value'].median().reset_index()
        median_variable_df['sample_label'] = f'median_{variable}'
        # extract max
        max_variable_df = df[
            df.sample_label == variable].groupby([
            'case_admission_id',
            'relative_sample_date_hourly_cat'
        ])['value'].max().reset_index()
        max_variable_df['sample_label'] = f'max_{variable}'
        # extract min
        min_variable_df = df[
            df.sample_label == variable].groupby([
            'case_admission_id',
            'relative_sample_date_hourly_cat'
        ])['value'].min().reset_index()
        min_variable_df['sample_label'] = f'min_{variable}'
        temp_df = pd.concat([median_variable_df, max_variable_df, min_variable_df], axis=0)
        # all variables to downsample are from EHR
        temp_df['source'] = 'EHR'
        resampled_df = resampled_df.append(
            temp_df)
        # drop all rows of sample label variable
        resampled_df = \
            resampled_df[
                resampled_df.sample_label != variable]

    all_other_vars = [sample_label for sample_label in
                      df.sample_label.unique()
                      if sample_label not in variables_to_down_sample]

    # for all other variables, when more than one sample per hour is present, take the median
    for variable in all_other_vars:
        median_variable_df = df[
            df.sample_label == variable].groupby([
            'case_admission_id',
            'relative_sample_date_hourly_cat'
        ])['value'].median().reset_index()
        median_variable_df['sample_label'] = f'{variable}'

        median_variable_df['source'] = df[
            df.sample_label == variable]['source'].mode()[0]
        # Using mode as source leads to errors for the following labels: LDL cholesterol calcule, weight, cholesterol total
        # (they are wrongly labeled as coming from the storke registry

        # drop old rows of the variable
        resampled_df = \
            resampled_df[
                resampled_df.sample_label != variable]
        resampled_df = resampled_df.append(
            median_variable_df)

    return resampled_df
