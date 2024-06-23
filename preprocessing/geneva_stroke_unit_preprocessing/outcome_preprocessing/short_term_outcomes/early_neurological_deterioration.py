import numpy as np
import pandas as pd


def early_neurological_deterioration(df, require_min_repeats=False, min_delta=4):
    """
    Detects early neurological deterioration based on NIHSS scores from an input DataFrame.

    Args:
        df: A pandas DataFrame containing the necessary data. Must contain the following columns:
            - sample_label: The label of the sample.
            - source: The source of the sample.
            - sample_date: The date of the sample.
            - value: The value of the sample.
            Example: restricted_feature_df
        require_min_repeats (bool): Whether to require a minimum number of repeated measurements for detection.
        min_delta (int): The minimum difference in NIHSS scores to consider as deterioration.

    Returns:
        pandas DataFrame: A subset of the input DataFrame with early neurological deterioration detected.

    Raises:
        ValueError: If the input DataFrame is empty or does not contain the required columns.
    """

    nihss_df = df[
        (df['sample_label'] == 'NIHSS') & (df['source'] == 'EHR')]

    end_nihss_df = nihss_df.groupby('case_admission_id').apply(detect_end_events,
                                                                          require_min_repeats=require_min_repeats,
                                                                          min_delta=min_delta)
    # undo groupby
    end_nihss_df.reset_index(drop=True, inplace=True)

    end_nihss_df = end_nihss_df[end_nihss_df.end]
    end_nihss_df['relative_sample_date_hourly_cat'] = np.floor(
        end_nihss_df['relative_sample_date'])

    return end_nihss_df


def detect_end_events(temp, require_min_repeats=False, min_delta=4):
    temp['sample_date'] = pd.to_datetime(temp['sample_date'], format='%d.%m.%Y %H:%M')
    temp['value'] = temp['value'].astype(float)
    temp.sort_values('sample_date', inplace=True)

    if require_min_repeats:
        # for a given patient, compute minimum NIHSS confirmed by at least 2 consecutive measurements
        temp['same_as_previous'] = (temp['value'].shift(1) == temp['value']).astype(int)
        temp['score_with_min_1_repeat'] = temp['value']
        temp.loc[temp['same_as_previous'] == 0, 'score_with_min_1_repeat'] = np.nan
        # for every row, compute min in rows with same_as_previous up to this row
        temp['min_nihss'] = temp['score_with_min_1_repeat'].expanding().min()
        drop_cols = ['same_as_previous', 'score_with_min_1_repeat', 'min_nihss', 'delta_to_min']
    else:
        temp['min_nihss'] = temp['value'].expanding().min()
        drop_cols = ['min_nihss', 'delta_to_min']

    temp['delta_to_min'] = temp['value'] - temp['min_nihss']
    temp['end'] = temp['delta_to_min'] >= min_delta

    # only retain first end event
    temp['n_end'] = temp['end'].cumsum()
    temp.loc[temp['n_end'] > 1, 'end'] = False

    drop_cols.append('n_end')
    temp.drop(drop_cols, axis=1, inplace=True)
    return temp