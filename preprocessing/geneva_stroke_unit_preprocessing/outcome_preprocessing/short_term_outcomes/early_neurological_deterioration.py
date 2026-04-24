import numpy as np
import pandas as pd


def early_neurological_deterioration(df, require_min_repeats=False, min_delta=4, keep_multiple_events=True):
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
        keep_multiple_events (bool): If True, allows detection of multiple deterioration events by resetting the baseline after each event.

    Returns:
        pandas DataFrame: A subset of the input DataFrame with early neurological deterioration detected.

    Raises:
        ValueError: If the input DataFrame is empty or does not contain the required columns.
    """

    nihss_df = df[
        (df['sample_label'] == 'NIHSS') & (df['source'] == 'EHR')]

    end_nihss_df = nihss_df.groupby('case_admission_id').apply(detect_end_events,
                                                                          require_min_repeats=require_min_repeats,
                                                                          min_delta=min_delta,
                                                                            keep_multiple_events=keep_multiple_events)
    # undo groupby
    end_nihss_df.reset_index(drop=True, inplace=True)

    end_nihss_df = end_nihss_df[end_nihss_df.end]
    end_nihss_df['relative_sample_date_hourly_cat'] = np.floor(
        end_nihss_df['relative_sample_date'])

    return end_nihss_df


def detect_end_events(temp, require_min_repeats=False, min_delta=4, keep_multiple_events=True):
    """
    Detect neurological deterioration events based on NIHSS increases.
    
    Parameters:
    - temp: DataFrame with NIHSS measurements
    - require_min_repeats: If True, only consider NIHSS values confirmed by at least 2 consecutive measurements
    - min_delta: Minimum increase in NIHSS score to be considered deterioration
    - keep_multiple_events: If True, detect multiple deterioration events by resetting baseline after each event
    
    Returns:
    - DataFrame with detected events
    """
    temp = temp.copy()
    temp['sample_date'] = pd.to_datetime(temp['sample_date'], format='%d.%m.%Y %H:%M')
    temp['value'] = temp['value'].astype(float)
    temp.sort_values('sample_date', inplace=True)
    
    # Initialize columns for tracking
    temp['min_nihss'] = np.nan
    temp['delta_to_min'] = np.nan
    temp['end'] = False
    
    if not keep_multiple_events:
        # Original behavior - just use expanding min and mark first event only
        if require_min_repeats:
            # For a given patient, compute minimum NIHSS confirmed by at least 2 consecutive measurements
            temp['same_as_previous'] = (temp['value'].shift(1) == temp['value']).astype(int)
            temp['score_with_min_1_repeat'] = temp['value']
            temp.loc[temp['same_as_previous'] == 0, 'score_with_min_1_repeat'] = np.nan
            temp['min_nihss'] = temp['score_with_min_1_repeat'].expanding().min()
        else:
            temp['min_nihss'] = temp['value'].expanding().min()
            
        temp['delta_to_min'] = temp['value'] - temp['min_nihss']
        temp['end'] = temp['delta_to_min'] >= min_delta
        
        # Only retain first end event
        temp['n_end'] = temp['end'].cumsum()
        temp.loc[temp['n_end'] > 1, 'end'] = False
        drop_cols = ['n_end']
        
    else:
        # New behavior - reset minimum after each event
        current_min = np.inf
        last_event_idx = -1
        
        if require_min_repeats:
            # Mark scores that are repeated at least once
            temp['same_as_previous'] = (temp['value'].shift(1) == temp['value']).astype(int)
            temp['valid_score'] = (temp['same_as_previous'] == 1) | (temp['same_as_previous'].shift(-1) == 1)
        else:
            temp['valid_score'] = True
            
        # Process rows sequentially to detect multiple events
        for i, row in temp.iterrows():
            if not pd.isna(row['value']):
                if require_min_repeats and not row['valid_score']:
                    # Skip this measurement as it's not confirmed
                    temp.at[i, 'min_nihss'] = current_min
                    continue
                    
                # Update minimum if this is a new minimum since last event
                if row['value'] < current_min:
                    current_min = row['value']
                    
                # Calculate delta and check for event
                temp.at[i, 'min_nihss'] = current_min
                temp.at[i, 'delta_to_min'] = row['value'] - current_min
                
                if temp.at[i, 'delta_to_min'] >= min_delta:
                    # This is a deterioration event
                    temp.at[i, 'end'] = True
                    # Reset the minimum to this new value for subsequent measurements
                    current_min = row['value']
        
        if require_min_repeats:
            drop_cols = ['same_as_previous', 'valid_score']
        else:
            drop_cols = []
    
    # Drop temporary columns
    if 'drop_cols' in locals() and len(drop_cols) > 0:
        temp.drop(drop_cols, axis=1, inplace=True)
        
    return temp