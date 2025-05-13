from preprocessing.geneva_stroke_unit_preprocessing.outcome_preprocessing.short_term_outcomes.early_neurological_deterioration import \
    early_neurological_deterioration


def preprocess_short_term_outcomes(df, end_require_min_repeats=False, end_min_delta=4, end_keep_multiple_events=True):
    """
    Preprocess short term outcomes:
        - Early neurological deterioration

    Args:
        df: A pandas DataFrame containing the necessary data. Must contain the following columns:
            - sample_label: The label of the sample.
            - source: The source of the sample.
            - sample_date: The date of the sample.
            - value: The value of the sample.
            Example: restricted_feature_df
        end_require_min_repeats (bool): Whether to require a minimum number of repeated measurements for detection of END
        end_min_delta (int): The minimum difference in NIHSS scores to consider as END.
        keep_multiple_events (bool): If True, allows detection of multiple deterioration events by resetting the baseline after each event.

    Returns:
        pandas DataFrame: A subset of the input DataFrame with short term outcomes detected.

    Raises:
        ValueError: If the input DataFrame is empty or does not contain the required columns.
    """

    end_df = early_neurological_deterioration(df, require_min_repeats=end_require_min_repeats, min_delta=end_min_delta, keep_multiple_events=end_keep_multiple_events)
    end_df['outcome_label'] = 'early_neurological_deterioration'
    # store arguments for each outcome
    end_df['outcome_args'] = f'require_min_repeats={end_require_min_repeats}, min_delta={end_min_delta}, keep_multiple_events={end_keep_multiple_events}'

    return end_df