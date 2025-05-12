import argparse
import json
import os
import pandas as pd
import time

from prediction.utils.utils import ensure_dir
from preprocessing.geneva_stroke_unit_preprocessing.outcome_preprocessing.short_term_outcomes.short_term_outcomes_preprocessing import \
    preprocess_short_term_outcomes
from preprocessing.geneva_stroke_unit_preprocessing.outcome_preprocessing.long_term_outcomes.outcome_preprocessing import \
    preprocess_outcomes
from preprocessing.preprocessing_tools.encoding_categorical_variables.encode_categorical_variables import encode_categorical_variables
from preprocessing.preprocessing_tools.handling_missing_values.impute_missing_values import impute_missing_values
from preprocessing.preprocessing_tools.normalisation.normalisation import normalise_data
from preprocessing.preprocessing_tools.preprocessing_verification.outcome_presence_verification import outcome_presence_verification
from preprocessing.preprocessing_tools.preprocessing_verification.variable_presence_verification import variable_presence_verification
from preprocessing.preprocessing_tools.resample_to_time_bins.resample_to_hourly_features import resample_to_hourly_features
from preprocessing.geneva_stroke_unit_preprocessing.variable_assembly.variable_database_assembly import assemble_variable_database
from preprocessing.geneva_stroke_unit_preprocessing.variable_assembly.relative_timestamps import transform_to_relative_timestamps


def preprocess(ehr_data_path:str, stroke_registry_data_path:str, patient_selection_path:str, log_dir: str,
               imaging_data_path:str='', restrict_to_patients_with_imaging_data_available:bool=False,
               include_short_term_outcomes:bool=True, short_term_outcomes_config:dict={},
                verbose:bool=True, desired_time_range:int=72) -> pd.DataFrame:
    """
    Apply geneva_stroke_unit_preprocessing pipeline detailed in ./geneva_stroke_unit_preprocessing/readme.md
    :param ehr_data_path: path to EHR data
    :param stroke_registry_data_path: path to stroke registry (admission) data
    :param patient_selection_path: path to patient selection file
    :param log_dir: path to logging directory (this will receive logs of excluded patients and those that were not found)
    :param imaging_data_path: path to imaging data
    :param restrict_to_patients_with_imaging_data_available: if True, restricts the database to patients with imaging data available
    :param verbose:
    :param desired_time_range: number of hours to use for imputation
    :return: preprocessed feature Dataframe, preprocessed outcome dataframe
    """

    # 1. Restrict to patient selection (& filter out patients with no EHR data or EHR data with wrong dates)
    # 2. Preprocess EHR and stroke registry variables
    # 3. Restrict to variable selection
    # 4. Assemble database from lab/scales/ventilation/vitals + stroke registry subparts +/- imaging
    print('STARTING VARIABLE PREPROCESSING')
    feature_df = assemble_variable_database(ehr_data_path, stroke_registry_data_path, patient_selection_path,
                                            imaging_data_path=imaging_data_path,
                                            restrict_to_patients_with_imaging_data_available=restrict_to_patients_with_imaging_data_available,
                                            log_dir=log_dir, verbose=verbose)
    print(f'A. Number of patients: {feature_df.case_admission_id.nunique()}')

    # 5. Transform timestamps to relative timestamps from first measure
    # 6. Restrict to time range
    # - Exclude patients with data sampled in a time window < 12h
    # - Restrict to desired time range: 72h
    print('TRANSFORMING TO RELATIVE TIME AND RESTRICTING TIME RANGE')
    restricted_feature_df = transform_to_relative_timestamps(feature_df, drop_old_columns=False,
                                                             restrict_to_time_range=True, desired_time_range=desired_time_range,
                                                             enforce_min_time_range=True, min_time_range=12,
                                                             log_dir=log_dir)
    print(f'B. Number of patients: {restricted_feature_df.case_admission_id.nunique()}')

    # 7. Encoding categorical variables (one-hot)
    print('ENCODING CATEGORICAL VARIABLES')
    cat_encoded_restricted_feature_df = encode_categorical_variables(restricted_feature_df, verbose=verbose,
                                                                     log_dir=log_dir)
    print(f'C. Number of patients: {cat_encoded_restricted_feature_df.case_admission_id.nunique()}')


    # 8. Resampling to hourly frequency
    print('RESAMPLING TO HOURLY FREQUENCY')
    resampled_df = resample_to_hourly_features(cat_encoded_restricted_feature_df, verbose=verbose)
    print(f'D. Number of patients: {resampled_df.case_admission_id.nunique()}')

    # 9. imputation of missing values
    print('IMPUTING MISSING VALUES')
    imputed_missing_df = impute_missing_values(resampled_df, verbose=verbose, log_dir=log_dir, desired_time_range=desired_time_range)
    print(f'E. Number of patients: {imputed_missing_df.case_admission_id.nunique()}')

    # 10. normalisation
    print('APPLYING NORMALISATION')
    normalised_df = normalise_data(imputed_missing_df, verbose=verbose, log_dir=log_dir)
    print(f'F. Number of patients: {normalised_df.case_admission_id.nunique()}')

    # 11. geneva_stroke_unit_preprocessing outcomes
    preprocessed_long_term_outcomes_df = preprocess_outcomes(stroke_registry_data_path, patient_selection_path, verbose=verbose)
    if include_short_term_outcomes:
        preprocessed_short_term_outcomes_df = preprocess_short_term_outcomes(restricted_feature_df, **short_term_outcomes_config)
        preprocessed_outcomes = (preprocessed_long_term_outcomes_df, preprocessed_short_term_outcomes_df)
    else:
        preprocessed_outcomes = (preprocessed_long_term_outcomes_df)

    return normalised_df, preprocessed_outcomes


def preprocess_and_save(ehr_data_path:str, stroke_registry_data_path:str, patient_selection_path:str, output_dir:str,
                        imaging_data_path:str='', restrict_to_patients_with_imaging_data_available:bool=False,
                        include_short_term_outcomes: bool = True, short_term_outcomes_config: dict = {},
                        feature_file_prefix:str = 'preprocessed_features', outcome_file_prefix:str = 'preprocessed_outcomes', verbose:bool=True):

    # verify that all provided paths exist
    for path in [ehr_data_path, stroke_registry_data_path, patient_selection_path, imaging_data_path]:
        if path != '':
            assert os.path.exists(path), f'Path {path} does not exist'

    timestamp = time.strftime("%d%m%Y_%H%M%S")
    desired_time_range = 72
    output_dir = os.path.join(output_dir, f'gsu_{os.path.basename(ehr_data_path)}_prepro_{timestamp}')
    log_dir = os.path.join(output_dir, f'logs_{timestamp}')
    saved_args = locals()
    ensure_dir(log_dir)

    # save saved_args to log_dir
    with open(os.path.join(log_dir, 'preprocessing_arguments.json'), 'w') as fp:
        json.dump(saved_args, fp)

    preprocessed_feature_df, preprocessed_outcomes = preprocess(ehr_data_path, stroke_registry_data_path,
                                                                  patient_selection_path, log_dir=log_dir,
                                                                  imaging_data_path=imaging_data_path,
                                                                  restrict_to_patients_with_imaging_data_available = restrict_to_patients_with_imaging_data_available,
                                                                    include_short_term_outcomes=include_short_term_outcomes,
                                                                    short_term_outcomes_config=short_term_outcomes_config,
                                                                    verbose=verbose,
                                                                  desired_time_range=desired_time_range)

    features_save_path = os.path.join(output_dir, f'{feature_file_prefix}_{timestamp}.csv')
    preprocessed_feature_df.to_csv(features_save_path)

    outcomes_save_path = os.path.join(output_dir, f'{outcome_file_prefix}_{timestamp}.csv')

    if include_short_term_outcomes:
        preprocessed_long_term_outcome_df, preprocessed_short_term_outcome_df = preprocessed_outcomes
        short_term_outcomes_path = os.path.join(output_dir, f'{outcome_file_prefix}_short_term_{timestamp}.csv')
        preprocessed_short_term_outcome_df.to_csv(short_term_outcomes_path)
    else:
        preprocessed_long_term_outcome_df = preprocessed_outcomes[0]

    preprocessed_long_term_outcome_df.to_csv(outcomes_save_path)

    # verification of geneva_stroke_unit_preprocessing
    variable_presence_verification(preprocessed_feature_df, desired_time_range=desired_time_range)
    outcome_presence_verification(preprocessed_long_term_outcome_df, preprocessed_feature_df, log_dir=log_dir)



if __name__ == '__main__':
    """
    Example usage:
    python preprocessing_pipeline.py.py -e /Users/jk1/-/-/-/Extraction20220629 -r /Users/jk1/-/-/-/post_hoc_modified/stroke_registry_post_hoc_modified.xlsx 
    -p /Users/jk1/-/-/high_frequency_data_patient_selection_with_details.csv -o /Users/jk1/-/opsum_prepro_output -i /Users/jk1/.../perfusion_imaging_data/random_subset_for_imaging_extraction.xlsx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--ehr', type=str, help='EHR data path')
    parser.add_argument('-r', '--registry', type=str, help='Registry data path')
    parser.add_argument('-p', '--patients', type=str, help='Patient selection file')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-i', '--imaging', type=str, help='Imaging data path', default='')
    parser.add_argument('-ri', '--restrict_to_patients_with_imaging_data_available', action='store_true', help='Restrict to patients with imaging data available', default=False)
    parser.add_argument('-s', '--include_short_term_outcomes', action='store_true', help='Include short term outcomes', default=False)
    parser.add_argument('-end_min_repeats', '--end_min_repeats', type=bool, help='Whether to require a minimum number of repeated measurements for detection of END', default=False)
    parser.add_argument('-end_min_delta', '--end_min_delta', type=int, help='The minimum difference in NIHSS scores to consider as END', default=4)
    parser.add_argument('-end_keep_multiple_events', '--end_keep_multiple_events', type=bool, help='If True, allows detection of multiple deterioration events by resetting the baseline after each event', default=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose', default=False)

    args = parser.parse_args()

    preprocess_and_save(args.ehr, args.registry, args.patients, args.output_dir,
                        imaging_data_path=args.imaging,
                        restrict_to_patients_with_imaging_data_available=args.restrict_to_patients_with_imaging_data_available,
                        include_short_term_outcomes=args.include_short_term_outcomes,
                        short_term_outcomes_config={
                            'end_require_min_repeats': args.end_min_repeats, 
                            'end_min_delta': args.end_min_delta,
                            'end_keep_multiple_events': args.end_keep_multiple_events
                            },
                        verbose=args.verbose)
