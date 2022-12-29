import argparse
import json
import os
import shutil
import time

import pandas as pd

from preprocessing.mimic_preprocessing.database_assembly.database_assembly import assemble_variable_database
from preprocessing.mimic_preprocessing.database_assembly.relative_timestamps import transform_to_relative_timestamps
from preprocessing.mimic_preprocessing.outcome_preprocessing.outcome_preprocessing import preprocess_outcomes
from prediction.utils.utils import ensure_dir
from preprocessing.preprocessing_tools.resample_to_time_bins.resample_to_hourly_features import resample_to_hourly_features
from preprocessing.preprocessing_tools.encoding_categorical_variables.encode_categorical_variables import encode_categorical_variables
from preprocessing.preprocessing_tools.handling_missing_values.impute_missing_values import impute_missing_values
from preprocessing.preprocessing_tools.normalisation.normalisation import normalise_data
from preprocessing.preprocessing_tools.preprocessing_verification.outcome_presence_verification import outcome_presence_verification
from preprocessing.preprocessing_tools.preprocessing_verification.variable_presence_verification import variable_presence_verification


def preprocess(extracted_tables_path: str, admission_notes_data_path: str,
               reference_population_imputation_path: str,
                reference_population_normalisation_parameters_path: str,
               preproccessed_monitoring_data_path: str = '',
                mimic_admission_nihss_db_path: str = '',
               log_dir: str = '', verbose:bool=True, desired_time_range:int=72) -> pd.DataFrame:
    """
    Apply geneva_stroke_unit_preprocessing pipeline detailed in ./geneva_stroke_unit_preprocessing/readme.md to the MIMIC-III dataset.

    :param extracted_tables_path: Path to the folder containing the extracted tables from the MIMIC-III dataset.
    :param admission_notes_data_path: Path to the folder containing the data extracted from admission & discharge notes.
    :param reference_population_imputation_path: Path to the file containing the reference population statistics. This will be used for imputation of missing values (> 2/3 of subjects)
    :param reference_population_normalisation_parameters_path: Path to the file containing the reference population normalisation parameters. This will be used to normalise this dataset
    :param preproccessed_monitoring_data_path: Path to the folder containing the preprocessed monitoring data.
    :param mimic_admission_nihss_db_path: Path to the folder containing the MIMIC admission nihss database.
    :param log_dir: path to logging directory (this will receive logs of excluded patients and those that were not found)
    :param verbose:
    :param desired_time_range: number of hours to use for imputation
    :return: preprocessed feature Dataframe, preprocessed outcome dataframe
    """

    # 1. Restrict to patient selection (& filter out patients with no data from admission/discharge notes)
    # 2. Preprocess EHR and notes variables
    # 3. Restrict to variable selection
    # 4. Assemble database from subparts
    print('STARTING VARIABLE PREPROCESSING')
    feature_df = assemble_variable_database(extracted_tables_path, admission_notes_data_path,
                                            preproccessed_monitoring_data_path, mimic_admission_nihss_db_path,
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
    imputed_missing_df = impute_missing_values(resampled_df, reference_population_imputation_path=reference_population_imputation_path,
                                               verbose=verbose, log_dir=log_dir, desired_time_range=desired_time_range)
    print(f'E. Number of patients: {imputed_missing_df.case_admission_id.nunique()}')

    # 10. normalisation
    print('APPLYING NORMALISATION')
    normalised_df = normalise_data(imputed_missing_df,
                                   reference_population_normalisation_parameters_path=reference_population_normalisation_parameters_path,
                                   verbose=verbose, log_dir=log_dir)
    print(f'F. Number of patients: {normalised_df.case_admission_id.nunique()}')

    # 11. geneva_stroke_unit_preprocessing outcomes
    outcome_table_path = os.path.join(extracted_tables_path, 'outcome_df.csv')
    preprocessed_outcomes_df = preprocess_outcomes(outcome_table_path, verbose=verbose)

    return normalised_df, preprocessed_outcomes_df



def preprocess_and_save(
                extracted_tables_path: str, admission_notes_data_path: str,
                reference_population_imputation_path: str,
                reference_population_normalisation_parameters_path: str,
                output_dir: str,
                preproccessed_monitoring_data_path: str = '',
                mimic_admission_nihss_db_path: str = '',
                feature_file_prefix:str = 'preprocessed_features', outcome_file_prefix:str = 'preprocessed_outcomes',
                verbose:bool=True):

    timestamp = time.strftime("%d%m%Y_%H%M%S")
    desired_time_range = 72
    output_dir = os.path.join(output_dir, f'mimic_prepro_{timestamp}')
    log_dir = os.path.join(output_dir, f'logs_{timestamp}')
    saved_args = locals()
    ensure_dir(log_dir)

    # save saved_args to log_dir
    with open(os.path.join(log_dir, 'preprocessing_arguments.json'), 'w') as fp:
        json.dump(saved_args, fp)

    preprocessed_feature_df, preprocessed_outcome_df = preprocess(extracted_tables_path, admission_notes_data_path,
                           reference_population_imputation_path, reference_population_normalisation_parameters_path,
                           preproccessed_monitoring_data_path, mimic_admission_nihss_db_path,
                                                                  log_dir=log_dir, verbose=verbose,
                                                                  desired_time_range=desired_time_range)

    features_save_path = os.path.join(output_dir, f'{feature_file_prefix}_{timestamp}.csv')
    outcomes_save_path = os.path.join(output_dir, f'{outcome_file_prefix}_{timestamp}.csv')

    preprocessed_feature_df.to_csv(features_save_path)
    preprocessed_outcome_df.to_csv(outcomes_save_path)

    if preproccessed_monitoring_data_path != '':
        # copy file to log dir
        shutil.copy(preproccessed_monitoring_data_path, log_dir)

    # verification of geneva_stroke_unit_preprocessing
    variable_presence_verification(preprocessed_feature_df, desired_time_range=desired_time_range)
    outcome_presence_verification(preprocessed_outcome_df, preprocessed_feature_df, log_dir=log_dir, outcomes=['Death in hospital', '3M Death'])


if __name__ == '__main__':
    """
    Example usage:
    python preprocessing_pipeline.py -e /.../mimic/extraction -a '/.../mimic_data/combined_notes_labels.xlsx'
                                        -ri '/.../opsum_prepro_output/logs_27122022_191859/tp0_imputation_parameters.csv'
                                        -rn '/.../opsum_prepro_output/logs_27122022_191859/normalisation_parameters.csv'
                                        -o '/.../mimic/preprocessing/'
                                        -m '/.../mimic/preprocessing/preprocessed_monitoring_df.csv'
                                        -mi '/.../stroke_datasets/national-institutes-of-health-stroke-scale-nihss-annotations-for-the-mimic-iii-database-1.0.0/mimic_nihss_database.csv'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--ehr_tables_path', type=str, help='EHR tables main path')
    parser.add_argument('-a','--admission_notes_path', type=str, help='Data extracted from admission notes')
    parser.add_argument('-ri','--reference_population_imputation_path', type=str, help='Path to imputation stats from reference population')
    parser.add_argument('-rn','--reference_population_normalisation_parameters_path', type=str, help='Path to normalisation stats from reference population')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-m', '--preprocessed_monitoring_path', type=str, help='Path to preprocessed monitoring data')
    parser.add_argument('-mi', '--mimic_admission_nihss_db_path', type=str, help='Path to the mimic nihss data')
    args = parser.parse_args()

    preprocess_and_save(args.ehr_tables_path, args.admission_notes_path,
           args.reference_population_imputation_path, args.reference_population_normalisation_parameters_path,
            args.output_dir,
           args.preprocessed_monitoring_path, args.mimic_admission_nihss_db_path, verbose=True)