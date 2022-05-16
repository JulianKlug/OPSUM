import argparse
import os

import pandas as pd
import time

from preprocessing.encoding_categorical_variables.encode_categorical_variables import encode_categorical_variables
from preprocessing.handling_missing_values.impute_missing_values import impute_missing_values
from preprocessing.normalisation.normalisation import normalise_data
from preprocessing.outcome_preprocessing.outcome_preprocessing import preprocess_outcomes
from preprocessing.resample_to_time_bins.resample_to_hourly_features import resample_to_hourly_features
from preprocessing.variable_assembly.variable_database_assembly import assemble_variable_database
from preprocessing.variable_assembly.relative_timestamps import transform_to_relative_timestamps


def preprocess(ehr_data_path:str, stroke_registry_data_path:str, patient_selection_path:str, verbose:bool=True) -> pd.DataFrame:
    """
    Apply preprocessing pipeline detailed in ./preprocessing/readme.md
    :param ehr_data_path: path to EHR data
    :param stroke_registry_data_path: path to stroke registry (admission) data
    :param patient_selection_path: path to patient selection file
    :return: preprocessed feature Dataframe, preprocessed outcome dataframe
    """

    # 1. Restrict to patient selection
    # 2. Preprocess EHR and stroke registry variables
    # 3. Restrict to variable selection
    # 4. Assemble database from lab/scales/ventilation/vitals + stroke registry subparts
    print('STARTING VARIABLE PREPROCESSING')
    feature_df = assemble_variable_database(ehr_data_path, stroke_registry_data_path, patient_selection_path, verbose=verbose)

    # 5. Transform timestamps to relative timestamps from first measure
    # 6. Restrict to time range
    print('TRANSFORMING TO RELATIVE TIME AND RESTRICTING TIME RANGE')
    restricted_feature_df = transform_to_relative_timestamps(feature_df, drop_old_columns=False,
                                                             restrict_to_time_range=True)

    # 7. Encoding categorical variables (one-hot)
    print('ENCODING CATEGORICAL VARIABLES')
    cat_encoded_restricted_feature_df = encode_categorical_variables(restricted_feature_df, verbose=verbose)

    # 8. Resampling to hourly frequency
    print('RESAMPLING TO HOURLY FREQUENCY')
    resampled_df = resample_to_hourly_features(cat_encoded_restricted_feature_df, verbose=verbose)

    # 9. imputation of missing values
    print('IMPUTING MISSING VALUES')
    imputed_missing_df = impute_missing_values(resampled_df, verbose=verbose)

    # 10. normalisation
    print('APPLYING NORMALISATION')
    normalised_df = normalise_data(imputed_missing_df, verbose=verbose)

    # 11. preprocessing outcomes
    preprocessed_outcomes_df = preprocess_outcomes(stroke_registry_data_path, patient_selection_path, verbose=verbose)

    return normalised_df, preprocessed_outcomes_df


def preprocess_and_save(ehr_data_path:str, stroke_registry_data_path:str, patient_selection_path:str, output_dir:str,
                        feature_file_prefix:str = 'preprocessed_features', outcome_file_prefix:str = 'preprocessed_outcomes', verbose:bool=True):

    preprocessed_feature_df, preprocessed_outcome_df = preprocess(ehr_data_path, stroke_registry_data_path, patient_selection_path, verbose=verbose)
    timestamp = time.strftime("%d%m%Y_%H%M%S")
    features_save_path = os.path.join(output_dir, f'{feature_file_prefix}_{timestamp}.csv')
    outcomes_save_path = os.path.join(output_dir, f'{outcome_file_prefix}_{timestamp}.csv')

    preprocessed_feature_df.to_csv(features_save_path)
    preprocessed_outcome_df.to_csv(outcomes_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--ehr', type=str, help='EHR data path')
    parser.add_argument('-r', '--registry', type=str, help='Registry data path')
    parser.add_argument('-p', '--patients', type=str, help='Patient selection file')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    preprocess_and_save(args.ehr, args.registry, args.patients, args.output_dir)
