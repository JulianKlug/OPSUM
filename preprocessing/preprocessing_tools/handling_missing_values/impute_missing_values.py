import pandas as pd
from tqdm import tqdm
import numpy as np
import os

categorical_vars = [
    'sex_male',
'referral_in-hospital_event',
'referral_other_hospital',
'referral_self_referral_or_gp',
'prestroke_disability_(rankin)_1.0',
'prestroke_disability_(rankin)_2.0',
'prestroke_disability_(rankin)_3.0',
'prestroke_disability_(rankin)_4.0',
'prestroke_disability_(rankin)_5.0',
'antihypert._drugs_pre-stroke_yes',
'lipid_lowering_drugs_pre-stroke_yes',
'antiplatelet_drugs_yes',
'anticoagulants_yes',
'medhist_hypertension_yes',
'medhist_diabetes_yes',
'medhist_hyperlipidemia_yes',
'medhist_smoking_yes',
'medhist_atrial_fibr._yes',
'medhist_chd_yes',
'medhist_pad_yes',
'medhist_cerebrovascular_event_true',
'categorical_onset_to_admission_time_541-1440min',
'categorical_onset_to_admission_time_<270min',
'categorical_onset_to_admission_time_>1440min',
'categorical_onset_to_admission_time_intra_hospital',
'categorical_onset_to_admission_time_onset_unknown',
'wake_up_stroke_true',
'categorical_ivt_91-270min',
'categorical_ivt_<90min',
'categorical_ivt_>540min',
'categorical_ivt_no_ivt',
'categorical_iat_<270min',
'categorical_iat_>540min',
'categorical_iat_no_iat',
]


def impute_missing_values(df:pd.DataFrame, reference_population_imputation_path: str ='',
                          verbose:bool=True, log_dir:str='',
                          desired_time_range:int=72) -> pd.DataFrame:
    """
    Impute missing values in the dataframe.
    Missing values, are imputed by last observation carried forward (LOCF).
    Population medians in the datasets were used for missing values occurring before the first actual measurement.

    Requirements:
    - Should be run after encoding categorical variables and resampling to timebins
    - Should be run before normalization

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values.
    reference_population_imputation_path : str, optional
        Path to the reference population imputation values (i.e. logs obtained from imputation from reference population). The default is ''.
    verbose : bool
        If True, print the number of missing values in each column.
    log_dir : str
        Path to the log directory. The default is ''.
    desired_time_range : int
        The desired time range in hours. The default is 72.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed.
    """
    imputed_missing_df = df.copy()

    if reference_population_imputation_path != '':
        reference_population_imputation_df = pd.read_csv(reference_population_imputation_path)
        labels_imputed_from_reference_population = []

    # Compute missingness of variables before imputing missing variables
    if log_dir != '':
        if verbose:
            print('Computing missingness.')
        missingness_df_columns = ['sample_label', 'n_missing_overall'] + [f'n_missing_h{i}' for i in
                                                                          range(desired_time_range)]
        missingness_df = pd.DataFrame(columns=missingness_df_columns)
        for variable in tqdm(df.sample_label.unique()):
            # compute number of missing cid overall
            n_missing_cids_overall = df.case_admission_id.nunique() - df[
                df.sample_label == variable].case_admission_id.nunique()
            # compute number of missing cid per time bin
            n_missing_cids_per_time_bin = [df.case_admission_id.nunique() -
                                           df[(df.sample_label == variable)
                                                        & (df.relative_sample_date_hourly_cat == time_bin)]
                                           .case_admission_id.nunique()
                                           for time_bin in range(desired_time_range)]

            missingness_df = missingness_df.append(
                pd.DataFrame([[variable, n_missing_cids_overall] + n_missing_cids_per_time_bin],
                             columns=missingness_df_columns))

        missingness_df.to_csv(os.path.join(log_dir, 'missingness.csv'), index=False)


    # Handle first missing values (timebin 0)
    # -> fill with population median/mode
    if verbose:
        print('Fill fist missing values via population mean/median.')

    # log population-wise imputation
    imputation_parameters_columns = ['variable', 'imputed_value', 'imputation_method', 'imputation_range']
    imputation_parameters_df = pd.DataFrame(columns=imputation_parameters_columns)

    for sample_label in tqdm(imputed_missing_df.sample_label.unique()):
        # find case_admission_ids with no value for sample_label in first timebin
        patients_with_no_sample_label_tp0 = set(imputed_missing_df.case_admission_id.unique()).difference(set(
            imputed_missing_df[(imputed_missing_df.sample_label == sample_label) & (
                        imputed_missing_df.relative_sample_date_hourly_cat == 0)].case_admission_id.unique()))

        n_missing_cids_overall = df.case_admission_id.nunique() - df[
            df.sample_label == sample_label].case_admission_id.nunique()

        if sample_label == 'FIO2':
            # for FIO2, impute with 21.0%
            imputed_tp0_value = 21.0
            imputation_method = 'default FiO2'
            imputation_range = 'overall'

        elif (n_missing_cids_overall > 2/3 * df.case_admission_id.nunique()) & (reference_population_imputation_path != ''):
        #  if sample label has a lot of missing values (~50%), then use mean/median of the reference population
            if sample_label in categorical_vars:
                # not implemented
                raise NotImplementedError('Imputation from reference population of categorical variables is not implemented.')
            else:
                # use median
                imputed_tp0_value = reference_population_imputation_df[
                                        (reference_population_imputation_df.variable == sample_label)
                                        & (reference_population_imputation_df.imputation_method == 'median')]\
                                        ['imputed_value'].iloc[0]
            labels_imputed_from_reference_population.append([sample_label, imputed_tp0_value, len(patients_with_no_sample_label_tp0)])
            imputation_method = 'reference_population_median'
            imputation_range = 'reference_population'
        elif sample_label in categorical_vars:
            # for categorical vars, impute with mode
            n_missing_cids_tp0 = len(patients_with_no_sample_label_tp0)
            if n_missing_cids_tp0 > 0.5 * df.case_admission_id.nunique():
                # impute over all timepoints
                imputed_tp0_value = imputed_missing_df[imputed_missing_df.sample_label == sample_label].value.mode().iloc[0]
                imputation_range = 'all_tp'
            else:
                imputed_tp0_value = imputed_missing_df[(imputed_missing_df.sample_label == sample_label) & (
                        imputed_missing_df.relative_sample_date_hourly_cat == 0)].value.mode()[0]
                imputation_range = 'tp0'
            imputation_method = 'mode'
        else:
            # for numerical vars, impute with median
            n_missing_cids_tp0 = len(patients_with_no_sample_label_tp0)
            if n_missing_cids_tp0 > 0.5 * df.case_admission_id.nunique():
                # impute over all timepoints
                imputed_tp0_value = imputed_missing_df[imputed_missing_df.sample_label == sample_label].value.median()
                imputation_range = 'all_tp'
            else:
                imputed_tp0_value = imputed_missing_df[(imputed_missing_df.sample_label == sample_label) & (
                        imputed_missing_df.relative_sample_date_hourly_cat == 0)].value.median()
                imputation_range = 'tp0'
            imputation_method = 'median'
        if verbose:
            print(
                f'{len(patients_with_no_sample_label_tp0)} patients with no {sample_label} in first timebin for which {imputed_tp0_value} was imputed')

        sample_label_original_source = \
            imputed_missing_df[imputed_missing_df.sample_label == sample_label].source.mode(dropna=True)[0]

        imputed_sample_label = pd.DataFrame({'case_admission_id': list(patients_with_no_sample_label_tp0),
                                             'sample_label': sample_label,
                                             'relative_sample_date_hourly_cat': 0,
                                             'source': f'{sample_label_original_source}_pop_imputed',
                                             'value': imputed_tp0_value})

        # impute missing values for sample_label in first timebin
        imputed_missing_df = imputed_missing_df.append(imputed_sample_label, ignore_index=True)

        imputation_parameters_df = imputation_parameters_df.append(
            pd.DataFrame([[sample_label, imputed_tp0_value, imputation_method, imputation_range]], columns=imputation_parameters_columns))

    if log_dir != '':
        if (reference_population_imputation_path != ''):
            # save labels imputed from reference population
            pd.DataFrame(labels_imputed_from_reference_population, columns=['label', 'imputed_value', 'imputed_for_n_subjects'])\
                .to_csv(os.path.join(log_dir, 'labels_imputed_from_reference_population.csv'), index=False)

        # save imputation parameters
        imputation_parameters_df.to_csv(os.path.join(log_dir, 'tp0_imputation_parameters.csv'), index=False)

    # following missing values (timebin > 0)
    # -> Fill missing timebin values by last observation carried forward
    if verbose:
        print('Fill missing values via LOCF.')

    locf_imputed_missing_df = imputed_missing_df.groupby(['case_admission_id', 'sample_label']).apply(
        lambda x: x.set_index('relative_sample_date_hourly_cat').reindex(range(0, 72)))
    locf_imputed_missing_df.value = locf_imputed_missing_df.value.fillna(method='ffill')
    locf_imputed_missing_df.sample_label = locf_imputed_missing_df.sample_label.fillna(method='ffill')
    locf_imputed_missing_df.case_admission_id = locf_imputed_missing_df.case_admission_id.fillna(method='ffill')

    locf_imputed_missing_df['source_imputation'] = locf_imputed_missing_df.source.apply(lambda x: '' if type(x) == str else np.nan)
    locf_imputed_missing_df.source_imputation = locf_imputed_missing_df.source_imputation.fillna('_locf_imputed')
    locf_imputed_missing_df.source = locf_imputed_missing_df.source.fillna(method='ffill')
    locf_imputed_missing_df.source += locf_imputed_missing_df.source_imputation
    locf_imputed_missing_df.drop(columns=['source_imputation'], inplace=True)

    # reset relative_sample_date_hourly_cat as column
    locf_imputed_missing_df.reset_index(level=2, inplace=True)
    # drop groupby index
    locf_imputed_missing_df.reset_index(inplace=True, drop=True)

    return locf_imputed_missing_df




