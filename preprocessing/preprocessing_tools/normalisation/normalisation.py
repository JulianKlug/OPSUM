import pandas as pd
from tqdm import tqdm
import os

variables_to_normalize = [
'proBNP',
'bilirubine totale',
'thrombocytes',
'creatinine',
'calcium corrige',
'hemoglobine',
'INR',
'potassium',
'hematocrite',
'uree',
'erythrocytes',
'glucose',
'leucocytes',
'hemoglobine glyquee',
'sodium',
'proteine C-reactive',
'ALAT',
'FIO2',
'max_NIHSS',
'max_diastolic_blood_pressure',
'max_heart_rate',
'max_mean_blood_pressure',
'max_oxygen_saturation',
'max_respiratory_rate',
'max_systolic_blood_pressure',
'min_NIHSS',
'min_diastolic_blood_pressure',
'min_heart_rate',
'min_mean_blood_pressure',
'min_oxygen_saturation',
'min_respiratory_rate',
'min_systolic_blood_pressure',
'median_NIHSS',
'median_diastolic_blood_pressure',
'median_heart_rate',
'median_mean_blood_pressure',
'median_oxygen_saturation',
'median_respiratory_rate',
'median_systolic_blood_pressure',
'temperature',
'weight',
'age',
'triglycerides',
'ASAT',
'cholesterol HDL',
'Glasgow Coma Scale',
'fibrinogene',
'PTT',
'cholesterol total',
'LDL cholesterol calcule',
'chlore',
'lactate',
'CBF',
'T10',
'T4',
'T6',
'T8',
]

def normalise_data(df: pd.DataFrame, reference_population_normalisation_parameters_path:str = '',
                    verbose:bool = True, log_dir: str = '') -> pd.DataFrame:
    """
    Normalise all continuous variables in the dataframe.
      - Winsorize values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
      - Scale to a mean of 0 with an SD of 1
    :param df: dataframe after restriction to plausible values
    :param reference_population_normalisation_parameters_path: path to normalisation parameters of the reference population, this will be used instead of those of the currently processed population
        This should be used if a pretrained model is to be used on another dataset without retraining
    :param verbose:
    :param log_dir: directory to save logs to (mean and std for every normalised variable)
    :return:
    """

    if verbose:
        print(f'Following variables are not normalized:')
        for variable in df.sample_label.unique():
            if variable not in variables_to_normalize:
                print(f"'{variable}',")

    if reference_population_normalisation_parameters_path != '':
        if verbose:
            print('Using parameters from reference population for normalisation.')
        reference_population_normalisation_parameters_df = pd.read_csv(reference_population_normalisation_parameters_path)

    # Winsorize: values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
    winsorized_restricted_feature_df = df.copy()
    winsorizing_parameters_columns = ['variable', 'q25', 'q75']
    winsorizing_parameters_df = pd.DataFrame(columns=winsorizing_parameters_columns)
    if verbose:
        print('Winsorizing...')
    for variable in tqdm(variables_to_normalize):
        if variable not in winsorized_restricted_feature_df.sample_label.unique():
            if verbose:
                print(f'Variable {variable} not found in the dataframe')
            continue

        temp = winsorized_restricted_feature_df[winsorized_restricted_feature_df.sample_label == variable].value.copy()
        # skip variables with insufficient range (FiO2, GCS)
        winsorizing_parameters_df = winsorizing_parameters_df.append(
            pd.DataFrame([[variable, temp.quantile(0.25), temp.quantile(0.75)]], columns=winsorizing_parameters_columns))

        # the winsorizing parameters for the validation dataset are derived from reference dataset
        if reference_population_normalisation_parameters_path != '':
            # use winsorisation parameters from reference population
            ref_pop_q25 = reference_population_normalisation_parameters_df[
                reference_population_normalisation_parameters_df.variable == variable].q25.iloc[0]
            ref_pop_q75 = reference_population_normalisation_parameters_df[
                reference_population_normalisation_parameters_df.variable == variable].q75.iloc[0]
            if ref_pop_q25 == ref_pop_q75:
                continue
            temp = temp.clip(lower=ref_pop_q25 - 1.5 * (ref_pop_q75 - ref_pop_q25),
                             upper=ref_pop_q75 + 1.5 * (ref_pop_q75 - ref_pop_q25))
        else:
            # derive winsorization parameters from current population
            if temp.quantile(0.75) == temp.quantile(0.25):
                continue
            temp = temp.clip(lower=temp.quantile(0.25) - 1.5 * (temp.quantile(0.75) - temp.quantile(0.25)),
                             upper=temp.quantile(0.75) + 1.5 * (temp.quantile(0.75) - temp.quantile(0.25)))
        winsorized_restricted_feature_df.loc[winsorized_restricted_feature_df.sample_label == variable, 'value'] = temp

    # Scale to a mean of 0 with an SD of 1
    normalized_winsorized_restricted_feature_df = winsorized_restricted_feature_df.copy()
    # preparing a dataframe with mean and std for every normalised variable to save to log file to enable reverse operation
    normalisation_parameters_columns = ['variable', 'original_mean', 'original_std']
    normalisation_parameters_df = pd.DataFrame(columns=normalisation_parameters_columns)
    if verbose:
        print('Normalising...')
    for variable in tqdm(variables_to_normalize):
        temp = normalized_winsorized_restricted_feature_df[
            normalized_winsorized_restricted_feature_df.sample_label == variable].value.copy()
        normalisation_parameters_df = normalisation_parameters_df.append(
            pd.DataFrame([[variable, temp.mean(), temp.std()]], columns=normalisation_parameters_columns))

        print(variable)
        if reference_population_normalisation_parameters_path != '':
            # use normalisation parameters from a reference population
            ref_pop_mean = reference_population_normalisation_parameters_df[
                                    reference_population_normalisation_parameters_df.variable == variable].original_mean.iloc[0]
            ref_pop_std = reference_population_normalisation_parameters_df[
                                    reference_population_normalisation_parameters_df.variable == variable].original_std.iloc[0]

            temp = (temp - ref_pop_mean) / ref_pop_std
        else:
            # use normalisation parameters from processed cohort
            temp = (temp - temp.mean()) / temp.std()
        normalized_winsorized_restricted_feature_df.loc[
            normalized_winsorized_restricted_feature_df.sample_label == variable, 'value'] = temp

    if log_dir != '':
        if reference_population_normalisation_parameters_path != '':
            reference_population_normalisation_parameters_df.to_csv(os.path.join(log_dir, 'reference_population_normalisation_parameters.csv'), index=False)
        else:
            # merge winsorizing_parameters_df to normalisation_parameters_df on variable
            normalisation_parameters_df = normalisation_parameters_df.merge(winsorizing_parameters_df, on='variable', how='left')
            normalisation_parameters_df.to_csv(os.path.join(log_dir, 'normalisation_parameters.csv'), index=False)


    return normalized_winsorized_restricted_feature_df
