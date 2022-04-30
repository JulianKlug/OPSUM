import pandas as pd

variables_to_normalize = [
'proBNP',
'bilirubine totale',
'thrombocytes',
'creatinine',
'calcium corrige',
'hemoglobine',
'INR',
'potassium',
'glycemie moyenne estimee',
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
]

def normalise_data(df: pd.DataFrame, verbose:bool = True) -> pd.DataFrame:
    """
    Normalise all continuous variables in the dataframe.
      - Winsorize values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
      - Scale to a mean of 0 with an SD of 1
    :param df: dataframe after restriction to plausible values
    :param verbose:
    :return:
    """

    if verbose:
        print(f'Following variables are not normalized:')
        for variable in df.sample_label.unique():
            if variable not in variables_to_normalize:
                print(f"'{variable}',")

    # Winsorize: values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
    winsorized_restricted_feature_df = df.copy()
    for variable in variables_to_normalize:
        temp = winsorized_restricted_feature_df[winsorized_restricted_feature_df.sample_label == variable].value.copy()
        # skip variables with insufficient range (FiO2, GCS)
        if temp.quantile(0.75) == temp.quantile(0.25):
            continue
        temp = temp.clip(lower=temp.quantile(0.25) - 1.5 * (temp.quantile(0.75) - temp.quantile(0.25)),
                         upper=temp.quantile(0.75) + 1.5 * (temp.quantile(0.75) - temp.quantile(0.25)))
        winsorized_restricted_feature_df.loc[winsorized_restricted_feature_df.sample_label == variable, 'value'] = temp

    # Scale to a mean of 0 with an SD of 1
    normalized_winsorized_restricted_feature_df = winsorized_restricted_feature_df.copy()
    for variable in variables_to_normalize:
        temp = normalized_winsorized_restricted_feature_df[
            normalized_winsorized_restricted_feature_df.sample_label == variable].value.copy()
        temp = (temp - temp.mean()) / temp.std()
        normalized_winsorized_restricted_feature_df.loc[
            normalized_winsorized_restricted_feature_df.sample_label == variable, 'value'] = temp

    return normalized_winsorized_restricted_feature_df
