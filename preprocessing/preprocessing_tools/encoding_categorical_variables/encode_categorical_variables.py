import pandas as pd
import os

categorical_variables = [
    'Sex',
    'Referral',
    'Prestroke disability (Rankin)',
    'Antihypert. drugs pre-stroke',
    'Lipid lowering drugs pre-stroke',
    'Antiplatelet drugs',
    'Anticoagulants',
    'MedHist Hypertension',
    'MedHist Diabetes',
    'MedHist Hyperlipidemia',
    'MedHist Smoking',
    'MedHist Atrial Fibr.',
    'MedHist CHD',
    'MedHist PAD',
    'MedHist cerebrovascular_event',
    'categorical_onset_to_admission_time',
    'wake_up_stroke',
    'categorical_IVT',
    'categorical_IAT'
]

def encode_categorical_variables(df: pd.DataFrame, reference_categorical_encoding_path: str = '',
                                 verbose:bool = True, log_dir:str = '') -> pd.DataFrame:
    """
    Encode categorical variables as numeric values through one-hot encoding.
    The difference between binary and non-binary variables is irrelevant when one-hot encoding variables:
        - binary variables are encoded as 0 and 1 of one of the two categories as (variable_category1)
        - non-binary variables are encoded as 0 and 1 of for the n-1 categories as (variable_category1, variable_category2, …, variable_category_n-1)
    :param df: DataFrame with columns ['case_admission_id', 'sample_date', 'source', 'first_sample_date', 'relative_sample_date', 'sample_label', 'value']
    :param verbose:
    :param log_dir: path to save logs
    :return: dataframe with encoded categorical variables
    """

    if verbose:
        print(f'Following variables are not considered categorical and will not be encoded:')
        for variable in df.sample_label.unique():
            if variable not in categorical_variables:
                print(f"'{variable}',")

    if reference_categorical_encoding_path != '':
        if verbose:
            print(f'Loading reference categorical encoding from {reference_categorical_encoding_path}')
        reference_categorical_encoding = pd.read_csv(reference_categorical_encoding_path)

    one_hot_encoded_df = df.copy()
    log_columns = ['sample_label', 'baseline_value', 'other_categories']
    log_df = pd.DataFrame(columns=log_columns)

    for categorical_variable in categorical_variables:
        if reference_categorical_encoding_path == '':
            # If there is no reference categorical encoding, create one from the data
            dummy_coded_temp = pd.get_dummies(one_hot_encoded_df[
                                                  one_hot_encoded_df.sample_label == categorical_variable],
                                              columns=['value'], prefix=str(categorical_variable).lower(), drop_first=True)

            # find baseline values for each categorical variable
            baseline_value = [var
                              for var in one_hot_encoded_df[
                                  one_hot_encoded_df.sample_label == categorical_variable][
                                  'value'].unique()
                              if str(var) not in
                              [col_name.split(str(categorical_variable).lower() + '_')[-1] for col_name in
                               dummy_coded_temp.columns]]
        else:
            # If there is a reference categorical encoding, use it
            dummy_coded_temp = pd.get_dummies(one_hot_encoded_df[
                                                  one_hot_encoded_df.sample_label == categorical_variable],
                                              columns=['value'], prefix=str(categorical_variable).lower())
            # remove baseline column
            reference_baseline_value = reference_categorical_encoding[reference_categorical_encoding.sample_label == categorical_variable]['baseline_value'].values[0]\
                .replace("[", "").replace("]", "").replace("'", "")
            reference_baseline_column = str(categorical_variable).lower() + '_' + reference_baseline_value
            dummy_coded_temp = dummy_coded_temp.drop(columns=[reference_baseline_column])
            baseline_value = reference_baseline_value

            # add columns for variables not present in the data but present in the reference encoding
            reference_other_categories = reference_categorical_encoding[reference_categorical_encoding.sample_label == categorical_variable]['other_categories'].values[0]\
                .replace("[", "").replace("]", "").replace("'", "").split(', ')
            for category in reference_other_categories:
                category_column = str(categorical_variable).lower() + '_' + category
                if category_column not in dummy_coded_temp.columns:
                    dummy_coded_temp[str(categorical_variable).lower() + '_' + category] = 0


        if log_dir != '':
            # other values
            other_categories = [col_name.split(str(categorical_variable).lower() + '_')[-1] for col_name in
                                    set(dummy_coded_temp.columns) - set(one_hot_encoded_df.columns)]
            log_df = log_df.append(pd.DataFrame([[categorical_variable, baseline_value, other_categories]],
                                                columns=log_columns))

        if verbose:
            print(f'Baseline for {categorical_variable}: {baseline_value}')

        dummy_coded_temp.columns = [str(col).lower().replace(' ', '_') for col in dummy_coded_temp.columns]
        dummy_coded_temp.drop(columns=['sample_label'], inplace=True)
        dummy_coded_temp = dummy_coded_temp.melt(
            id_vars=['case_admission_id', 'sample_date', 'source', 'first_sample_date', 'relative_sample_date'],
            var_name='sample_label', value_name='value')
        one_hot_encoded_df = one_hot_encoded_df.append(dummy_coded_temp)

        # drop original non-binary categorical variable
        one_hot_encoded_df = one_hot_encoded_df[
            one_hot_encoded_df.sample_label != categorical_variable]

    if log_dir != '':
        log_df.to_csv(os.path.join(log_dir, 'categorical_variable_encoding.csv'), index=False)

    return one_hot_encoded_df

