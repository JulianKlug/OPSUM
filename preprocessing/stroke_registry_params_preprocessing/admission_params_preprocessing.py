import pandas as pd
import numpy as np
import os

selected_admission_data_columns = [
    "Age (calc.)",
    "Sex",
    "Referral",
    "Prestroke disability (Rankin)",
    "NIH on admission",
    "1st syst. bp",
    "1st diast. bp",
    "Weight",
    "Antihypert. drugs pre-stroke",
    "Lipid lowering drugs pre-stroke",
    "Hormone repl. or contracept.",
    "Antiplatelet drugs",
    "Anticoagulants",
    "MedHist Stroke",
    "MedHist TIA",
    "MedHist ICH",
    "MedHist Hypertension",
    "MedHist Diabetes",
    "MedHist Hyperlipidemia",
    "MedHist Smoking",
    "MedHist Atrial Fibr.",
    "MedHist CHD",
    "MedHist Prost. heart valves",
    "MedHist PAD",
    "1st glucose",
    "1st cholesterol total",
    "1st cholesterol LDL",
    "1st creatinine",
]

# dropping some columns because of insufficient data / irrelevant features
admission_data_to_drop = [
    'MedHist Prost. heart valves',
    'Hormone repl. or contracept.'
]


def restrict_variable_to_possible_ranges(df, variable_name, possible_value_ranges, verbose=False):
    """
    Restricts a variable to the possible ranges in the possible_value_ranges dataframe.
    """
    variable_range = possible_value_ranges[possible_value_ranges['variable_label'] == variable_name]
    variable_range = variable_range.iloc[0]
    clean_df = df.copy()
    # set score to np.nan if outside of range
    clean_df.loc[(df[variable_name] < variable_range['Min']), variable_name] = np.nan
    clean_df.loc[(df[variable_name] > variable_range['Max']), variable_name] = np.nan
    if verbose:
        print(f'Excluding {clean_df[variable_name].isna().sum()} observations because out of range')
    excluded_df = df[clean_df[variable_name].isna()]
    return clean_df, excluded_df


def preprocess_admission_data(stroke_registry_df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    stroke_registry_df['patient_id'] = stroke_registry_df['Case ID'].apply(lambda x: x[8:-4])
    stroke_registry_df['EDS_last_4_digits'] = stroke_registry_df['Case ID'].apply(lambda x: x[-4:])

    stroke_registry_df['begin_date'] = pd.to_datetime(stroke_registry_df['Arrival at hospital'],
                                                         format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                          stroke_registry_df['Arrival time']
    stroke_registry_df['case_admission_id'] = stroke_registry_df['patient_id'].astype(str) \
                                                 + stroke_registry_df['EDS_last_4_digits'].astype(str) + '_' +  \
                                                        stroke_registry_df['begin_date'].apply(
                                                        lambda bd: ''.join(bd.split(' ')[0].split('.')))

    admission_data_df = stroke_registry_df[selected_admission_data_columns
                                              + ['case_admission_id', 'begin_date']]
    admission_data_df = admission_data_df.drop(admission_data_to_drop, axis=1)

    # restricting to plausible range
    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              'possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)

    admission_data_df.rename(columns={'Weight': 'weight'}, inplace=True)
    admission_data_df, _ = restrict_variable_to_possible_ranges(admission_data_df, 'weight', possible_value_ranges,
                                                                verbose=verbose)

    admission_data_df.rename(columns={'Age (calc.)': 'age'}, inplace=True)
    admission_data_df, excluded_age_df = restrict_variable_to_possible_ranges(admission_data_df,
                                                                              'age', possible_value_ranges,
                                                                              verbose=verbose)

    admission_data_df.rename(columns={'1st syst. bp': 'sys'}, inplace=True)
    admission_data_df, excluded_sys_df = restrict_variable_to_possible_ranges(admission_data_df,
                                                                              'sys', possible_value_ranges,
                                                                              verbose=verbose)
    admission_data_df.rename(columns={'sys': 'systolic_blood_pressure'}, inplace=True)

    admission_data_df.rename(columns={'1st diast. bp': 'dia'}, inplace=True)
    admission_data_df, excluded_dia_df = restrict_variable_to_possible_ranges(admission_data_df,
                                                                              'dia', possible_value_ranges,
                                                                              verbose=verbose)
    admission_data_df.rename(columns={'dia': 'diastolic_blood_pressure'}, inplace=True)

    admission_data_df.rename(columns={'1st glucose': 'glucose'}, inplace=True)
    admission_data_df, excluded_glucose_df = restrict_variable_to_possible_ranges(admission_data_df,
                                                                                  'glucose',
                                                                                  possible_value_ranges,
                                                                                  verbose=verbose)

    admission_data_df.rename(columns={'1st creatinine': 'creatinine'}, inplace=True)
    admission_data_df, excluded_creatinine_df = restrict_variable_to_possible_ranges(
        admission_data_df, 'creatinine', possible_value_ranges, verbose=verbose)

    # reducing categorical variable space
    admission_data_df.loc[
        admission_data_df['Referral'] == 'Other Stroke Unit or Stroke Center', 'Referral'] = 'Other hospital'
    admission_data_df.loc[
        admission_data_df['Referral'] == 'General Practitioner', 'Referral'] = 'Self referral or GP'
    admission_data_df.loc[
        admission_data_df['Referral'] == 'Self referral', 'Referral'] = 'Self referral or GP'

    # fusion of similar variable categories
    admission_data_df['MedHist cerebrovascular_event'] = (
                admission_data_df[['MedHist Stroke', 'MedHist TIA', 'MedHist ICH']] == 'yes').any(axis=1)
    admission_data_df.drop(columns=['MedHist Stroke', 'MedHist TIA', 'MedHist ICH'], inplace=True)

    # Rename columns for EHR correspondence
    admission_data_df.rename(columns={'1st cholesterol total': 'cholesterol total',
                                      '1st cholesterol LDL':'LDL cholesterol calcule',
                                      'NIH on admission':'NIHSS'}, inplace=True)

    # dealing with missing values
    # - for variables with DPI overlap -> leave NaN for now (should be dealt with after fusion)
    # - for variables with no DPI overlap -> fill with median
    variables_with_dpi_overlap = ['case_admission_id', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'glucose',
                                  'creatinine', 'NIHSS', 'weight', 'cholesterol total', 'LDL cholesterol calcule']
    continuous_variables = ['age']
    for variable in admission_data_df.columns:
        if variable in variables_with_dpi_overlap:
            continue
        if variable in continuous_variables:
            admission_data_df[variable].fillna(admission_data_df[variable].median(skipna=True),
                                                        inplace=True)
        else:
            admission_data_df[variable].fillna(admission_data_df[variable].mode(dropna=True)[0],
                                                        inplace=True)


    # melt dataframe keeping patient_id and begin_date constant into two columns for sample_label and value
    admission_data_df = pd.melt(admission_data_df, id_vars=['case_admission_id', 'begin_date'], var_name='sample_label')

    # drop rows with missing values
    admission_data_df = admission_data_df.dropna()

    return admission_data_df
