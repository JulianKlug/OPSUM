import pandas as pd
import numpy as np
import os

selected_admission_data_columns = [
    "Age (calc.)",
    "Sex",
    "Time of symptom onset known",
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

# dropping some columns because of insufficient data
admission_data_to_drop = [
    '1st cholesterol total',
    '1st cholesterol LDL',
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


def preprocess_admission_data(stroke_db_df: pd.DataFrame, patient_selection_df: pd.DataFrame,
                              verbose=False) -> pd.DataFrame:
    stroke_db_df['patient_id'] = stroke_db_df['Case ID'].apply(lambda x: x[8:-4])
    stroke_db_df['EDS_last_4_digits'] = stroke_db_df['Case ID'].apply(lambda x: x[-4:])

    patient_selection_df['case_id'] = patient_selection_df['patient_id'].astype(str) + patient_selection_df[
        'EDS_last_4_digits'].astype(str)
    selected_stroke_db_df = stroke_db_df[
        stroke_db_df['Case ID'].apply(lambda x: x[8:]).isin(patient_selection_df['case_id'].tolist())]
    selected_stroke_db_df['begin_date'] = pd.to_datetime(selected_stroke_db_df['Arrival at hospital'],
                                                         format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                          selected_stroke_db_df['Arrival time']
    selected_stroke_db_df['case_admission_id'] = selected_stroke_db_df['patient_id'].astype(str) \
                                                 + selected_stroke_db_df['EDS_last_4_digits'].astype(str) + '_' +  \
                                                        selected_stroke_db_df['begin_date'].apply(
                                                        lambda bd: ''.join(bd.split(' ')[0].split('.')))

    admission_data_df = selected_stroke_db_df[selected_admission_data_columns
                                              + ['case_admission_id', 'begin_date']]
    admission_data_df = admission_data_df.drop(admission_data_to_drop, axis=1)

    # reducing categorical variable space
    admission_data_df.loc[
        admission_data_df['Referral'] == 'Other Stroke Unit or Stroke Center', 'Referral'] = 'Other hospital'
    admission_data_df.loc[
        admission_data_df['Referral'] == 'General Practitioner', 'Referral'] = 'Other hospital'

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

    # melt dataframe keeping patient_id and begin_date constant into two columns for sample_label and value
    admission_data_df = pd.melt(admission_data_df, id_vars=['case_admission_id', 'begin_date'], var_name='sample_label')

    # drop rows with missing values
    admission_data_df = admission_data_df.dropna()

    return admission_data_df

