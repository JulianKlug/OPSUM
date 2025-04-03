import argparse
import os
import numpy as np
import pandas as pd

from preprocessing.mimic_preprocessing.database_assembly.further_exclusion_criteria import \
    apply_further_exclusion_criteria

CONTINUOUS_CHARACTERISTICS = [
    'Age (calc.)',
    'Prestroke disability (Rankin)',
    'NIH on admission',
    'BMI',
    '3M mRS'
]

CATEGORICAL_CHARACTERISTICS = [
    'Sex',
    'IVT with rtPA',
    'IAT',
    'MedHist Hypertension',
    'MedHist Diabetes',
    'MedHist Hyperlipidemia',
    'MedHist Atrial Fibr.',
    '3M Death'
]


def extract_patient_characteristics(admission_notes_data_path: str, extracted_monitoring_path: str,
                                    extracted_admission_table_path: str, preprocessed_outcomes_path: str,
                                    continuous_characteristics: list = CONTINUOUS_CHARACTERISTICS,
                                    categorical_characteristics: list = CATEGORICAL_CHARACTERISTICS) -> pd.DataFrame:
    """
    Extracts patient characteristics for the MIMIC data.
    :param admission_notes_data_path: path to the data manually extracted from the admission/discharge notes.
    :param extracted_monitoring_data_path: path to the extracted monitoring data.
    :param continuous_characteristics: list of continuous characteristics to extract.
    :param categorical_characteristics: list of categorical characteristics to extract.
    :return: a dataframe with the patient characteristics.
    """

    # load the data
    admission_data_df = pd.read_excel(admission_notes_data_path)
    admission_table_df = pd.read_csv(extracted_admission_table_path)
    monitoring_df = pd.read_csv(extracted_monitoring_path)
    outcomes_df = pd.read_csv(preprocessed_outcomes_path)

    # restrict to patients admitted to ICU with stroke as primary reason and with onset to admission < 7 d
    admission_data_df = admission_data_df[admission_data_df['admitted to ICU for stroke'] == 'y']
    admission_data_df = admission_data_df[admission_data_df['onset to ICU admission > 7d'] == 'n']
    admission_data_df['case_admission_id'] = admission_data_df['hadm_id'].astype(str) + '_' + admission_data_df[
        'icustay_id'].astype(str)

    # Deduce patient selection from admission data
    # Apply further exclusion criteria
    hadm_ids_to_exclude = apply_further_exclusion_criteria(admission_data_df['case_admission_id'].unique(), extracted_admission_table_path, log_dir='')
    admission_data_df = admission_data_df[~admission_data_df['case_admission_id'].isin(hadm_ids_to_exclude)]
    patient_selection = admission_data_df['case_admission_id'].unique()

    # extract BMI from monitoring data
    height_labels = ['Height (cm)', 'Height', 'Admit Ht']
    weight_labels = ['Admit Wt', 'Admission Weight (lbs.)', 'Admission Weight (Kg)', 'Previous WeightF',
                     'Previous Weight', 'Daily Weight']
    monitoring_df['case_admission_id'] = monitoring_df['hadm_id'].astype(int).astype(str) + '_' + monitoring_df[
        'icustay_id'].astype(int).astype(str)
    monitoring_df = monitoring_df[monitoring_df['case_admission_id'].isin(patient_selection)]
    height_df = monitoring_df[monitoring_df['label'].isin(height_labels)]
    height_df.dropna(subset=['valuenum'], inplace=True)
    # convert inches to cm if necessary
    height_df.loc[height_df['valueuom'].isin(['inches', 'Inch']), 'valuenum'] = height_df.loc[
                                                                                    height_df['valueuom'].isin(
                                                                                        ['inches',
                                                                                         'Inch']), 'valuenum'] * 2.54
    # take only the first (by charrttime) height measurement per case and keep case_admission_id
    height_df = height_df.sort_values(by=['charttime']).groupby('case_admission_id').first().reset_index()[
        ['case_admission_id', 'valuenum']]
    height_df.rename(columns={'valuenum': 'height'}, inplace=True)

    weight_df = monitoring_df[monitoring_df['label'].isin(weight_labels)]
    weight_df.dropna(subset=['valuenum'], inplace=True)
    # take only the first (by charrttime) weight measurement per case and keep case_admission_id
    weight_df = weight_df.sort_values(by=['charttime']).groupby('case_admission_id').first().reset_index()[
        ['case_admission_id', 'valuenum']]
    weight_df.rename(columns={'valuenum': 'weight'}, inplace=True)

    bmi_df = pd.merge(height_df, weight_df, on='case_admission_id', how='inner')
    bmi_df['BMI'] = bmi_df['weight'] / (bmi_df['height'] / 100) ** 2
    admission_data_df = pd.merge(admission_data_df, bmi_df, on='case_admission_id', how='left')

    # Extract data from admission table
    # Preprocessing admission table data
    admission_table_df = admission_table_df[
        ['subject_id', 'hadm_id', 'icustay_id', 'dob', 'admittime', 'age', 'gender', 'admission_location']]
    admission_table_df['case_admission_id'] = admission_table_df['hadm_id'].astype(str) + '_' + admission_table_df[
        'icustay_id'].astype(str)
    admission_table_df.drop_duplicates(inplace=True)
    # for patients with age at 300; set to 90 (this is an artifact of the MIMIC database, to anonymize the data)
    admission_table_df.loc[admission_table_df['age'] > 250, 'age'] = 90

    admission_table_df.rename(columns={'gender': 'Sex'}, inplace=True)
    # encode 'Sex' to ['Female', 'Male']
    admission_table_df.loc[admission_table_df.Sex == 'F', 'Sex'] = 'Female'
    admission_table_df.loc[admission_table_df.Sex == 'M', 'Sex'] = 'Male'

    # join admission table data to admission data
    admission_data_df = pd.merge(admission_data_df, admission_table_df, on='case_admission_id', how='left')
    admission_data_df['IVT with rtPA'] = ~admission_data_df['IVT time'].isna()
    admission_data_df['IAT'] = ~admission_data_df['IAT time'].isna()

    # extract data from outcomes
    outcomes_df = outcomes_df[['case_admission_id', 'Death in hospital', '3M Death']].drop_duplicates()
    admission_data_df = pd.merge(admission_data_df, outcomes_df, on='case_admission_id', how='left')
    admission_data_df['3M mRS'] = np.nan

    # Align nomenclature
    admission_data_df.rename(columns={'age': 'Age (calc.)'}, inplace=True)
    admission_data_df.rename(columns={'prestroke mRS': 'Prestroke disability (Rankin)'}, inplace=True)
    admission_data_df.rename(columns={'admission NIHSS': 'NIH on admission'}, inplace=True)

    medhist_columns = ['MedHist Hypertension',
                      'MedHist Diabetes',
                      'MedHist Hyperlipidemia',
                      'MedHist Atrial Fibr.',
                      ]
    admission_data_df[medhist_columns] = admission_data_df[medhist_columns].replace({'y': 'yes', 'n': 'no'})
    admission_data_df[['IVT with rtPA', 'IAT']] = admission_data_df[['IVT with rtPA', 'IAT']].replace({True: 'yes',
                                                                                                        False: 'no'})
    admission_data_df[['Death in hospital', '3M Death']] = admission_data_df[['Death in hospital', '3M Death']].replace(
        {1: 'yes', 0: 'no'})

    # Extract population statistics
    patient_characteristics_df = pd.DataFrame()
    patient_characteristics_df['n patients'] = [len(admission_data_df)]

    # extract continuous characteristics
    for characteristic in continuous_characteristics:
        patient_characteristics_df[f'median {characteristic}'] = [admission_data_df[characteristic].median()]
        patient_characteristics_df[f'Q25 {characteristic}'] = [admission_data_df[characteristic].quantile(0.25)]
        patient_characteristics_df[f'Q75 {characteristic}'] = [admission_data_df[characteristic].quantile(0.75)]
        # count number of missing values for characteristic
        patient_characteristics_df[f'n missing {characteristic}'] = [admission_data_df[characteristic].isnull().sum()]

    for characteristic in categorical_characteristics:
        # get number of most common value for each categorical characteristic
        patient_characteristics_df[f'{characteristic} {admission_data_df[characteristic].value_counts().idxmax()}'] = [
            admission_data_df[characteristic].value_counts()[0]]
        # get percentage as fraction of non_nan
        # patient_characteristics_df[f'% {characteristic} {admission_data_df[characteristic].value_counts().idxmax()}'] = [admission_data_df[characteristic].value_counts()[0]/admission_data_df[characteristic].count()]
        # get percentage as fraction of total (including missing values)
        patient_characteristics_df[
            f'% {characteristic} {admission_data_df[characteristic].value_counts().idxmax()}'] = [
            admission_data_df[characteristic].value_counts()[0] / len(admission_data_df)]
        patient_characteristics_df[f'n missing {characteristic}'] = [admission_data_df[characteristic].isnull().sum()]

    return patient_characteristics_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--admission_notes_data_path', type=str, required=True)
    parser.add_argument('-t', '--extracted_tables_path', type=str, required=True)
    parser.add_argument('-po', '--preprocessed_outcomes', type=str, required=True)
    args = parser.parse_args()

    extracted_monitoring_data_path = os.path.join(args.extracted_tables_path, 'monitoring_df.csv')
    extracted_admission_table_path = os.path.join(args.extracted_tables_path, 'admission_df.csv')

    patient_characteristics_df = extract_patient_characteristics(args.admission_notes_data_path,
                                                                 extracted_monitoring_data_path,
                                                                 extracted_admission_table_path,
                                                                 args.preprocessed_outcomes)

    patient_characteristics_df.to_csv(
        os.path.join(os.path.dirname(args.preprocessed_outcomes), 'mimic_patient_characteristics.csv'), index=False)
