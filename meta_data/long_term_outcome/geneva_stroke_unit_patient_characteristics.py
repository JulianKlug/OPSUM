import os.path
import numpy as np
import pandas as pd

from preprocessing.geneva_stroke_unit_preprocessing.utils import create_registry_case_identification_column

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


def outcome_preprocessing(df: pd.DataFrame):
    # if death in hospital, set mRs to 6
    df.loc[df['Death in hospital'] == 'yes', '3M mRS'] = 6
    # if 3M Death and 3M mRS nan, set mrs to 6
    df.loc[(df['3M Death'] == 'yes') & (df['3M mRS'].isna()), '3M mRS'] = 6

    # if death in hospital set 3M Death to yes
    df.loc[df['Death in hospital'] == 'yes', '3M Death'] = 'yes'
    # if 3M mRs == 6, set 3M Death to yes
    df.loc[df['3M mRS'] == 6, '3M Death'] = 'yes'
    # if 3M mRs not nan and not 6, set 3M Death to no
    df.loc[(df['3M mRS'] != 6) &
                                      (~df['3M mRS'].isna())
                                      & (df['3M Death'].isna()), '3M Death'] = 'no'

    return df


def extract_patient_characteristics(case_admission_ids: pd.DataFrame, stroke_registry_data_path: str,
                                    continuous_characteristics: list = CONTINUOUS_CHARACTERISTICS,
                                    categorical_characteristics: list = CATEGORICAL_CHARACTERISTICS) -> pd.DataFrame:
    """
    Extracts patient characteristics from the stroke registry data.
    :param patient_id_path: path to the patient id file to select the patients to extract the characteristics from (is produced during training as pid_train.tsv / pid_test.tsv).
    :param stroke_registry_data_path: path to the stroke registry data (after post-hoc modification).
    :param continuous_characteristics: list of continuous characteristics to extract.
    :param categorical_characteristics: list of categorical characteristics to extract.
    :return: a dataframe with the patient characteristics.
    """
    # load stroke registry data
    stroke_registry_df = pd.read_excel(stroke_registry_data_path)
    stroke_registry_df['patient_id'] = stroke_registry_df['Case ID'].apply(lambda x: x[8:-4]).astype(str)
    stroke_registry_df['case_admission_id'] = create_registry_case_identification_column(stroke_registry_df)

    # preprocess outcome variables
    stroke_registry_df = outcome_preprocessing(stroke_registry_df)

    # select patients to extract characteristics from
    selected_stroke_registry_df = stroke_registry_df[stroke_registry_df['case_admission_id'].isin(case_admission_ids)]
    selected_stroke_registry_df.drop_duplicates(subset=['case_admission_id'], inplace=True)

    patient_characteristics_df = pd.DataFrame()

    patient_characteristics_df['n admissions'] = [len(case_admission_ids)]

    # extract continuous characteristics
    for characteristic in continuous_characteristics:
        patient_characteristics_df[f'median {characteristic}'] = [selected_stroke_registry_df[characteristic].median()]
        patient_characteristics_df[f'Q25 {characteristic}'] = [selected_stroke_registry_df[characteristic].quantile(0.25)]
        patient_characteristics_df[f'Q75 {characteristic}'] = [selected_stroke_registry_df[characteristic].quantile(0.75)]
        # count number of missing values for characteristic
        patient_characteristics_df[f'n missing {characteristic}'] = [selected_stroke_registry_df[characteristic].isnull().sum()]

    for characteristic in categorical_characteristics:
        # get number of most common value for each categorical characteristic
        patient_characteristics_df[f'{characteristic} {selected_stroke_registry_df[characteristic].value_counts().idxmax()}'] = [selected_stroke_registry_df[characteristic].value_counts()[0]]
        # get percentage as fraction of non_nan
        # patient_characteristics_df[f'% {characteristic} {selected_stroke_registry_df[characteristic].value_counts().idxmax()}'] = [selected_stroke_registry_df[characteristic].value_counts()[0]/selected_stroke_registry_df[characteristic].count()]
        # get percentage as fraction of total (including missing values)
        patient_characteristics_df[f'% {characteristic} {selected_stroke_registry_df[characteristic].value_counts().idxmax()}'] = [selected_stroke_registry_df[characteristic].value_counts()[0]/len(selected_stroke_registry_df)]
        patient_characteristics_df[f'n missing {characteristic}'] = [selected_stroke_registry_df[characteristic].isnull().sum()]




    return patient_characteristics_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--test_patient_id_path', type=str, default='../pid_test.tsv')
    parser.add_argument('-p2', '--train_patient_id_path', type=str, default='../pid_train.tsv')
    parser.add_argument('-s', '--stroke_registry_data_path', type=str, default='../stroke_registry_post_hoc_modified.xlsx')
    parser.add_argument('-d', '--features_path', type=str, default='../preprocessed_features_01012023_233050.csv')
    parser.add_argument('-O', '--outcomes_path', type=str, default='../preprocessed_outcomes_01012023_233050.csv')
    parser.add_argument('-o', '--output_path', type=str, default='.')
    parser.add_argument('--outcome', type=str, default='3M Death')
    args = parser.parse_args()

    data_df = pd.read_csv(args.features_path)
    outcomes_df = pd.read_csv(args.outcomes_path)
    data_df['pid'] = data_df['case_admission_id'].apply(lambda x: x.split('_')[0])
    outcomes_df['pid'] = outcomes_df['case_admission_id'].apply(lambda x: x.split('_')[0])

    test_patient_id_df = pd.read_csv(args.test_patient_id_path, sep='\t')
    test_patient_id_df['patient_id'] = test_patient_id_df['patient_id'].astype(str)

    admissions_test_set = outcomes_df[
        (outcomes_df.case_admission_id.isin(data_df.case_admission_id.unique()))
        & (~outcomes_df[args.outcome].isnull())
        & (outcomes_df.pid.isin(test_patient_id_df['patient_id']))].case_admission_id.unique()

    train_patient_id_df = pd.read_csv(args.train_patient_id_path, sep='\t')
    train_patient_id_df['patient_id'] = train_patient_id_df['patient_id'].astype(str)

    admissions_train_set = outcomes_df[
        (outcomes_df.case_admission_id.isin(data_df.case_admission_id.unique()))
        & (~outcomes_df[args.outcome].isnull())
        & (outcomes_df.pid.isin(train_patient_id_df['patient_id']))].case_admission_id.unique()

    all_admissions = np.concatenate([admissions_test_set, admissions_train_set])

    overall_patient_characteristics_df = extract_patient_characteristics(all_admissions, args.stroke_registry_data_path)
    overall_patient_characteristics_df.to_csv(os.path.join(args.output_path,
                                                   f'patient_characteristics_overall.tsv'), sep='\t', index=False)

    test_patient_characteristics_df = extract_patient_characteristics(admissions_test_set, args.stroke_registry_data_path)
    test_patient_characteristics_df.to_csv(os.path.join(args.output_path,
                                                    f'patient_characteristics_test.tsv'), sep='\t', index=False)

    train_patient_characteristics_df = extract_patient_characteristics(admissions_train_set, args.stroke_registry_data_path)
    train_patient_characteristics_df.to_csv(os.path.join(args.output_path,
                                                    f'patient_characteristics_train.tsv'), sep='\t', index=False)










