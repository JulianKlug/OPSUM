import pandas as pd

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


def extract_patient_characteristics(patient_id_path: str, stroke_registry_data_path: str,
                                    continuous_characteristics: list = CONTINUOUS_CHARACTERISTICS,
                                    categorical_characteristics: list = CATEGORICAL_CHARACTERISTICS) -> pd.DataFrame:
    """
    Extracts patient characteristics from the stroke registry data.
    :param patient_id_path: path to the patient id file to select the patients to extract the characteristics from.
    :param stroke_registry_data_path: path to the stroke registry data (after post-hoc modification).
    :param continuous_characteristics: list of continuous characteristics to extract.
    :param categorical_characteristics: list of categorical characteristics to extract.
    :return: a dataframe with the patient characteristics.
    """
    # load stroke registry data
    stroke_registry_df = pd.read_excel(stroke_registry_data_path)
    stroke_registry_df['patient_id'] = stroke_registry_df['Case ID'].apply(lambda x: x[8:-4]).astype(str)

    # only keep firs admission for duplicate patients
    stroke_registry_df = stroke_registry_df.drop_duplicates(subset=['patient_id'], keep='first')

    patient_id_df = pd.read_csv(patient_id_path, sep='\t')
    patient_id_df['patient_id'] = patient_id_df['patient_id'].astype(str)

    # select patients to extract characteristics from
    selected_stroke_registry_df = stroke_registry_df[stroke_registry_df['patient_id'].isin(patient_id_df['patient_id'])]

    patient_characteristics_df = pd.DataFrame()

    patient_characteristics_df['n patients'] = [len(selected_stroke_registry_df)]

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
    patient_id_path = ''
    stroke_registry_data_path = ''

    patient_characteristics_df = extract_patient_characteristics(patient_id_path, stroke_registry_data_path)
    patient_characteristics_df.to_csv('', sep='\t', index=False)

    print()









