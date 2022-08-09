import warnings
import os
import numpy as np
import pandas as pd

from preprocessing.utils import remove_french_accents_and_cedillas_from_dataframe, create_ehr_case_identification_column

columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                   'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                   'eds_final_begin', 'eds_final_end', 'eds_final_patient_id',
                   'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                   'date_from', 'date_to', 'patient_id_manual', 'stroke_onset_date', 'Referral', 'match_by']

identification_columns = ['case_admission_id', 'sample_date']

# defining equivalent dosage labels
fibrinogen_equivalent_dosage_labels = ['fibrinogène', 'fibrinogène, antigène']
creatinine_equivalent_dosage_labels = ['créatinine', 'U-créatinine, colorimétrie', 'créatinine, colorimétrie']
hematocrit_equivalent_dosage_labels = ['hématocrite', 'G-Sgc-hématocrite, ABL', 'G-Sgv-hématocrite, ABL',
                                       'Hématocrite, Smart 546', 'G-Sgv-hématocrite', 'hématocrite, pocH-100i',
                                       'G-Sgvm-hématocrite, ABL', 'hématocrite, impédancemétrie',
                                       'G-Sgv-hématocrite, ABL', 'G-Sga-hématocrite, ABL']
potassium_equivalent_dosage_labels = ['potassium', 'G-Sga-potassium, ABL', 'G-Sgv-potassium, ABL', 'Potassium, Piccolo',
                                      'potassium, potentiométrie', 'G-Sgvm-potassium, ABL', 'G-Sgc-potassium, ABL',
                                      'G-Sgv-potassium', 'U-potassium, potentiométrie indirecte']
sodium_equivalent_dosage_labels = ['sodium', 'G-Sga-sodium, ABL', 'G-Sgv-sodium, ABL', 'sodium, potentiométrie',
                                   'Sodium, Piccolo', 'G-Sgvm-sodium, ABL', 'U-sodium, potentiométrie indirecte',
                                   'G-Sgc-sodium, ABL', 'G-Sgv-sodium']
urea_equivalent_dosage_labels = ['urée', 'urée, colorimétrie', 'U-urée, colorimétrie']
hba1c_equivalent_dosage_labels = ['hémoglobine glyquée',
                                  'hémoglobine glyquée (HbA1c), immunologique d\x92agglutination latex']
hemoglobin_equivalent_dosage_labels = ['hémoglobine', 'G-Sga-hémoglobine, ABL', 'G-Sgv-hémoglobine, ABL',
                                       'hémoglobine, pocH-100i', 'hémoglobine, HemoCue 201', 'G-Sgvm-hémoglobine, ABL',
                                       'G-Sgc-hémoglobine, ABL', 'G-Sgv-hémoglobine']
thrombocytes_equivalent_dosage_labels = ['thrombocytes', 'Thrombocytes, pocH-100i']
leucocytes_equivalent_dosage_labels = ['leucocytes', 'Leucocytes, pocH-100i']
erythrocytes_equivalent_dosage_labels = ['érythrocytes', 'érythrocytes, numération, impédancemétrie']
inr_equivalent_dosage_labels = ['INR', 'INR, turbodensitométrie']
crp_equivalent_dosage_labels = ['protéine C-réactive', 'Protéine C-Réactive  (CRP), Piccolo',
                                'protéine C-réactive (CRP), immunoturbidimétrique latex CP',
                                'protéine C-réactive, Smart 546']
glucose_equivalent_dosage_labels = ['glucose', 'G-Sga-glucose, ABL', 'G-Sgv-glucose, ABL', 'Glucose',
                                    'Glucose, Piccolo', 'glucose, PAP', 'G-Sgvm-glucose, ABL', 'G-Sgv-glucose',
                                    'G-Sgc-glucose, ABL', 'U-glucose, PAP colorimétrie']
bilirubine_equivalent_dosage_labels = ['bilirubine totale', 'G-Sga-bilirubine totale, ABL',
                                       'G-Sgv-bilirubine totale, ABL', 'Bilirubine totale, Piccolo',
                                       'bilirubine totale, colorimétrie', 'G-Sgvm-bilirubine totale, ABL']
asat_equivalent_dosage_labels = ['ASAT', 'Aspartate aminotransférase (ASAT), Piccolo',
                                 'aspartate aminotransférase (ASAT), colorimétrie']
alat_equivalent_dosage_labels = ['ALAT', 'Alanine aminotransférase (ALAT), Piccolo',
                                 'alanine aminotransférase (ALAT), colorimétrie']
doac_xa_equivalent_dosage_labels = ['Activité anti-Xa (DOAC)', 'Activité anti-Xa (rivaroxaban)',
                                    'Activité anti-Xa (apixaban)', 'Activité anti-Xa (edoxaban)',
                                    'Activité anti-Xa (Apixaban)']
ldl_equivalent_dosage_labels = ['LDL cholestérol calculé', 'cholestérol non-HDL']

equivalence_lists = [fibrinogen_equivalent_dosage_labels, creatinine_equivalent_dosage_labels,
                     hematocrit_equivalent_dosage_labels,
                     potassium_equivalent_dosage_labels, sodium_equivalent_dosage_labels,
                     urea_equivalent_dosage_labels,
                     hba1c_equivalent_dosage_labels, hemoglobin_equivalent_dosage_labels,
                     thrombocytes_equivalent_dosage_labels,
                     leucocytes_equivalent_dosage_labels, erythrocytes_equivalent_dosage_labels,
                     inr_equivalent_dosage_labels,
                     crp_equivalent_dosage_labels, glucose_equivalent_dosage_labels,
                     bilirubine_equivalent_dosage_labels,
                     asat_equivalent_dosage_labels, alat_equivalent_dosage_labels, doac_xa_equivalent_dosage_labels,
                     ldl_equivalent_dosage_labels]

dosage_labels_to_exclude = ['érythrocytes agglutinés', 'Type d\'érythrocytes', 'Type des érythrocytes',
                            'érythrocytes en rouleaux',
                            'Cristaux cholestérol',
                            'potassium débit', 'urée débit', 'sodium débit', 'glucose débit',
                            'protéine C-réactive, POCT',
                            'activité anti-Xa (HBPM), autre posologie',
                            'activité anti-Xa (HBPM), thérapeutique, 1x /jour']

blood_material_equivalents = ['sga', 'sgv', 'sgvm', 'sgc']
material_to_exclude = ['LCR', 'liqu. pleural', 'épanchement', 'sg cordon', 'liqu. abdo.', 'liqu. ascite', 'liqu.']
non_numerical_values_to_remove = ['ERROR', 'nan', 'SANS RES.', 'Hémolysé', 'sans resultat',
                                  'NON REALISE', 'NON INTERPRÉT.', 'COA', 'TAM']


def preprocess_labs(lab_df: pd.DataFrame, material_to_include: list = ['any_blood'],
                    verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess the labs dataframe
    :param lab_df:
    :param material_to_include: list of materials to include where material is one of the following: 'any_blood', 'urine'
    :param verbose: print preprocessing safety details
    :return:
    """
    lab_df = lab_df.copy()
    lab_df['case_admission_id'] = create_ehr_case_identification_column(lab_df)

    lab_df.drop(columns_to_drop, axis=1, inplace=True)

    lab_names = set([c.split('_')[0] for c in lab_df.columns if c not in identification_columns])
    new_lab_column_headers = set(
        ['_'.join(c.split('_')[1:]) for c in lab_df.columns if c not in identification_columns])

    print('Labs measured:', lab_names)

    # split lab df into individual lab dfs for every lab name
    lab_df_split_by_lab_name = []

    for _, lab_name in enumerate(lab_names):
        selected_columns = identification_columns + [c for c in lab_df.columns if c.split('_')[0] == lab_name]
        individual_lab_df = lab_df[selected_columns].dropna(subset=[f'{lab_name}_value'])
        individual_lab_df.columns = identification_columns + ['_'.join(c.split('_')[1:]) for c in
                                                              individual_lab_df.columns if c.startswith(lab_name)]
        individual_lab_df['lab_name'] = lab_name
        lab_df_split_by_lab_name.append(individual_lab_df)

    reorganised_lab_df = pd.concat(lab_df_split_by_lab_name, ignore_index=True)

    equalized_reorganised_lab_df = reorganised_lab_df.copy()
    for equivalence_list in equivalence_lists:
        equalized_reorganised_lab_df.loc[
            reorganised_lab_df['dosage_label'].isin(equivalence_list[1:]), 'dosage_label'] = equivalence_list[0]

    equalized_reorganised_lab_df = equalized_reorganised_lab_df[
        ~equalized_reorganised_lab_df['dosage_label'].isin(dosage_labels_to_exclude)]

    # check that units correspond
    for dosage_label in equalized_reorganised_lab_df['dosage_label'].unique():
        units_for_dosage_label = \
            equalized_reorganised_lab_df[equalized_reorganised_lab_df['dosage_label'] == dosage_label][
                'unit_of_measure'].unique()
        print(dosage_label, units_for_dosage_label)
        if len(units_for_dosage_label) > 1:
            warnings.warn(f'{dosage_label} has different units: {units_for_dosage_label}')
            raise ValueError(f'{dosage_label} has different units: {units_for_dosage_label}')

    # fixing material equivalents and materials to exclude
    for dosage_label in ['pO2', 'pCO2', 'pH']:
        # for pO2, pCO2 and ph, exclude values with material_label other than 'sga'
        equalized_reorganised_lab_df = equalized_reorganised_lab_df[~equalized_reorganised_lab_df[
            (equalized_reorganised_lab_df['dosage_label'].str.contains(dosage_label)) &
            (equalized_reorganised_lab_df['material_label'] != 'sga')
        ]]

        # raise error if pO2, pCO2 or pH come from arterial and venous blood
        dosage_label_materials = \
            equalized_reorganised_lab_df[equalized_reorganised_lab_df['dosage_label'].str.contains(dosage_label)][
                'material_label'].unique()
        if 'sga' in dosage_label_materials and len(dosage_label_materials) > 1:
            raise ValueError(f'{dosage_label} has arterial and other materials: {dosage_label_materials}')

    equalized_reorganised_lab_df.loc[
        reorganised_lab_df['material_label'].isin(blood_material_equivalents), 'material_label'] = 'any_blood'
    equalized_reorganised_lab_df = equalized_reorganised_lab_df[
        ~equalized_reorganised_lab_df['material_label'].isin(material_to_exclude)]
    equalized_reorganised_lab_df = equalized_reorganised_lab_df[
        equalized_reorganised_lab_df['material_label'].isin(material_to_include)]


    # correct non numeric values
    equalized_reorganised_lab_df = correct_non_numerical_values(equalized_reorganised_lab_df)
    # remove non numerical values in value column
    equalized_reorganised_lab_df = equalized_reorganised_lab_df[
        ~equalized_reorganised_lab_df['value'].isin(non_numerical_values_to_remove)]
    equalized_reorganised_lab_df.dropna(subset=['value'], inplace=True)
    remaining_non_numerical_values = \
    equalized_reorganised_lab_df[pd.to_numeric(equalized_reorganised_lab_df['value'], errors='coerce').isnull()][
        'value'].unique()
    print('Remaining non-numerical values:', remaining_non_numerical_values)
    if len(remaining_non_numerical_values) > 0:
        raise ValueError(f'Remaining non-numerical values: {remaining_non_numerical_values}')
    equalized_reorganised_lab_df['value'] = pd.to_numeric(equalized_reorganised_lab_df['value'], errors='coerce')

    # correct negative values
    # set negative values for dosage label 'hémoglobine' to NaN (NaN values will be removed later)
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'hémoglobine') & (
                equalized_reorganised_lab_df['value'] < 0), 'value'] = np.NAN
    # set negative values for dosage label 'glucose' to NaN (NaN values will be removed later)
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'glucose') & (
                equalized_reorganised_lab_df['value'] < 0), 'value'] = np.NAN
    equalized_reorganised_lab_df.dropna(subset=['value'], inplace=True)
    # warn if negative values are still present
    if len(equalized_reorganised_lab_df[equalized_reorganised_lab_df['value'] < 0]) > 0:
        warnings.warn('Negative values are present. Check data.')

    # remove all french accents and cedillas
    equalized_reorganised_lab_df = remove_french_accents_and_cedillas_from_dataframe(equalized_reorganised_lab_df)

    # restrict to possible value ranges
    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              'possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)
    for variable in possible_value_ranges['variable_label'].dropna().unique():
        possible_value_ranges_for_variable = possible_value_ranges[
            possible_value_ranges['variable_label'] == variable]
        equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == variable)
                                         & (~equalized_reorganised_lab_df['value'].between(
                                             possible_value_ranges_for_variable['Min'].values[0],
                                             possible_value_ranges_for_variable['Max'].values[0])), 'value'] = np.NAN
    if verbose:
        print(f'Excluding {equalized_reorganised_lab_df["value"].isna().sum()} observations because out of range')
    equalized_reorganised_lab_df.dropna(subset=['value'], inplace=True)


    # get mean number of values per dosage label patient admission id
    median_observations_per_case_admission_id = \
        equalized_reorganised_lab_df.groupby(['case_admission_id', 'dosage_label'])['value'].count().reset_index()

    if verbose:
        print(median_observations_per_case_admission_id.groupby('dosage_label').median())
        print(equalized_reorganised_lab_df.groupby('dosage_label')['value'].describe())

    return equalized_reorganised_lab_df


def correct_non_numerical_values(equalized_reorganised_lab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct non-numerical values in the labs dataframe
    - For < and > signs, replace with + and - 0.05 of the value
    - remove apostrophes
    :param equalized_reorganised_lab_df:
    :return: df with corrected values
    """
    # replace apostrophes with empty string in value column
    equalized_reorganised_lab_df['value'] = equalized_reorganised_lab_df['value'].str.replace("'", "")

    # for > and < signs, add or subtract 5% of max/min value
    # replace >83.2 value if dosage label is 'glucose'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'glucose') & (
            equalized_reorganised_lab_df['value'] == '>83.2'), 'value'] = 83.2 + 0.05 * 83.2
    # replace >70000 value if dosage label is 'proBNP'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'proBNP') & (
            equalized_reorganised_lab_df['value'] == '>70000'), 'value'] = 70000 + 0.05 * 70000
    # replace >474 value if dosage label is 'Activité anti-Xa (DOAC)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'Activité anti-Xa (DOAC)') & (
            equalized_reorganised_lab_df['value'] == '>474'), 'value'] = 474 + 0.05 * 474
    # replace >445 value if dosage label is 'Activité anti-Xa (DOAC)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'Activité anti-Xa (DOAC)') & (
            equalized_reorganised_lab_df['value'] == '>445'), 'value'] = 445 + 0.05 * 445
    # replace >160.0 value if dosage label is 'PTT'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'PTT') & (
            equalized_reorganised_lab_df['value'] == '>160.0'), 'value'] = 160.0 + 0.05 * 160.0
    # replace >11.00 value if dosage label is 'INR'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'INR') & (
            equalized_reorganised_lab_df['value'] == '>11.00'), 'value'] = 11.00 + 0.05 * 11.00
    # replace >11.00 and >11.0 value if dosage label is 'fibrinogène'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'fibrinogène') & (
            equalized_reorganised_lab_df['value'] == '>11.00'), 'value'] = 11.00 + 0.05 * 11.00
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'fibrinogène') & (
            equalized_reorganised_lab_df['value'] == '>11.0'), 'value'] = 11.0 + 0.05 * 11.0
    # replace >1.60 value if dosage label is 'activité anti-Xa (HBPM), thérapeutique, 2x /jour'
    equalized_reorganised_lab_df.loc[
        (equalized_reorganised_lab_df['dosage_label'] == 'activité anti-Xa (HBPM), thérapeutique, 2x /jour') & (
                equalized_reorganised_lab_df['value'] == '>1.60'), 'value'] = 1.60 + 0.05 * 1.60
    # replace >1.31 value if dosage label is 'activité anti-Xa (HNF)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'activité anti-Xa (HNF)') & (
            equalized_reorganised_lab_df['value'] == '>1.31'), 'value'] = 1.31 + 0.05 * 1.31
    # replace >1.20 value if dosage label is 'activité anti-Xa (HNF)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'activité anti-Xa (HNF)') & (
            equalized_reorganised_lab_df['value'] == '>1.20'), 'value'] = 1.20 + 0.05 * 1.20

    # replace <5 value if dosage label is 'ALAT'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'ALAT') & (
            equalized_reorganised_lab_df['value'] == '<5'), 'value'] = 5 - 0.05 * 5
    # replace <3 value if dosage label is bilirubine totale
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'bilirubine totale') & (
            equalized_reorganised_lab_df['value'] == '<3'), 'value'] = 3 - 0.05 * 3
    # replace <10 value if dosage label is 'Activité anti-Xa (DOAC)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'Activité anti-Xa (DOAC)') & (
            equalized_reorganised_lab_df['value'] == '<10'), 'value'] = 10 - 0.05 * 10
    # replace <1.00 if dosage label is 'INR'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'INR') & (
            equalized_reorganised_lab_df['value'] == '<1.00'), 'value'] = 1.00 - 0.05 * 1.00
    # replace <0.7, <0.5, <0.4 if dosage label is 'fibrinogène'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'fibrinogène') & (
            equalized_reorganised_lab_df['value'] == '<0.7'), 'value'] = 0.7 - 0.05 * 0.7
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'fibrinogène') & (
            equalized_reorganised_lab_df['value'] == '<0.5'), 'value'] = 0.5 - 0.05 * 0.5
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'fibrinogène') & (
            equalized_reorganised_lab_df['value'] == '<0.4'), 'value'] = 0.4 - 0.05 * 0.4
    # replace <0.30 if label is 'protéine C-réactive'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'protéine C-réactive') & (
            equalized_reorganised_lab_df['value'] == '<0.30'), 'value'] = 0.30 - 0.05 * 0.30

    return equalized_reorganised_lab_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess labs from separate lab files in data folder')
    parser.add_argument('data_path')
    parser.add_argument('-o', '--output_dir', help='Directory to save output', required=False, default=None)
    parser.add_argument('-m', '--material_to_include', help='Material to include', required=False,
                        default=['any_blood'])
    args = parser.parse_args()
    lab_file_start = 'labo'
    lab_files = [pd.read_csv(os.path.join(args.data_path, f), delimiter=';', encoding='utf-8')
                 for f in os.listdir(args.data_path)
                 if f.startswith(lab_file_start)]
    lab_df = pd.concat(lab_files, ignore_index=True)
    preprocessed_lab_df = preprocess_labs(lab_df, material_to_include=args.material_to_include)
    if args.output_dir is not None:
        preprocessed_lab_df.to_csv(os.path.join(args.output_dir, 'preprocessed_labs.csv'), index=False,
                                   encoding='utf-8')
