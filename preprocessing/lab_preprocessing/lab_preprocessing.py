import warnings
import os
import numpy as np
import pandas as pd

from preprocessing.patient_selection.filter_ehr_patients import filter_ehr_patients
from preprocessing.utils import remove_french_accents_and_cedillas_from_dataframe, create_ehr_case_identification_column
from preprocessing.variable_assembly.variable_selection import restrict_to_selected_variables

columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                   'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                   'eds_final_begin', 'eds_final_end', 'eds_final_patient_id',
                   'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                   'date_from', 'date_to', 'patient_id_manual', 'stroke_onset_date', 'Referral', 'match_by',
                   'multiple_id']

identification_columns = ['case_admission_id', 'sample_date']

# defining equivalent dosage labels
equivalent_labels_path = os.path.join(os.path.dirname(__file__), 'equivalent_labels.csv')
# defining desired units of measure
dosage_units_path = os.path.join(os.path.dirname(__file__), 'dosage_units.csv')
# defining selected variables path (path to the variable selection file)
selected_variables_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'variable_assembly/selected_variables.xlsx')

blood_material_equivalents = ['sga', 'sgv', 'sgvm', 'sgc', 'sgv ponction', 'sgv cathéter', 'sga cathéter', 'cathéter artériel', 'cathéter veineux', 'plasma', 'Sang', 'sg cordon',
                              'sgv catheter', 'sga catheter', 'catheter arteriel', 'catheter veineux']

unit_of_measure_equivalents = [['UI/ml', 'U/ml']]

non_numerical_values_to_remove = ['ERROR', 'nan', 'SANS RES.', 'Hémolysé', 'Hemolyse', 'sans resultat','voir POCT',
                                  'NON REALISE', 'NON INTERPRÉT.', 'NON INTERPRET.', 'COA', 'TAM', '****.**', '-100000.0', '----']


def preprocess_labs(lab_df: pd.DataFrame, material_to_include: list = ['any_blood'], log_dir:str = '',
                    verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess the labs dataframe
    :param lab_df:
    :param material_to_include: list of materials to include where material is one of the following: 'any_blood', 'urine'
    :param log_dir: directory where to save the log file
    :param verbose: print preprocessing safety details
    :return:
    """
    lab_df = lab_df.copy()
    lab_df['case_admission_id'] = create_ehr_case_identification_column(lab_df)

    lab_df.drop(columns_to_drop, axis=1, inplace=True)

    # After reorganisation of df, final columns are:
    target_structure_columns = ['case_admission_id','sample_date','analyse_label','dosage_label','material_label','unit_of_measure','value','lower_limit','upper_limit']
    # when coming from another structure, there will be an extra lab_name column

    # verify that target_structure_columns is a subset of lab_df.columns
    if set(target_structure_columns).issubset(lab_df.columns):
        reorganised_lab_df = lab_df

    else:
        # Reorganize from individual lab columns to a structure with common columns
        lab_names = set([c.split('_')[0] for c in lab_df.columns if c not in identification_columns])
        print('Labs measured:', lab_names)

        # split lab df into individual lab dfs for every lab name
        lab_df_split_by_lab_name = []

        for lab_name in lab_names:
            selected_columns = identification_columns + [c for c in lab_df.columns if c.split('_')[0] == lab_name]
            individual_lab_df = lab_df[selected_columns].dropna(subset=[f'{lab_name}_value'])
            individual_lab_df.columns = identification_columns + ['_'.join(c.split('_')[1:]) for c in
                                                                  individual_lab_df.columns if c.startswith(lab_name)]
            individual_lab_df['lab_name'] = lab_name
            lab_df_split_by_lab_name.append(individual_lab_df)

        reorganised_lab_df = pd.concat(lab_df_split_by_lab_name, ignore_index=True)

    equalized_reorganised_lab_df = reorganised_lab_df

    ### PROCESSING LABELS ###
    # Align equivalent labels to a unified label
    equivalent_labels_df = pd.read_csv(equivalent_labels_path)
    for column in equivalent_labels_df.columns:
        equivalence_list = equivalent_labels_df[f'{column}'].dropna().values
        equalized_reorganised_lab_df.loc[
            equalized_reorganised_lab_df['dosage_label'].isin(equivalence_list[1:]), 'dosage_label'] = equivalence_list[
            0]

    # remove all French accents and cedillas
    equalized_reorganised_lab_df = remove_french_accents_and_cedillas_from_dataframe(equalized_reorganised_lab_df)

    # Restrict to selected variables
    selected_variables = pd.read_excel(selected_variables_path)['included']
    dropped_dosage_labels = equalized_reorganised_lab_df[~equalized_reorganised_lab_df.dosage_label.isin(selected_variables)].dosage_label.unique()
    equalized_reorganised_lab_df = equalized_reorganised_lab_df.drop(
        equalized_reorganised_lab_df[~equalized_reorganised_lab_df.dosage_label.isin(selected_variables)].index)
    included_dosage_labels = equalized_reorganised_lab_df.dosage_label.unique()

    ### PROCESSING MATERIALS ###
    # Retain only selected materials
    equalized_reorganised_lab_df.loc[
        reorganised_lab_df['material_label'].isin(blood_material_equivalents), 'material_label'] = 'any_blood'
    excluded_material = list(set(equalized_reorganised_lab_df.material_label.unique()) - set(material_to_include))

    equalized_reorganised_lab_df = equalized_reorganised_lab_df[
        equalized_reorganised_lab_df['material_label'].isin(material_to_include)]

    # fixing material equivalents and materials to exclude
    for dosage_label in ['pO2', 'pCO2', 'pH']:
        # for pO2, pCO2 and ph, exclude values with material_label other than 'sga'
        equalized_reorganised_lab_df = equalized_reorganised_lab_df.drop(
            equalized_reorganised_lab_df[
                (equalized_reorganised_lab_df['dosage_label'].str.contains(dosage_label)) &
                (equalized_reorganised_lab_df['material_label'] != 'sga')
                ].index)

        # raise error if pO2, pCO2 or pH come from arterial and venous blood
        dosage_label_materials = \
            equalized_reorganised_lab_df[equalized_reorganised_lab_df['dosage_label'].str.contains(dosage_label)][
                'material_label'].unique()
        if 'sga' in dosage_label_materials and len(dosage_label_materials) > 1:
            raise ValueError(f'{dosage_label} has arterial and other materials: {dosage_label_materials}')

    ### PROCESSING UNITS OF MEASURE ###
    # Convert equivalent units of measure to a unified unit of measure
    for units_equivalence_list in unit_of_measure_equivalents:
        equalized_reorganised_lab_df.loc[
            equalized_reorganised_lab_df['unit_of_measure'].isin(units_equivalence_list[1:]), 'unit_of_measure'] = \
            units_equivalence_list[0]

    # Only retain certain units for materials with multiple units
    dosage_units_df = pd.read_csv(dosage_units_path)
    for dosage_label in dosage_units_df.columns:
        equalized_reorganised_lab_df.drop(
            equalized_reorganised_lab_df[
                  (equalized_reorganised_lab_df['dosage_label'] == dosage_label)
                  & (~equalized_reorganised_lab_df.unit_of_measure.isin(dosage_units_df[dosage_label]))
                  ].index, inplace=True)

    # check that units correspond
    for dosage_label in equalized_reorganised_lab_df['dosage_label'].unique():
        units_for_dosage_label = \
            equalized_reorganised_lab_df[equalized_reorganised_lab_df['dosage_label'] == dosage_label][
                'unit_of_measure'].unique()
        print(dosage_label, units_for_dosage_label)
        if len(units_for_dosage_label) > 1:
            warnings.warn(f'{dosage_label} has different units: {units_for_dosage_label}')
            raise ValueError(f'{dosage_label} has different units: {units_for_dosage_label}')


    ### PROCESSING VALUES ###
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
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['hémoglobine', 'hemoglobine'])) & (
                equalized_reorganised_lab_df['value'] < 0), 'value'] = np.NAN
    # set negative values for dosage label 'glucose' to NaN (NaN values will be removed later)
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'glucose') & (
                equalized_reorganised_lab_df['value'] < 0), 'value'] = np.NAN
    equalized_reorganised_lab_df.dropna(subset=['value'], inplace=True)
    # warn if negative values are still present (except base deficit)
    if len(equalized_reorganised_lab_df[(equalized_reorganised_lab_df['value'] < 0)
                                        & (~equalized_reorganised_lab_df.dosage_label.str.contains('cBase'))]) > 0:
        warnings.warn('Negative values are present. Check data.')


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

    n_observations_out_ouf_range = equalized_reorganised_lab_df["value"].isna().sum()
    if verbose:
        print(f'Excluding {n_observations_out_ouf_range} observations because out of range')
    equalized_reorganised_lab_df.dropna(subset=['value'], inplace=True)


    # get mean number of values per dosage label patient admission id
    median_observations_per_case_admission_id = \
        equalized_reorganised_lab_df.groupby(['case_admission_id', 'dosage_label'])['value'].count().reset_index()
    median_observations_per_case_admission_id_df = median_observations_per_case_admission_id.groupby('dosage_label').median()
    median_observations_per_case_admission_id_df.rename(columns={'value': 'median_observations_per_case_admission_id'}, inplace=True)
    descriptive_stats_df = equalized_reorganised_lab_df.groupby('dosage_label')['value'].describe()

    if verbose:
        print('Median observations per case admission id:')
        print(median_observations_per_case_admission_id_df)
        print('Descriptive statistics:')
        print(descriptive_stats_df)

    if log_dir != '':
        data_to_log = [included_dosage_labels, dropped_dosage_labels,
                       material_to_include, excluded_material,
                       [n_observations_out_ouf_range]
                       ]
        log_columns = ['included_dosage_labels', 'dropped_dosage_labels',
                      'included_material', 'excluded_material',
                      'n_observations_out_ouf_range']
        log_dataframe = pd.DataFrame(data_to_log).T
        log_dataframe.columns = log_columns
        log_dataframe = pd.concat([log_dataframe, possible_value_ranges], axis=1)
        log_dataframe.to_csv(os.path.join(log_dir, 'lab_preprocessing_log.csv'), index=False)

        median_observations_per_case_admission_id_df.to_csv(os.path.join(log_dir, 'median_observations_per_case_admission_id.csv'), index=True)
        descriptive_stats_df.to_csv(os.path.join(log_dir, 'descriptive_stats.csv'), index=True)




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
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['Activité anti-Xa (DOAC)', 'Activite anti-Xa (DOAC)'])) & (
            equalized_reorganised_lab_df['value'] == '>474'), 'value'] = 474 + 0.05 * 474
    # replace >445 value if dosage label is 'Activité anti-Xa (DOAC)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['Activité anti-Xa (DOAC)', 'Activite anti-Xa (DOAC)'])) & (
            equalized_reorganised_lab_df['value'] == '>445'), 'value'] = 445 + 0.05 * 445
    # replace >160.0 value if dosage label is 'PTT'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'PTT') & (
            equalized_reorganised_lab_df['value'] == '>160.0'), 'value'] = 160.0 + 0.05 * 160.0
    # replace >11.00 value if dosage label is 'INR'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'INR') & (
            equalized_reorganised_lab_df['value'] == '>11.00'), 'value'] = 11.00 + 0.05 * 11.00
    # replace >11.00 and >11.0 value if dosage label is 'fibrinogène'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['fibrinogène', 'fibrinogene'])) & (
            equalized_reorganised_lab_df['value'] == '>11.00'), 'value'] = 11.00 + 0.05 * 11.00
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['fibrinogène', 'fibrinogene'])) & (
            equalized_reorganised_lab_df['value'] == '>11.0'), 'value'] = 11.0 + 0.05 * 11.0
    # replace >1.60 value if dosage label is 'activité anti-Xa (HBPM), thérapeutique, 2x /jour'
    equalized_reorganised_lab_df.loc[
        (equalized_reorganised_lab_df['dosage_label'].isin(['activité anti-Xa (HBPM), thérapeutique, 2x /jour', 'activite anti-Xa (HBPM), therapeutique, 2x /jour'])) & (
                equalized_reorganised_lab_df['value'] == '>1.60'), 'value'] = 1.60 + 0.05 * 1.60
    # replace >1.31 value if dosage label is 'activité anti-Xa (HNF)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['activité anti-Xa (HNF)', 'activite anti-Xa (HNF)'])) & (
            equalized_reorganised_lab_df['value'] == '>1.31'), 'value'] = 1.31 + 0.05 * 1.31
    # replace >1.20 value if dosage label is 'activité anti-Xa (HNF)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['activité anti-Xa (HNF)', 'activite anti-Xa (HNF)'])) & (
            equalized_reorganised_lab_df['value'] == '>1.20'), 'value'] = 1.20 + 0.05 * 1.20
    # replace >180 value if dosage label is 'sodium'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'sodium') & (
            equalized_reorganised_lab_df['value'] == '>180'), 'value'] = 180 + 0.05 * 180
    # replace >200.00 value if dosage label is 'protéine C-réactive'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['protéine C-réactive', 'proteine C-reactive'])) & (
            equalized_reorganised_lab_df['value'] == '>200.00'), 'value'] = 200.00 + 0.05 * 200.00

    # replace <5 value if dosage label is 'ALAT'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'ALAT') & (
            equalized_reorganised_lab_df['value'] == '<5'), 'value'] = 5 - 0.05 * 5
    # replace <3 value if dosage label is bilirubine totale
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'bilirubine totale') & (
            equalized_reorganised_lab_df['value'] == '<3'), 'value'] = 3 - 0.05 * 3
    # replace <10 value if dosage label is 'Activité anti-Xa (DOAC)'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['Activité anti-Xa (DOAC)', 'Activite anti-Xa (DOAC)'])) & (
            equalized_reorganised_lab_df['value'] == '<10'), 'value'] = 10 - 0.05 * 10
    # replace <1.00 if dosage label is 'INR'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'] == 'INR') & (
            equalized_reorganised_lab_df['value'] == '<1.00'), 'value'] = 1.00 - 0.05 * 1.00
    # replace <0.7, <0.5, <0.4 if dosage label is 'fibrinogène'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['fibrinogène', 'fibrinogene'])) & (
            equalized_reorganised_lab_df['value'] == '<0.7'), 'value'] = 0.7 - 0.05 * 0.7
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['fibrinogène', 'fibrinogene'])) & (
            equalized_reorganised_lab_df['value'] == '<0.5'), 'value'] = 0.5 - 0.05 * 0.5
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['fibrinogène', 'fibrinogene'])) & (
            equalized_reorganised_lab_df['value'] == '<0.4'), 'value'] = 0.4 - 0.05 * 0.4
    # replace <0.30 if label is 'protéine C-réactive'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['protéine C-réactive', 'proteine C-reactive'])) & (
            equalized_reorganised_lab_df['value'] == '<0.30'), 'value'] = 0.30 - 0.05 * 0.30
    # replace '<0.08' if label is 'cholestérol HDL'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['cholestérol HDL', 'cholesterol HDL'])) & (
        equalized_reorganised_lab_df['value'] == '<0.08'), 'value'] = 0.08 - 0.05 * 0.08
    # replace '<18' if label is 'créatinine'
    equalized_reorganised_lab_df.loc[(equalized_reorganised_lab_df['dosage_label'].isin(['créatinine', 'creatinine'])) & (
        equalized_reorganised_lab_df['value'] == '<18'), 'value'] = 18 - 0.05 * 18

    return equalized_reorganised_lab_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess labs from separate lab files in data folder')
    parser.add_argument('data_path')
    parser.add_argument('-o', '--output_dir', help='Directory to save output', required=False, default=None)
    parser.add_argument('-m', '--material_to_include', help='Material to include', required=False,
                        default=['any_blood'])
    parser.add_argument('-p', '--patient_selection_path', help='Path to patient selection file', required=False, default='')
    args = parser.parse_args()
    lab_file_start = 'labo'
    lab_files = [pd.read_csv(os.path.join(args.data_path, f), delimiter=';', encoding='utf-8', dtype=str)
                 for f in os.listdir(args.data_path)
                 if f.startswith(lab_file_start)]
    lab_df = pd.concat(lab_files, ignore_index=True)
    lab_df = filter_ehr_patients(lab_df, args.patient_selection_path)
    preprocessed_lab_df = preprocess_labs(lab_df, material_to_include=args.material_to_include, log_dir=args.output_dir)
    if args.output_dir is not None:
        preprocessed_lab_df.to_csv(os.path.join(args.output_dir, 'preprocessed_labs.csv'), index=False,
                                   encoding='utf-8')
