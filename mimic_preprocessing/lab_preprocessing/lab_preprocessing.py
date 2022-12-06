import pandas as pd
from tqdm import tqdm
import os
import numpy as np


def preprocess_labs(lab_df: pd.DataFrame, log_dir:str = '', verbose: bool = True) -> pd.DataFrame:
    """
    Preprocess the labs dataframe
    - Align to unified equivalent label
    - Create the corrected calcium label
    - restrict to selected variables
    - convert non-numerical values
    - convert to target units of measure
    - restrict to plausible range
    - log descriptive stats

    :param lab_df:
    :param log_dir: directory where to save the log file
    :param verbose: print preprocessing safety details
    :return: preprocessed labs dataframe
    """

    selected_reference_values = pd.read_excel('./selected_lab_values.xlsx')

    ## ALIGN LABEL NAMES TO DPI LABEL NAMES
    aligned_lab_df = lab_df.copy()
    for _, row in selected_reference_values.iterrows():
        if pd.isna(row.MIMIC_equivalent_name):
            if row.DPI_name == 'calcium corrige':
                continue
            else:
                raise ValueError(f'No MIMIC equivalent for {row.DPI_name}')
        equivalence_list = pd.DataFrame([row.MIMIC_equivalent_name, row.other_MIMIC_equivalents]).dropna()[0].tolist()
        aligned_lab_df.loc[
            aligned_lab_df['label'].isin(equivalence_list), 'label'] = row.DPI_name

    ## CREATE A VARIABLE FOR CORRECTED CALCIUM
    # -> search for albumin and calcium drawn at same time
    calcium_components_df = aligned_lab_df[aligned_lab_df.label.isin(['Calcium, Total', 'Albumin'])]
    calcium_components_df['hadm_id_charttime'] = calcium_components_df['hadm_id'].astype(str) + '_' + \
                                                 calcium_components_df['charttime'].astype(str)
    calcium_components_df = calcium_components_df.drop_duplicates()[
        (calcium_components_df.drop_duplicates().duplicated(subset=['hadm_id_charttime'], keep=False))]

    corrected_calcium_df = calcium_components_df[calcium_components_df.label == 'Calcium, Total'].copy()
    corrected_calcium_df['corrected_value'] = corrected_calcium_df['valuenum']
    for index, row in tqdm(corrected_calcium_df.iterrows()):
        # Formula: adjusted [Ca](mmol/L) = total [Ca](mmol/L) + 0.02 (40 - [albumin](g/L))
        # Conversion factors:
        # - calcium (mg/dl -> mmol/L) : *0.2495
        # - albumin (g/dL -> g/L): *10
        simultaneous_albumin = calcium_components_df[
            (calcium_components_df.label == 'Albumin') & (calcium_components_df.charttime == row.charttime)]
        corrected_calcium_df.at[index, 'corrected_value'] = row['corrected_value'] * 0.2495 + 0.02 * (
                    40 - simultaneous_albumin.valuenum.values[0] * 10)

    corrected_calcium_df['corrected_valueuom'] = 'mmol/l'
    corrected_calcium_df['label'] = 'calcium corrige'
    corrected_calcium_df['value'] = corrected_calcium_df['corrected_value']
    corrected_calcium_df['valuenum'] = corrected_calcium_df['corrected_value']
    corrected_calcium_df['valueuom'] = corrected_calcium_df['corrected_valueuom']
    corrected_calcium_df.drop(columns=['corrected_value', 'corrected_valueuom', 'hadm_id_charttime'], inplace=True)
    aligned_lab_df = pd.concat([aligned_lab_df, corrected_calcium_df])

    ## RESTRICT TO SELECTED VARIABLES
    selected_lab_labels = selected_reference_values['DPI_name'].tolist()
    restricted_lab_df = aligned_lab_df.copy()
    restricted_lab_df = restricted_lab_df[restricted_lab_df.label.isin(selected_lab_labels)]
    assert restricted_lab_df.label.unique().tolist().sort() == selected_lab_labels.sort()

    ## CONVERT NON NUMERICAL VALUES
    # - ">" & "greater than" are replaced by value + 5% of value
    non_interpretable_non_numerical_values = ['NEG', 'TR', 'UNABLE TO REPORT', 'ERROR', 'UNABLE TO PERFORM']

    restricted_lab_df.loc[restricted_lab_df['value'].str.contains(">", case=False, na=False), ['value', 'valuenum']] = \
        restricted_lab_df[restricted_lab_df['value'].str.contains(">", case=False, na=False)] \
            .apply(lambda row: float(row['value'].split('>')[1]) * 1.05, axis=1)

    restricted_lab_df.loc[
        restricted_lab_df['value'].str.contains("greater than ", case=False, na=False), ['value', 'valuenum']] = \
        restricted_lab_df[restricted_lab_df['value'].str.contains("greater than ", case=False, na=False)] \
            .apply(lambda row: float(row['value'].split('GREATER THAN ')[1]) * 1.05, axis=1)

    restricted_lab_df.loc[
        restricted_lab_df['value'].str.contains("IS HIGHEST MEASURED ", case=False, na=False), ['value', 'valuenum']] = \
        restricted_lab_df[restricted_lab_df['value'].str.contains("IS HIGHEST MEASURED ", case=False, na=False)] \
            .apply(lambda row: float(row['value'].split('IS HIGHEST MEASURED ')[0]) * 1.05, axis=1)

    if verbose:
        print(
            f'Excluding {len(restricted_lab_df[(restricted_lab_df.valuenum.isna()) & (~restricted_lab_df.value.isna())])} values because non-numerical')

    restricted_lab_df.drop(
        restricted_lab_df[(restricted_lab_df.valuenum.isna()) & (~restricted_lab_df.value.isna())].index, inplace=True)


    ## CONVERT UNITS
    print('Converting units')
    # Not perfect in terms of efficiency or conciseness, but no bottleneck + works + readable (-> leave like this for now)
    for _, row in tqdm(selected_reference_values[selected_reference_values.unit_conversion_needed == 1].iterrows()):
        restricted_lab_df.valuenum = restricted_lab_df.apply(lambda X: X.valuenum * row['multiplicative_factor']
        if X.label == row['DPI_name'] else X.valuenum, axis=1)
        restricted_lab_df.valueuom = restricted_lab_df.apply(lambda X: row['DPI_units']
        if X.label == row['DPI_name'] else X.valueuom, axis=1)

    # Converting units    for proteine C - reactive(some values are ine mg / dl and not mg / l)
    restricted_lab_df.loc[restricted_lab_df.valueuom == 'mg/dL', 'valuenum'] = \
                            restricted_lab_df[restricted_lab_df.valueuom == 'mg/dL']['valuenum'] * 10
    restricted_lab_df.loc[restricted_lab_df.valueuom == 'mg/dL', 'valueuom'] = 'mg/l'

    restricted_lab_df.value = restricted_lab_df.valuenum.astype(str)

    # Replace units that are equivalent
    restricted_lab_df.loc[(restricted_lab_df['label'].isin(['chlore', 'sodium', 'potassium']))
                          & (restricted_lab_df['valueuom'] == 'mEq/L'), 'valueuom'] = 'mmol/l'
    restricted_lab_df.loc[restricted_lab_df['valueuom'] == 'mmol/L', 'valueuom'] = 'mmol/l'
    restricted_lab_df.loc[(restricted_lab_df['valueuom'] == 'IU/L'), 'valueuom'] = 'U/l'
    restricted_lab_df.loc[(restricted_lab_df['valueuom'] == 'm/uL'), 'valueuom'] = 'T/l'
    restricted_lab_df.loc[(restricted_lab_df['valueuom'] == 'K/uL'), 'valueuom'] = 'G/l'
    restricted_lab_df.loc[(restricted_lab_df['valueuom'] == 'pg/mL'), 'valueuom'] = 'ng/l'

    # Verify that units in selected_reference_values are consistent with those in restricted_lab_df
    for _, row in selected_reference_values.iterrows():
        if pd.isna(row.DPI_units):
            continue
        if row.DPI_units != restricted_lab_df[restricted_lab_df.label == row.DPI_name].valueuom.unique()[0]:
            raise ValueError(f'Units for {row.DPI_name} do not correspond', row.DPI_units,
                             restricted_lab_df[restricted_lab_df.label == row.DPI_name].valueuom.unique()[0])

    ## RESTRICT TO PLAUSSIBLE RANGE
    plausible_restricted_lab_df = restricted_lab_df.copy()

    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(''))),
                                              'preprocessing/possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)
    plausible_restricted_lab_df['out_of_range'] = False
    for variable in possible_value_ranges['variable_label'].dropna().unique():
        possible_value_ranges_for_variable = possible_value_ranges[possible_value_ranges['variable_label'] == variable]
        plausible_restricted_lab_df.loc[(plausible_restricted_lab_df['label'] == variable)
                                        & (~plausible_restricted_lab_df['valuenum'].between(
            possible_value_ranges_for_variable['Min'].values[0],
            possible_value_ranges_for_variable['Max'].values[0])),
                                        'out_of_range'] = True

    n_observations_out_ouf_range = len(plausible_restricted_lab_df[(plausible_restricted_lab_df["out_of_range"]) & (
        ~plausible_restricted_lab_df.valuenum.isna())])
    if verbose:
        print(f'Excluding {n_observations_out_ouf_range} observations because out of range')

    plausible_restricted_lab_df.loc[plausible_restricted_lab_df['out_of_range'] == True, ['value', 'valuenum']] = np.NAN

    plausible_restricted_lab_df.dropna(subset=['valuenum'], inplace=True)

    ## LOG DESCRIPTIVE STATS
    # get mean number of values per dosage label patient admission id
    median_observations_per_case_admission_id = \
        plausible_restricted_lab_df.groupby(['hadm_id', 'label'])['valuenum'].count().reset_index()
    median_observations_per_case_admission_id_df = median_observations_per_case_admission_id.groupby('label').median()
    median_observations_per_case_admission_id_df.rename(columns={'value': 'median_observations_per_case_admission_id'},
                                                        inplace=True)
    descriptive_stats_df = plausible_restricted_lab_df.groupby('label')['valuenum'].describe()

    if verbose:
        print('Median observations per case admission id:')
        print(median_observations_per_case_admission_id_df)
        print('Descriptive statistics:')
        print(descriptive_stats_df)

    if log_dir != '':
        data_to_log = [selected_reference_values['DPI_name'].tolist(),
                       [n_observations_out_ouf_range]
                       ]
        log_columns = ['included_dosage_labels',
                      'n_observations_out_ouf_range']
        log_dataframe = pd.DataFrame(data_to_log).T
        log_dataframe.columns = log_columns
        log_dataframe = pd.concat([log_dataframe, possible_value_ranges], axis=1)
        log_dataframe.to_csv(os.path.join(log_dir, 'lab_preprocessing_log.csv'), index=False)

        median_observations_per_case_admission_id_df.to_csv(os.path.join(log_dir, 'median_observations_per_case_admission_id.csv'), index=True)
        descriptive_stats_df.to_csv(os.path.join(log_dir, 'descriptive_stats.csv'), index=True)


    return plausible_restricted_lab_df




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess labs from separate lab files in data folder')
    parser.add_argument('data_path')
    parser.add_argument('-o', '--output_dir', help='Directory to save output', required=False, default=None)
    parser.add_argument('-p', '--patient_selection_path', help='Path to patient selection file', required=False, default='')
    args = parser.parse_args()

    lab_df = pd.read_csv(args.data_path)
    preprocessed_lab_df = preprocess_labs(lab_df, log_dir=args.output_dir)
    if args.output_dir is not None:
        preprocessed_lab_df.to_csv(os.path.join(args.output_dir, 'preprocessed_labs.csv'), index=False,
                                   encoding='utf-8')





