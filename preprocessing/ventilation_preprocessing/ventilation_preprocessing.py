import pandas as pd
import os
import numpy as np

from preprocessing.utils import create_case_identification_column


def restrict_variable_to_possible_ranges(df, variable_name, possible_value_ranges, verbose=False):
    """
    Restricts a variable to the possible ranges in the possible_value_ranges dataframe.
    """
    variable_range = possible_value_ranges[possible_value_ranges['variable_label'] == variable_name]
    variable_range = variable_range.iloc[0]
    clean_df = df.copy()
    clean_df[variable_name] = df[variable_name].apply(
        lambda x: np.nan if x < variable_range['Min'] or x > variable_range['Max'] else x)
    if verbose:
        print(f'Excluding {clean_df[variable_name].isna().sum()} observations because out of range')
    excluded_df = df[clean_df[variable_name].isna()]
    clean_df = clean_df.dropna()
    return clean_df, excluded_df


def preprocess_ventilation(ventilation_df, verbose=False):
    ventilation_df['case_admission_id'] = create_case_identification_column(ventilation_df)

    columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                       'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                       'eds_final_begin', 'eds_final_end', 'eds_final_patient_id',
                       'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                       'date_from', 'date_to']
    ventilation_df.drop(columns_to_drop, axis=1, inplace=True)

    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              'possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)

    ventilation_df['FIO2'] = ventilation_df['FIO2'].astype(float)

    # Converting    O2    flow    to FIO2
    ventilation_df['O2'] = ventilation_df['O2'].astype(str).apply(lambda t: t.replace(',', '.'))
    ventilation_df['O2'] = ventilation_df['O2'].astype(float)

    ventilation_df.loc[(ventilation_df['O2_unit'] == '%') & (ventilation_df['FIO2'].isnull()), 'FIO2'] = \
        ventilation_df[(ventilation_df['O2_unit'] == '%') & (ventilation_df['FIO2'].isnull())]['O2']
    # %%
    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'] > 15), 'O2'] = np.nan
    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'] < 0), 'O2'] = np.nan

    # %%
    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'].notnull()), 'FIO2'] = 20 + 4 * \
                                                                                                            ventilation_df[
                                                                                                                (
                                                                                                                            ventilation_df[
                                                                                                                                'O2_unit'] == 'L/min') & (
                                                                                                                    ventilation_df[
                                                                                                                        'O2'].notnull())][
                                                                                                                'O2']

    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'] == 0), 'FIO2'] = 21

    variables_to_drop = ['air', 'air_unit', 'peep', 'peep_unit', 'startingFlow', 'startingFlow_unit',
                         'flow', 'flow_unit', 'temperature', 'temperature_unit',
                         'ai', 'ai_unit', 'epap', 'epap_unit', 'ipap', 'ipap_unit', 'slop',
                         'slop_unit', 'ti_max', 'ti_max_unit', 'ti_min', 'ti_min_unit',
                         'trigger_insp', 'trigger_insp_unit', 'duration', 'duration_unit']
    ventilation_df.drop(variables_to_drop, axis=1, inplace=True)
    fio2_df = ventilation_df[['case_admission_id', 'FIO2', 'FIO2_unit', 'datetime']].dropna()
    spo2_df = ventilation_df[['case_admission_id', 'spo2', 'spo2_unit', 'datetime']].dropna()

    # convert to numeric
    fio2_df['FIO2'] = pd.to_numeric(fio2_df['FIO2'], errors='coerce')
    spo2_df['spo2'] = pd.to_numeric(spo2_df['spo2'], errors='coerce')

    if verbose:
        print('FIO2:')
    fio2_df, _ = restrict_variable_to_possible_ranges(fio2_df, 'FIO2', possible_value_ranges,
                                                                     verbose=verbose)
    if verbose:
        print('SPO2:')
    spo2_df, _ = restrict_variable_to_possible_ranges(spo2_df, 'spo2', possible_value_ranges,
                                                                     verbose=verbose)

    return fio2_df, spo2_df