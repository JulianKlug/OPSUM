import numpy as np
import os
import pandas as pd

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


def preprocess_vitals(vitals_df, verbose=False):
    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              'possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)

    vitals_df['case_admission_id'] = create_case_identification_column(vitals_df)

    columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                       'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                       'eds_final_begin', 'eds_final_end', 'eds_final_patient_id',
                       'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                       'date_from', 'date_to', 'patient_value']
    vitals_df.drop(columns_to_drop, axis=1, inplace=True)

    # Preprocessing  temperature
    if verbose:
        print('Preprocessing temperature')
    temperature_df = vitals_df[['case_admission_id', 'datetime', 'temperature', 'temp_unit']].dropna()
    # convert ',' to '.' in temperature column
    temperature_df['temperature'] = temperature_df['temperature'].astype(str).apply(lambda t: t.replace(',', '.'))
    # remove trailing '.'
    temperature_df['temperature'] = temperature_df['temperature'].apply(lambda t: t.rstrip('.'))
    temperature_df['temperature'] = temperature_df['temperature'].astype(float)
    temperature_df, _ = restrict_variable_to_possible_ranges(temperature_df, 'temperature', possible_value_ranges,
                                                             verbose=verbose)
    temperature_df = temperature_df.rename(columns={'temperature': 'vital_value', 'temp_unit': 'vital_unit'})
    temperature_df['vital_name'] = 'temperature'

    # Preprocessing    systolic    blood    pressure
    if verbose:
        print('Preprocessing systolic blood pressure')
    sys_bp_df = vitals_df[['case_admission_id', 'datetime', 'sys', 'sys_unit']].dropna()
    sys_bp_df['sys'] = pd.to_numeric(sys_bp_df['sys'], errors='coerce')
    sys_bp_df, _ = restrict_variable_to_possible_ranges(sys_bp_df, 'sys', possible_value_ranges,
                                                                         verbose=verbose)
    sys_bp_df = sys_bp_df.rename(columns={'sys': 'vital_value', 'sys_unit': 'vital_unit'})
    sys_bp_df['vital_name'] = 'systolic_blood_pressure'

    # Preprocessing    diastolic blood    pressure
    if verbose:
        print('Preprocessing diastolic blood pressure')
    dia_bp_df = vitals_df[['case_admission_id', 'datetime', 'dia', 'dia_unit']].dropna()
    dia_bp_df['dia'] = pd.to_numeric(dia_bp_df['dia'], errors='coerce')
    dia_bp_df, _ = restrict_variable_to_possible_ranges(dia_bp_df, 'dia', possible_value_ranges,
                                                                         verbose=verbose)
    dia_bp_df = dia_bp_df.rename(columns={'dia': 'vital_value', 'dia_unit': 'vital_unit'})
    dia_bp_df['vital_name'] = 'diastolic_blood_pressure'

    # Preprocessing    mean blood    pressure
    if verbose:
        print('Preprocessing mean blood pressure')
    mean_bp_df = vitals_df[['case_admission_id', 'datetime', 'mean', 'mean_unit']].dropna()
    mean_bp_df['mean'] = pd.to_numeric(mean_bp_df['mean'], errors='coerce')
    mean_bp_df, _ = restrict_variable_to_possible_ranges(mean_bp_df, 'mean', possible_value_ranges,
                                                                           verbose=verbose)
    mean_bp_df = mean_bp_df.rename(columns={'mean': 'vital_value', 'mean_unit': 'vital_unit'})
    mean_bp_df['vital_name'] = 'mean_blood_pressure'

    # Preprocessing    heart rate
    if verbose:
        print('Preprocessing heart rate')
    pulse_df = vitals_df[['case_admission_id', 'datetime', 'pulse', 'pulse_unit']].dropna()
    pulse_df['pulse'] = pulse_df['pulse'].astype(str).apply(lambda p: p.replace(',', '.'))
    pulse_df = pulse_df[pulse_df['pulse'] != '.']
    pulse_df['pulse'] = pulse_df['pulse'].astype(float)
    pulse_df, _ = restrict_variable_to_possible_ranges(pulse_df, 'pulse', possible_value_ranges,
                                                                       verbose=verbose)
    pulse_df['pulse_unit'] = '/min'
    pulse_df = pulse_df.rename(columns={'pulse': 'vital_value', 'pulse_unit': 'vital_unit'})
    pulse_df['vital_name'] = 'heart_rate'

    # Preprocessing    respiratory rate
    if verbose:
        print('Preprocessing respiratory rate')
    resp_rate_df = vitals_df[['case_admission_id', 'datetime', 'fr', 'fr_unit']].dropna()
    resp_rate_df['fr'] = resp_rate_df['fr'].astype(str).apply(lambda r: r.replace(',', '.'))
    resp_rate_df = resp_rate_df[resp_rate_df['fr'] != '.']
    resp_rate_df['fr'] = resp_rate_df['fr'].astype(float)
    resp_rate_df, _ = restrict_variable_to_possible_ranges(resp_rate_df, 'fr',
                                                                               possible_value_ranges, verbose=verbose)
    resp_rate_df['fr_unit'] = '/min'
    resp_rate_df = resp_rate_df.rename(columns={'fr': 'vital_value', 'fr_unit': 'vital_unit'})
    resp_rate_df['vital_name'] = 'respiratory_rate'

    # Preprocessing    oxygen    saturation
    if verbose:
        print('Preprocessing oxygen saturation')
    spo2_df = vitals_df[['case_admission_id', 'datetime', 'spo2', 'spo2_unit']].dropna()
    spo2_df['spo2'] = pd.to_numeric(spo2_df['spo2'], errors='coerce')
    spo2_df, _ = restrict_variable_to_possible_ranges(spo2_df, 'spo2', possible_value_ranges,
                                                                     verbose=verbose)
    spo2_df = spo2_df.rename(columns={'spo2': 'vital_value', 'spo2_unit': 'vital_unit'})
    spo2_df['vital_name'] = 'oxygen_saturation'

    # Preprocessing    weight
    if verbose:
        print('Preprocessing weight')
    weight_df = vitals_df[['case_admission_id', 'datetime', 'weight', 'weight_unit']].dropna()
    weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
    weight_df, _ = restrict_variable_to_possible_ranges(weight_df, 'weight', possible_value_ranges,
                                                                         verbose=verbose)
    weight_df = weight_df.rename(columns={'weight': 'vital_value', 'weight_unit': 'vital_unit'})
    weight_df['vital_name'] = 'weight'

    preprocessed_vitals_df = pd.concat([sys_bp_df, dia_bp_df, mean_bp_df, pulse_df, resp_rate_df, spo2_df,
                                        temperature_df, weight_df], axis=0)

    return preprocessed_vitals_df



