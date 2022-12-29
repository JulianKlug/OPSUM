import numpy as np
import os
import pandas as pd

from preprocessing.geneva_stroke_unit_preprocessing.utils import create_ehr_case_identification_column, restrict_variable_to_possible_ranges, \
    safe_conversion_to_numeric

columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                   'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                   'eds_final_begin', 'eds_final_end', 'eds_final_patient_id',
                   'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                   'date_from', 'date_to', 'patient_id_manual', 'stroke_onset_date', 'Referral',
                   'match_by', 'multiple_id']


def string_to_numeric(df, column_name):
    """
    Convert a column with mixed string and numeric values to numeric.
    """
    # convert ',' to '.' in temperature column
    df[column_name] = df[column_name].astype(str).apply(lambda t: t.replace(',', '.'))
    # remove trailing '.'
    df[column_name] = df[column_name].apply(lambda t: t.rstrip('.'))
    # remove empty strings
    df = df[df[column_name] != '']
    df = df[df[column_name] != '-']
    # convert to numeric
    df = safe_conversion_to_numeric(df, column_name)
    return df


def harmonize_units(df, variable_name, unit_name, possible_value_ranges, equivalent_units):
    """
    Harmonize units of a variable.
    """
    target_unit = possible_value_ranges[possible_value_ranges.variable_label == variable_name].units.iloc[0]
    if target_unit in equivalent_units:
        df[unit_name].fillna(target_unit, inplace=True)
        # convert to regex with |
        df[unit_name].replace('|'.join(equivalent_units), target_unit, regex=True, inplace=True)
    else:
        raise ValueError(
            f'{variable_name} target unit as defined in possible_value_ranges_file, not part of {equivalent_units}')
    if len(df[unit_name].unique()) > 1:
        raise ValueError(f'{variable_name} units not unified:', df[unit_name].unique())

    return df


def preprocess_vitals(vitals_df, verbose=False):
    """
    Preprocess the dataframe of aggregated patientvalues (including vitals).
    Currently processed vital signs are: temperature, blood pressure, heart rate, respiratory rate, oxygen saturation, FiO2, weight

    Note: the structure of this dataframe has changes as of extraction 20220815 (therefore dataframes with subkey column have to be converted for compatibility)

    Arguments:
        vitals_df {pandas.DataFrame} -- Dataframe of aggregated patientvalues (including vitals).
        verbose {bool} -- If True, print some information.
    Returns:
        pandas.DataFrame -- Dataframe of patient vitals
    """

    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              'possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)

    vitals_df['case_admission_id'] = create_ehr_case_identification_column(vitals_df)

    vitals_df.drop(columns_to_drop, axis=1, inplace=True)

    ### Preprocessing temperature ###
    if verbose:
        print('Preprocessing temperature')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        temperature_df = vitals_df[vitals_df.patient_value.values == 'pv.temperature']
        temperature_df = temperature_df[temperature_df.subkey.values == 'temperature']
        temperature_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        temperature_df.rename(columns={'value': 'temperature', 'unit': 'temp_unit'}, inplace=True)
    else:
        temperature_df = vitals_df[['case_admission_id', 'datetime', 'temperature', 'temp_unit']]

    temperature_df.dropna(subset=['temperature'], inplace=True)

    temperature_df = string_to_numeric(temperature_df, 'temperature')

    if len(temperature_df['temp_unit'].unique()) > 1:
        raise ValueError('Temperature units not unified:', temperature_df['temp_unit'].unique())

    temperature_df, _ = restrict_variable_to_possible_ranges(temperature_df, 'temperature', possible_value_ranges,
                                                             verbose=verbose)
    temperature_df = temperature_df.drop_duplicates()
    temperature_df = temperature_df.rename(columns={'temperature': 'vital_value', 'temp_unit': 'vital_unit'})
    temperature_df['vital_name'] = 'temperature'

    ### Preprocessing glycemia ###
    # Glycemia did not exist prior to 2020-08-15, thus only one version is needed
    if 'subkey' in vitals_df.columns:
        if verbose:
            print('Preprocessing glycemia')
        glycemia_df = vitals_df[vitals_df.patient_value.values == 'pv.glycemia']
        glycemia_df = glycemia_df[glycemia_df.subkey.values == 'glycemia']
        glycemia_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        glycemia_df.dropna(subset=['value'], inplace=True)

        if len(glycemia_df['unit'].unique()) > 1:
            raise ValueError('Glycemia units not unified:', glycemia_df['unit'].unique())

        glycemia_df = string_to_numeric(glycemia_df, 'value')

        glycemia_df.rename(columns={'value': 'glucose'}, inplace=True)
        glycemia_df, excluded_glycemia_df = restrict_variable_to_possible_ranges(glycemia_df, 'glucose',
                                                                                 possible_value_ranges, verbose=True)
        glycemia_df = glycemia_df.drop_duplicates()
        glycemia_df = glycemia_df.rename(columns={'glucose': 'vital_value', 'unit': 'vital_unit'})
        glycemia_df['vital_name'] = 'glucose'

    ### Preprocessing systolic blood pressure ###
    if verbose:
        print('Preprocessing systolic blood pressure')

    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        sys_bp_df = vitals_df[(vitals_df.patient_value.values == 'pv.ta') & (vitals_df.subkey.values == 'sys')]
        sys_bp_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        sys_bp_df.rename(columns={'value': 'sys', 'unit': 'sys_unit'}, inplace=True)
    else:
        sys_bp_df = vitals_df[['case_admission_id', 'datetime', 'sys', 'sys_unit']]

    sys_bp_df.dropna(subset=['sys'], inplace=True)

    sys_bp_df['sys_unit'].fillna('mmHg', inplace=True)
    if len(sys_bp_df['sys_unit'].unique()) > 1:
        raise ValueError('Systolic blood pressure units not unified:', sys_bp_df['sys_unit'].unique())

    sys_bp_df = string_to_numeric(sys_bp_df, 'sys')

    sys_bp_df, _ = restrict_variable_to_possible_ranges(sys_bp_df, 'sys', possible_value_ranges,
                                                        verbose=verbose)
    sys_bp_df = sys_bp_df.drop_duplicates()
    sys_bp_df = sys_bp_df.rename(columns={'sys': 'vital_value', 'sys_unit': 'vital_unit'})
    sys_bp_df['vital_name'] = 'systolic_blood_pressure'

    ### Preprocessing diastolic blood pressure ###
    if verbose:
        print('Preprocessing diastolic blood pressure')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        dia_bp_df = vitals_df[(vitals_df.patient_value.values == 'pv.ta') & (vitals_df.subkey.values == 'dia')]
        dia_bp_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        dia_bp_df.rename(columns={'value': 'dia', 'unit': 'dia_unit'}, inplace=True)
    else:
        dia_bp_df = vitals_df[['case_admission_id', 'datetime', 'dia', 'dia_unit']]

    dia_bp_df.dropna(subset=['dia'], inplace=True)

    dia_bp_df['dia_unit'].fillna('mmHg', inplace=True)
    if len(dia_bp_df['dia_unit'].unique()) > 1:
        raise ValueError('Diasystolic blood pressure units not unified:', dia_bp_df['dia_unit'].unique())

    dia_bp_df = string_to_numeric(dia_bp_df, 'dia')

    dia_bp_df, _ = restrict_variable_to_possible_ranges(dia_bp_df, 'dia', possible_value_ranges,
                                                        verbose=verbose)
    dia_bp_df = dia_bp_df.drop_duplicates()
    dia_bp_df = dia_bp_df.rename(columns={'dia': 'vital_value', 'dia_unit': 'vital_unit'})
    dia_bp_df['vital_name'] = 'diastolic_blood_pressure'

    ### Preprocessing mean blood pressure ###
    if verbose:
        print('Preprocessing mean blood pressure')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        mean_bp_df = vitals_df[(vitals_df.patient_value.values == 'pv.ta') & (vitals_df.subkey.values == 'mean')]
        mean_bp_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        mean_bp_df.rename(columns={'value': 'mean', 'unit': 'mean_unit'}, inplace=True)
    else:
        mean_bp_df = vitals_df[['case_admission_id', 'datetime', 'mean', 'mean_unit']]

    mean_bp_df.dropna(subset=['mean'], inplace=True)

    mean_bp_df['mean_unit'].fillna('mmHg', inplace=True)
    if len(mean_bp_df['mean_unit'].unique()) > 1:
        raise ValueError('Mean blood pressure units not unified:', mean_bp_df['mean_unit'].unique())

    mean_bp_df = string_to_numeric(mean_bp_df, 'mean')

    mean_bp_df, _ = restrict_variable_to_possible_ranges(mean_bp_df, 'mean', possible_value_ranges,
                                                         verbose=verbose)
    mean_bp_df = mean_bp_df.drop_duplicates()
    mean_bp_df = mean_bp_df.rename(columns={'mean': 'vital_value', 'mean_unit': 'vital_unit'})
    mean_bp_df['vital_name'] = 'mean_blood_pressure'

    ### Preprocessing heart rate ###
    if verbose:
        print('Preprocessing heart rate')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        pulse_df = vitals_df[(vitals_df.patient_value.values == 'pv.pulse') & (vitals_df.subkey.values == 'pulse')]
        pulse_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        pulse_df.rename(columns={'value': 'pulse', 'unit': 'pulse_unit'}, inplace=True)
    else:
        pulse_df = vitals_df[['case_admission_id', 'datetime', 'pulse', 'pulse_unit']]

    pulse_df.dropna(subset=['pulse'], inplace=True)

    pulse_equivalent_units = ['bpm', 'puls./min.', '/min']
    pulse_df = harmonize_units(pulse_df, 'pulse', 'pulse_unit', possible_value_ranges, pulse_equivalent_units)

    pulse_df = string_to_numeric(pulse_df, 'pulse')

    pulse_df, _ = restrict_variable_to_possible_ranges(pulse_df, 'pulse', possible_value_ranges,
                                                       verbose=verbose)
    pulse_df = pulse_df.drop_duplicates()
    pulse_df = pulse_df.rename(columns={'pulse': 'vital_value', 'pulse_unit': 'vital_unit'})
    pulse_df['vital_name'] = 'heart_rate'

    ### Preprocessing respiratory rate ###
    if verbose:
        print('Preprocessing respiratory rate')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        resp_rate_df = vitals_df[(vitals_df.patient_value.values == 'pv.fr')]
        resp_rate_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        resp_rate_df.rename(columns={'value': 'fr', 'unit': 'fr_unit'}, inplace=True)
    else:
        resp_rate_df = vitals_df[['case_admission_id', 'datetime', 'fr', 'fr_unit']]

    resp_rate_df.dropna(subset=['fr'], inplace=True)

    resp_rate_equivalent_units = ['/min', 'cycles/min.']
    resp_rate_df = harmonize_units(resp_rate_df, 'fr', 'fr_unit', possible_value_ranges, resp_rate_equivalent_units)

    resp_rate_df = string_to_numeric(resp_rate_df, 'fr')

    resp_rate_df, _ = restrict_variable_to_possible_ranges(resp_rate_df, 'fr',
                                                           possible_value_ranges, verbose=verbose)
    resp_rate_df = resp_rate_df.drop_duplicates()
    resp_rate_df = resp_rate_df.rename(columns={'fr': 'vital_value', 'fr_unit': 'vital_unit'})
    resp_rate_df['vital_name'] = 'respiratory_rate'

    ### Preprocessing Oxygen saturation ###
    if verbose:
        print('Preprocessing oxygen saturation')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        spo2_df = vitals_df[(vitals_df.patient_value.values == 'pv.spo2') & (vitals_df.subkey.values == 'spo2')]
        spo2_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        spo2_df.rename(columns={'value': 'spo2', 'unit': 'spo2_unit'}, inplace=True)
    else:
        spo2_df = vitals_df[['case_admission_id', 'datetime', 'spo2', 'spo2_unit']]

    spo2_df.dropna(subset=['spo2'], inplace=True)
    spo2_df = harmonize_units(spo2_df, 'spo2', 'spo2_unit', possible_value_ranges, ['%'])
    spo2_df = string_to_numeric(spo2_df, 'spo2')
    spo2_df, _ = restrict_variable_to_possible_ranges(spo2_df, 'spo2', possible_value_ranges,
                                                      verbose=verbose)
    spo2_df = spo2_df.drop_duplicates()
    spo2_df = spo2_df.rename(columns={'spo2': 'vital_value', 'spo2_unit': 'vital_unit'})
    spo2_df['vital_name'] = 'oxygen_saturation'

    ### Preprocessing FiO2 ###
    if verbose:
        print('Preprocessing FiO2')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        fio2_df = vitals_df[(vitals_df.patient_value.values == 'pv.spo2')
                            & ((vitals_df.subkey.values == 'o2') | (vitals_df.subkey.values == 'fio2'))]
        fio2_df.drop(columns=['patient_value'], inplace=True)
        fio2_df.rename(columns={'value': 'FIO2', 'unit': 'FIO2_unit'}, inplace=True)
        fio2_df.dropna(subset=['FIO2'], inplace=True)
        fio2_df = string_to_numeric(fio2_df, 'FIO2')

        # Converting    O2    flow    to FIO2
        fio2_df.loc[(fio2_df['FIO2_unit'] == 'L/min') & (fio2_df['FIO2'] > 15), 'FIO2'] = np.nan
        fio2_df.loc[(fio2_df['FIO2_unit'] == 'L/min') & (fio2_df['FIO2'] < 0), 'FIO2'] = np.nan
        # Set to 21% when flow == 0
        fio2_df.loc[(fio2_df['FIO2_unit'] == 'L/min') & (fio2_df['FIO2'] == 0), 'FIO2'] = 21

        fio2_df.loc[(fio2_df['FIO2_unit'] == 'L/min')
                    & (fio2_df['FIO2'].notnull()), 'FIO2'] = 20 + 4 * fio2_df[
            (fio2_df['FIO2_unit'] == 'L/min')
            & (fio2_df['FIO2'].notnull())]['FIO2']
        fio2_df.loc[fio2_df['FIO2_unit'] == 'L/min', 'FIO2_unit'] = '%'

        fio2_df = harmonize_units(fio2_df, 'FIO2', 'FIO2_unit', possible_value_ranges, ['%'])

        fio2_df, excluded_fio2_df = restrict_variable_to_possible_ranges(fio2_df, 'FIO2', possible_value_ranges,
                                                                         verbose=True)
        fio2_df.dropna(subset=['FIO2'], inplace=True)
        fio2_df.drop(columns=['subkey'], inplace=True)
        fio2_df = fio2_df.drop_duplicates()

        fio2_df = fio2_df.rename(columns={'FIO2': 'vital_value', 'FIO2_unit': 'vital_unit'})
        fio2_df['vital_name'] = 'FIO2'

    ### Preprocessing Weight ###
    if verbose:
        print('Preprocessing weight')
    if 'subkey' in vitals_df.columns:
        # convert for compatibility with old data
        weight_df = vitals_df[((vitals_df.patient_value.values == 'pv.weight') & (vitals_df.subkey.values == 'weight'))
                              | ((vitals_df.patient_value.values == 'patient.sv.poids') & (
                vitals_df.subkey.values == 'Valeur'))]
        weight_df.drop(columns=['patient_value', 'subkey'], inplace=True)
        weight_df.rename(columns={'value': 'weight', 'unit': 'weight_unit'}, inplace=True)
    else:
        weight_df = vitals_df[['case_admission_id', 'datetime', 'weight', 'weight_unit']]

    weight_df.dropna(subset=['weight'], inplace=True)
    weight_df = harmonize_units(weight_df, 'weight', 'weight_unit', possible_value_ranges, ['kg'])

    weight_df = string_to_numeric(weight_df, 'weight')
    weight_df, _ = restrict_variable_to_possible_ranges(weight_df, 'weight', possible_value_ranges,
                                                        verbose=verbose)
    weight_df = weight_df.drop_duplicates()
    weight_df = weight_df.rename(columns={'weight': 'vital_value', 'weight_unit': 'vital_unit'})
    weight_df['vital_name'] = 'weight'

    if 'subkey' in vitals_df.columns:
        preprocessed_vitals_df = pd.concat([sys_bp_df, dia_bp_df, mean_bp_df, pulse_df, resp_rate_df, spo2_df,
                                            temperature_df, weight_df, glycemia_df, fio2_df], axis=0)
    else:
        preprocessed_vitals_df = pd.concat([sys_bp_df, dia_bp_df, mean_bp_df, pulse_df, resp_rate_df, spo2_df,
                                            temperature_df, weight_df], axis=0)

    return preprocessed_vitals_df
