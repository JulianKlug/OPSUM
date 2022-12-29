import pandas as pd
import os
import numpy as np

from preprocessing.geneva_stroke_unit_preprocessing.utils import create_ehr_case_identification_column, safe_conversion_to_numeric, \
    restrict_variable_to_possible_ranges

columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                   'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                   'eds_final_begin', 'eds_final_end', 'eds_final_patient_id',
                   'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                   'date_from', 'date_to', 'patient_id_manual', 'stroke_onset_date', 'Referral', 'match_by',
                   'multiple_id']

variables_to_drop = ['air', 'air_unit', 'peep', 'peep_unit', 'startingFlow', 'startingFlow_unit',
                     'flow', 'flow_unit', 'temperature', 'temperature_unit',
                     'ai', 'ai_unit', 'epap', 'epap_unit', 'ipap', 'ipap_unit', 'slop',
                     'slop_unit', 'ti_max', 'ti_max_unit', 'ti_min', 'ti_min_unit',
                     'trigger_insp', 'trigger_insp_unit', 'duration', 'duration_unit']


def preprocess_ventilation(ventilation_df, first_sample_date_df, verbose=False):
    """
    Preprocesses the ventilation dataframe.
    First sample dates from other EHR date is needed to infer the time of the first sample for FiO2 inference

    Arguments:
        ventilation_df: Dataframe with ventilation data
        first_sample_date_df: Dataframe with first sample dates from other EHR date (case_admission_id, first_sample_date)
        verbose: If True, prints some information about the geneva_stroke_unit_preprocessing
    Returns:
        Dataframe with preprocessed ventilation data
    """
    ventilation_df['case_admission_id'] = create_ehr_case_identification_column(ventilation_df)

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

    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'] > 15), 'O2'] = np.nan
    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'] < 0), 'O2'] = np.nan

    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min')
                       & (ventilation_df['O2'].notnull())
                       & (ventilation_df['FIO2'].isnull()), 'FIO2'] = 20 + 4 * ventilation_df[
        (ventilation_df['O2_unit'] == 'L/min')
        & (ventilation_df['O2'].notnull())
        & (ventilation_df['FIO2'].isnull())]['O2']

    # Set to 21% when flow == 0
    ventilation_df.loc[(ventilation_df['O2_unit'] == 'L/min') & (ventilation_df['O2'] == 0)
                       & (ventilation_df['FIO2'].isnull()), 'FIO2'] = 21


    # Remove unwanted variables
    ventilation_df.drop(variables_to_drop, axis=1, inplace=True)

    # Split into Fio2 and SpO2 df
    # Convert to float with safe conversion
    fio2_df = safe_conversion_to_numeric(
        ventilation_df[['case_admission_id', 'FIO2', 'FIO2_unit', 'datetime']].dropna(subset=['FIO2']), 'FIO2')
    spo2_df = safe_conversion_to_numeric(
        ventilation_df[['case_admission_id', 'spo2', 'spo2_unit', 'datetime']].dropna(subset=['spo2']), 'spo2')

    if verbose:
        print('FIO2:')
    fio2_df, _ = restrict_variable_to_possible_ranges(fio2_df, 'FIO2', possible_value_ranges, verbose=verbose)

    # Set fIo2 to 21% if no value exists for a specific case_admission_id
    case_admission_ids_with_no_fio2 = set(first_sample_date_df['case_admission_id']) - set(fio2_df['case_admission_id'])
    room_air_fio2_df = first_sample_date_df[
        first_sample_date_df.case_admission_id.isin(case_admission_ids_with_no_fio2)]
    room_air_fio2_df['FIO2'] = 21
    room_air_fio2_df['FIO2_unit'] = '%'
    room_air_fio2_df.rename(columns={'first_sample_date': 'datetime'}, inplace=True)
    fio2_df = pd.concat([fio2_df, room_air_fio2_df])

    if verbose:
        print('SPO2:')
    spo2_df, _ = restrict_variable_to_possible_ranges(spo2_df, 'spo2', possible_value_ranges,
                                                                     verbose=verbose)

    return fio2_df, spo2_df
