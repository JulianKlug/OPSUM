import argparse

import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from mimic_preprocessing.monitoring_preprocessing.GCS_extraction import get_GCS
from mimic_preprocessing.monitoring_preprocessing.nihss_extraction import get_nihss
from preprocessing.utils import restrict_variable_to_possible_ranges


def preprocess_monitoring(monitoring_df: pd.DataFrame, mimic_admission_nihss_db_path:str, verbose:bool = False):

    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(''))),
                                              'preprocessing/possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)

    ## FIO2 PROCESSING
    if verbose:
        print('Processing FIO2')
    FiO2_labels = ['FiO2 Set', 'Inspired O2 Fraction']
    O2_flow_labels = ['O2 Flow', 'O2 Flow (lpm)', 'O2 Flow (lpm) #2']
    O2_labels = FiO2_labels + O2_flow_labels

    fio2_df = monitoring_df[monitoring_df.label.isin(O2_labels)]
    fio2_df.dropna(subset=['valuenum'], inplace=True)

    # convert FiO2 set to percentage
    fio2_df.loc[fio2_df.label == 'FiO2 Set', 'valuenum'] = fio2_df[fio2_df.label == 'FiO2 Set'].valuenum * 100

    # Converting    O2    flow    to FIO2
    fio2_df.loc[(fio2_df.label.isin(O2_flow_labels)) & (fio2_df.valuenum > 15), 'valuenum'] = np.nan
    fio2_df.loc[(fio2_df.label.isin(O2_flow_labels)) & (fio2_df.valuenum < 0), 'valuenum'] = np.nan
    # Set to 21% when flow == 0
    fio2_df.loc[(fio2_df.label.isin(O2_flow_labels)) & (fio2_df.valuenum == 0), 'valuenum'] = 21

    fio2_df.loc[(fio2_df.label.isin(O2_flow_labels)) & (fio2_df.valuenum.notnull()), 'valuenum'] = \
        20 + 4 * fio2_df[(fio2_df.label.isin(O2_flow_labels)) & (fio2_df.valuenum.notnull())]['valuenum']

    # round up values between 20.5 and 21 to 21
    fio2_df.loc[(fio2_df.valuenum > 20.5) & (fio2_df.valuenum < 21), 'valuenum'] = 21

    fio2_df['valueuom'] = '%'
    fio2_df['value'] = fio2_df.valuenum.astype(str)
    fio2_df['label'] = 'FIO2'
    fio2_df = fio2_df.drop_duplicates()

    fio2_df.rename(columns={'valuenum': 'FIO2'}, inplace=True)
    fio2_df, _ = restrict_variable_to_possible_ranges(fio2_df, 'FIO2', possible_value_ranges,
                                                                     verbose=verbose)
    fio2_df.rename(columns={'FIO2': 'valuenum' }, inplace=True)

    ### SpO2 processing ###
    if verbose:
        print('Processing SpO2')

    o2_sat_labels = ['O2 saturation pulseoxymetry', 'SpO2']
    spo2_df = monitoring_df[monitoring_df.label.isin(o2_sat_labels)]
    spo2_df.dropna(subset=['valuenum'], inplace=True)
    spo2_df['valueuom'] = '%'
    spo2_df = spo2_df.drop_duplicates()
    spo2_df.rename(columns={'valuenum': 'spo2'}, inplace=True)
    spo2_df, excluded_spo2_df = restrict_variable_to_possible_ranges(spo2_df, 'spo2', possible_value_ranges,
                                                      verbose=verbose)
    spo2_df.rename(columns={'spo2': 'valuenum'}, inplace=True)
    spo2_df['label'] = 'oxygen_saturation'


    ### Preprocessing systolic blood pressure ###
    if verbose:
        print('Preprocessing systolic blood pressure')

    sys_bp_labels = ['Arterial BP [Systolic]', 'Non Invasive Blood Pressure systolic', 'NBP [Systolic]',
                     'Arterial Blood Pressure systolic', 'ART BP Systolic', 'Arterial BP #2 [Systolic]',
                     'Manual Blood Pressure Systolic Left', 'Manual Blood Pressure Systolic Right',
                     'Manual BP [Systolic]']

    sys_bp_df = monitoring_df[monitoring_df.label.isin(sys_bp_labels)]
    sys_bp_df.dropna(subset=['valuenum'], inplace=True)
    sys_bp_df['valueuom'] = 'mmHg'
    sys_bp_df.rename(columns={'valuenum': 'sys'}, inplace=True)
    sys_bp_df, _ = restrict_variable_to_possible_ranges(sys_bp_df, 'sys', possible_value_ranges,
                                                        verbose=verbose)
    sys_bp_df.rename(columns={'sys': 'valuenum'}, inplace=True)
    sys_bp_df = sys_bp_df.drop_duplicates()
    sys_bp_df['label'] = 'systolic_blood_pressure'

    ### Preprocessing diastolic blood pressure ###
    if verbose:
        print('Preprocessing diastolic blood pressure')

    dia_bp_labels = ['Arterial BP [Diastolic]', 'Non Invasive Blood Pressure diastolic', 'NBP [Diastolic]',
                     'Arterial Blood Pressure diastolic', 'ART BP Diastolic', 'Arterial BP #2 [Diastolic]',
                     'Manual BP [Diastolic]', 'Manual Blood Pressure Diastolic Left',
                     'Manual Blood Pressure Diastolic Right']

    dia_bp_df = monitoring_df[monitoring_df.label.isin(dia_bp_labels)]
    dia_bp_df.dropna(subset=['valuenum'], inplace=True)
    dia_bp_df['valueuom'] = 'mmHg'
    dia_bp_df.rename(columns={'valuenum': 'dia'}, inplace=True)
    dia_bp_df, _ = restrict_variable_to_possible_ranges(dia_bp_df, 'dia', possible_value_ranges,
                                                        verbose=verbose)
    dia_bp_df.rename(columns={'dia': 'valuenum'}, inplace=True)
    dia_bp_df = dia_bp_df.drop_duplicates()
    dia_bp_df['label'] = 'diastolic_blood_pressure'

    ### Preprocessing mean blood pressure ###
    if verbose:
        print('Preprocessing mean blood pressure')

    mean_bp_labels = ['Arterial BP Mean', 'Non Invasive Blood Pressure mean', 'NBP Mean',
                      'Arterial Blood Pressure mean', 'ART BP mean', 'Arterial BP Mean #2', 'Manual BP Mean(calc)']

    mean_bp_df = monitoring_df[monitoring_df.label.isin(mean_bp_labels)]
    mean_bp_df.dropna(subset=['valuenum'], inplace=True)
    mean_bp_df['valueuom'] = 'mmHg'
    mean_bp_df.rename(columns={'valuenum': 'mean'}, inplace=True)
    mean_bp_df, _ = restrict_variable_to_possible_ranges(mean_bp_df, 'mean', possible_value_ranges,
                                                        verbose=verbose)
    mean_bp_df.rename(columns={'mean': 'valuenum'}, inplace=True)
    mean_bp_df = mean_bp_df.drop_duplicates()
    mean_bp_df['label'] = 'mean_blood_pressure'


    ### Preprocessing heart rate ###
    if verbose:
        print('Preprocessing heart rate')
    heart_rate_labels = ['Heart Rate']
    heart_rate_df = monitoring_df[monitoring_df.label.isin(heart_rate_labels)]
    heart_rate_df.dropna(subset=['valuenum'], inplace=True)
    heart_rate_df['valueuom'] = possible_value_ranges[possible_value_ranges.variable_label == 'pulse'].units.values[0]
    heart_rate_df.rename(columns={'valuenum': 'pulse'}, inplace=True)
    heart_rate_df, _ = restrict_variable_to_possible_ranges(heart_rate_df, 'pulse', possible_value_ranges,
                                                        verbose=verbose)
    heart_rate_df.rename(columns={'pulse': 'valuenum'}, inplace=True)
    heart_rate_df = heart_rate_df.drop_duplicates()
    heart_rate_df['label'] = 'heart_rate'


    ### Preprocessing respiratory rate ###
    if verbose:
        print('Preprocessing respiratory rate')

    resp_rate_labels = ['Respiratory Rate', 'Respiratory Rate (spontaneous)', 'Respiratory Rate (Total)']
    resp_rate_df = monitoring_df[monitoring_df.label.isin(resp_rate_labels)]
    resp_rate_df.dropna(subset=['valuenum'], inplace=True)
    resp_rate_df['valueuom'] = possible_value_ranges[possible_value_ranges.variable_label == 'fr'].units.values[0]
    resp_rate_df.rename(columns={'valuenum': 'fr'}, inplace=True)
    resp_rate_df, _ = restrict_variable_to_possible_ranges(resp_rate_df, 'fr', possible_value_ranges,
                                                        verbose=verbose)
    resp_rate_df.rename(columns={'fr': 'valuenum'}, inplace=True)
    resp_rate_df = resp_rate_df.drop_duplicates()
    resp_rate_df['label'] = 'respiratory_rate'


    ### Preprocessing temperature ###
    if verbose:
        print('Preprocessing temperature')

    temperature_labels = ['Temperature F', 'Temperature C (calc)', 'Temperature Fahrenheit', 'Temperature C',
                          'Temperature F (calc)', 'Temperature Celsius']

    temperature_df = monitoring_df[monitoring_df.label.isin(temperature_labels)]

    temperature_df.dropna(subset=['valuenum', 'valueuom'], inplace=True)
    fahrenheit_equivalents = ['Deg. F', '?F']
    celsius_equivalents = ['Deg. C', '?C']
    temperature_df.loc[
        temperature_df.valueuom.isin(fahrenheit_equivalents),
        'valuenum'] = (temperature_df[temperature_df.valueuom.isin(fahrenheit_equivalents)].valuenum - 32) * (5 / 9)
    temperature_df.loc[
        temperature_df.valueuom.isin(fahrenheit_equivalents + celsius_equivalents),
        'valueuom'] = celsius_equivalents[0]
    temperature_df['value'] = temperature_df['valuenum'].astype(str)

    if len(temperature_df['valueuom'].unique()) > 1:
        raise ValueError('Temperature units not unified:', temperature_df['valueuom'].unique())

    temperature_df.rename(columns={'valuenum': 'temperature'}, inplace=True)
    temperature_df, excluded_temp_df = restrict_variable_to_possible_ranges(temperature_df, 'temperature', possible_value_ranges,
                                                            verbose=verbose)
    temperature_df.rename(columns={'temperature': 'valuenum'}, inplace=True)
    temperature_df = temperature_df.drop_duplicates()
    temperature_df['label'] = 'temperature'

    ### Preprocessing weight ###
    if verbose:
        print('Preprocessing weight')

    admission_weight_labels = ['Admission Weight (lbs.)', 'Admission Weight (Kg)', 'Previous WeightF',
                               'Previous Weight']
    monitoring_weight_labels = ['Daily Weight']

    admission_weight_df = monitoring_df[monitoring_df.label.isin(admission_weight_labels)]

    # transform lbs to kg
    admission_weight_df.loc[admission_weight_df.label == 'Admission Weight (lbs.)', 'valuenum'] = \
        admission_weight_df.loc[admission_weight_df.label == 'Admission Weight (lbs.)', 'valuenum'] * 0.453592
    admission_weight_df.loc[admission_weight_df.label == 'Admission Weight (lbs.)', 'valueuom'] = 'kg'

    admission_weight_df.dropna(subset=['valuenum', 'valueuom'], inplace=True)
    admission_weight_df['value'] = admission_weight_df['valuenum'].astype(str)

    if len(admission_weight_df['valueuom'].unique()) > 1:
        raise ValueError('Weight units not unified:', admission_weight_df['valueuom'].unique())

    admission_weight_df['charttime'] = admission_weight_df['admittime']
    admission_weight_df.drop_duplicates(inplace=True)

    weight_df = monitoring_df[monitoring_df.label.isin(monitoring_weight_labels)]
    weight_df.dropna(subset=['valuenum', 'valueuom'], inplace=True)
    weight_df.drop_duplicates(inplace=True)
    weight_df = weight_df.append(admission_weight_df)

    weight_df.rename(columns={'valuenum': 'weight'}, inplace=True)
    weight_df, excluded_weight_df = restrict_variable_to_possible_ranges(weight_df, 'weight', possible_value_ranges,
                                                            verbose=verbose)
    weight_df.rename(columns={'weight': 'valuenum'}, inplace=True)

    weight_df['label'] = 'weight'

    ### Preprocessing glucose ###
    if verbose:
        print('Preprocessing glucose')

    glucose_labels = ['Fingerstick Glucose', 'Glucose finger stick', 'Glucose', 'Glucose (serum)',
                      'Glucose (whole blood)', 'Glucose (70-105)']

    glucose_df = monitoring_df[monitoring_df.label.isin(glucose_labels)]
    glucose_df.dropna(subset=['valuenum'], inplace=True)

    # convert mg/dL to mmol/L
    glucose_df['valuenum'] = glucose_df['valuenum'] * 0.0555
    glucose_df['valueuom'] = possible_value_ranges[possible_value_ranges.variable_label == 'glucose'].units.values[0]
    glucose_df['value'] = glucose_df['valuenum'].astype(str)

    glucose_df.rename(columns={'valuenum': 'glucose'}, inplace=True)
    glucose_df, excluded_glucose_df = restrict_variable_to_possible_ranges(glucose_df, 'glucose', possible_value_ranges,
                                                            verbose=verbose)
    glucose_df.rename(columns={'glucose': 'valuenum'}, inplace=True)
    glucose_df = glucose_df.drop_duplicates()
    glucose_df['label'] = 'glucose'


    ## GCS PROCESSING
    if verbose:
        print('Preprocessing GCS')
    GCS_components = ['GCS - Eye Opening', 'GCS - Motor Response', 'GCS - Verbal Response', 'Eye Opening',
                      'Verbal Response', 'Motor Response', 'GCS Total']
    GCS_df = monitoring_df[monitoring_df.label.isin(GCS_components)]
    GCS_motor_components = ['GCS - Motor Response', 'Motor Response']
    # set of ids + timesteps for which the motor component is available - set of ids + timesteps for which the total GCS is available
    GCS_df['combined_id'] = GCS_df['hadm_id'].astype(str) + '_' + GCS_df['charttime'].astype(str)
    # then do difference of sets and iterate only over those
    combined_ids_without_total_GCS = set(GCS_df[GCS_df['label'] == 'GCS - Motor Response'].combined_id.unique()) - set(GCS_df[GCS_df['label'] == 'GCS Total'].combined_id.unique())
    target_motor_gcs_df = GCS_df[(GCS_df.combined_id.isin(combined_ids_without_total_GCS))
                                                & (GCS_df.label.isin(GCS_motor_components))]
    target_motor_gcs_df.drop_duplicates(subset=['combined_id'], inplace=True)

    for ind, gcs_motor_eval in tqdm(target_motor_gcs_df.iterrows(),
                                                total=len(target_motor_gcs_df)):
        gcs_motor_eval['label'] = 'GCS Total'
        gcs_motor_eval['valuenum'] = get_GCS(GCS_df, gcs_motor_eval.hadm_id, gcs_motor_eval.charttime)
        gcs_motor_eval['value'] = str(gcs_motor_eval['valuenum'])
        gcs_motor_eval['valueuom'] = 'points'
        GCS_df = GCS_df.append(gcs_motor_eval)

    combined_ids_without_total_GCS = set(GCS_df[GCS_df['label'] == 'GCS - Motor Response'].combined_id.unique()) - set(GCS_df[GCS_df['label'] == 'GCS Total'].combined_id.unique())
    if verbose:
        print('Number of GCS motor components without total GCS: {}'.format(len(combined_ids_without_total_GCS)))

    GCS_df = GCS_df[GCS_df.label == 'GCS Total']
    GCS_df.drop(columns=['combined_id'], inplace=True)
    GCS_df.dropna(subset=['valuenum'], inplace=True)
    GCS_df.rename(columns={'valuenum': 'Glasgow Coma Scale'}, inplace=True)
    GCS_df, excluded_GCS_df = restrict_variable_to_possible_ranges(GCS_df, 'Glasgow Coma Scale', possible_value_ranges,
                                                            verbose=verbose)
    GCS_df.rename(columns={'Glasgow Coma Scale': 'valuenum'}, inplace=True)
    GCS_df = GCS_df.drop_duplicates()
    GCS_df['label'] = 'Glasgow Coma Scale'

    ### Preprocessing NIHSS ###
    if verbose:
        print('Preprocessing NIHSS')

    NIHSS_labels = ["Level of Consciousness",
                    "Level of Conscious",
                    "Richmond-RAS Scale",
                    "Riker-SAS Scale",
                    "GCS Total",
                    "Ramsey SedationScale",
                    "PAR-Consciousness",
                    "Orientation",
                    "Orientation to Place",
                    "Orientation to Time",
                    "Orient/Clouding Sensory",
                    "Follows Commands",
                    "Commands Response",
                    "Visual Field Cut",
                    "Facial Droop",
                    "Face Droop",
                    "RU Strength/Movement",
                    "Strength R Arm",
                    "LU Strength/Movement",
                    "Strength L Arm",
                    "RL Strength/Movement",
                    "Strength R Leg",
                    "LL Strength/Movement",
                    "Strength L Leg",
                    "Ataxia",
                    "LUE Sensation",
                    "LLE Sensation",
                    "LLE [Sensation]",
                    "LUE [Sensation]",
                    "RUE Sensation",
                    "RLE Sensation",
                    "RLE [Sensation]",
                    "RUE [Sensation]",
                    "Braden Sensory Perception",
                    "Braden SensoryPercep",
                    "Speech",
                    "Slurred Speech"]

    NIHSS_df = monitoring_df[monitoring_df.label.isin(NIHSS_labels)]
    NIHSS_df.dropna(subset=['value'], inplace=True)

    # get NIHSS from admission extracted by NLP
    mimic_admission_df = pd.read_csv(mimic_admission_nihss_db_path)

    nihss_motor_components = [ "RU Strength/Movement",
                    "Strength R Arm",
                    "LU Strength/Movement",
                    "Strength L Arm",
                    "RL Strength/Movement",
                    "Strength R Leg",
                    "LL Strength/Movement",
                    "Strength L Leg"]

    NIHSS_df['combined_id'] = NIHSS_df['hadm_id'].astype(str) + '_' + NIHSS_df['charttime'].astype(str)
    # get all timepoints of motor evaluations
    motor_evaluations = NIHSS_df[NIHSS_df.label.isin(nihss_motor_components)].copy()
    motor_evaluations.drop_duplicates(subset=['combined_id'], inplace=True)
    NIHSS_df.drop(columns=['combined_id'], inplace=True)
    motor_evaluations.drop(columns=['combined_id'], inplace=True)

    preprocessed_NIHSS_df = pd.DataFrame()
    for ind, nihss_motor_eval in tqdm(motor_evaluations.iterrows(), total=motor_evaluations.shape[0]):
        nihss_motor_eval['label'] = 'NIHSS'
        nihss_motor_eval['valuenum'] = get_nihss(NIHSS_df, mimic_admission_df, nihss_motor_eval.hadm_id, nihss_motor_eval.charttime)
        nihss_motor_eval['value'] = nihss_motor_eval['valuenum'].astype(str)
        nihss_motor_eval['valueuom'] = 'points'
        preprocessed_NIHSS_df = preprocessed_NIHSS_df.append(nihss_motor_eval)


    preprocessed_NIHSS_df.rename(columns={'valuenum': 'NIHSS'}, inplace=True)
    preprocessed_NIHSS_df, excluded_NIHSS_df = restrict_variable_to_possible_ranges(preprocessed_NIHSS_df, 'NIHSS', possible_value_ranges,
                                                            verbose=verbose)
    preprocessed_NIHSS_df.rename(columns={'NIHSS': 'valuenum'}, inplace=True)
    preprocessed_NIHSS_df = preprocessed_NIHSS_df.drop_duplicates()


    # add subparts of monitoring back together
    preprocessed_df = pd.concat([sys_bp_df, dia_bp_df, mean_bp_df, heart_rate_df, resp_rate_df, spo2_df,
                                            temperature_df, weight_df, glucose_df, fio2_df, preprocessed_NIHSS_df, GCS_df], axis=0)

    return preprocessed_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitoring_data_path', '-d', type=str)
    parser.add_argument('--mimic_admission_nihss_db_path', '-a', type=str)
    parser.add_argument('--output_dir', '-o', type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    monitoring_df = pd.read_csv(args.monitoring_data_path)

    preprocessed_df = preprocess_monitoring(monitoring_df, args.mimic_admission_nihss_db_path, verbose=args.verbose)

    preprocessed_df.to_csv(os.path.join(output_dir, 'preprocessed_monitoring_df.csv'), index=False)

