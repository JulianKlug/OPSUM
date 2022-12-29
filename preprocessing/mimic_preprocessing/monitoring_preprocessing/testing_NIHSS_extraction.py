import pandas as pd
from tqdm import tqdm

from preprocessing.mimic_preprocessing.monitoring_preprocessing.nihss_extraction import get_level_of_consciousness_1a, \
    get_level_of_orientation_1b, get_level_of_command_response_1c, get_best_gaze_2, get_visual_fields_3, \
    get_facial_palsy_4, get_motor_arms_5, get_motor_legs_6, get_ataxia_7, get_sensory_8, get_language_9, \
    get_dysarthria_10, get_extinction_11

mimic_admission_db_path = '/Users/jk1/stroke_datasets/national-institutes-of-health-stroke-scale-nihss-annotations-for-the-mimic-iii-database-1.0.0/mimic_nihss_database.csv'
monitoring_path = '/Users/jk1/temp/mimic/extraction/monitoring_df.csv'

monitoring_df = pd.read_csv(monitoring_path)
mimic_admission_df = pd.read_csv(mimic_admission_db_path)
delta_nihss_df = mimic_admission_df.copy()

motor_ru_labels = ['RU Strength/Movement', 'Strength R Arm']
motor_lu_labels = ['LU Strength/Movement', 'Strength L Arm']
motor_rl_labels = ['RL Strength/Movement', 'Strength R Leg']
motor_ll_labels = ['LL Strength/Movement', 'Strength L Leg']
motor_labels = motor_ru_labels + motor_rl_labels + motor_ll_labels + motor_lu_labels
nihss_item_labels = [
'NIHSS',
'1a_LOC',
'1b_LOCQuestions',
'1c_LOCCommands',
'2_BestGaze',
'3_Visual',
'4_FacialPalsy',
'5a_LeftArm',
'5b_RightArm',
'6a_LeftLeg',
'6b_RightLeg',
'7_LimbAtaxia',
'8_Sensory',
'9_BestLanguage',
'10_Dysarthria',
'11_ExtinctionInattention'
]

for ind, subj in tqdm(mimic_admission_df.iterrows()):
    if len(monitoring_df[(monitoring_df.hadm_id == subj['hadm_id']) & (monitoring_df.label.isin(motor_labels))]) == 0:
        delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], nihss_item_labels] = None
        continue

    first_measurement_timepoint = monitoring_df[(monitoring_df.hadm_id == subj['hadm_id']) & (monitoring_df.label.isin(motor_labels))].sort_values(by=['charttime'], ascending=True).iloc[0].charttime

    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '1a_LOC'] = subj['1a_LOC'] - get_level_of_consciousness_1a(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '1b_LOCQuestions'] = subj['1b_LOCQuestions'] - get_level_of_orientation_1b(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '1c_LOCCommands'] = subj['1c_LOCCommands'] - get_level_of_command_response_1c(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '2_BestGaze'] = subj['2_BestGaze'] - get_best_gaze_2(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '3_Visual'] = subj['3_Visual'] - get_visual_fields_3(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '4_FacialPalsy'] = subj['4_FacialPalsy'] - get_facial_palsy_4(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '5a_LeftArm'] = (subj['5a_LeftArm'] + subj['5b_RightArm'] ) - get_motor_arms_5(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '5b_RightArm'] = delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '5a_LeftArm']
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '6a_LeftLeg'] = (subj['6a_LeftLeg'] + subj['6b_RightLeg']) - get_motor_legs_6(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '6b_RightLeg'] = delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '6a_LeftLeg']
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '7_LimbAtaxia'] = subj['7_LimbAtaxia'] - get_ataxia_7(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '8_Sensory'] = subj['8_Sensory'] - get_sensory_8(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '9_BestLanguage'] = subj['9_BestLanguage'] - get_language_9(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '10_Dysarthria'] = subj['10_Dysarthria'] - get_dysarthria_10(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)
    delta_nihss_df.loc[delta_nihss_df.hadm_id == subj['hadm_id'], '11_ExtinctionInattention'] = subj['11_ExtinctionInattention'] - get_extinction_11(monitoring_df, mimic_admission_df, subj['hadm_id'], first_measurement_timepoint)

# save delta nihss dataframe
delta_nihss_df.to_csv('delta_nihss.csv', index=False)