from typing import List

import numpy as np
import pandas as pd


def get_most_recent_measurements(monitoring_df: pd.DataFrame, hadm_id: int, reference_date_time: str,
                                 equivalent_measurement_labels: str, allowed_values: list) -> pd.DataFrame:
    """
    Method to extract the most recent measurements from the monitoring dataframe
    For a given hadm_id and reference_date_time, the most recent measurements of measurement labels is extracted, by finding the most recent measurement (occurring before) allowing only a given set of values.

    Args:
        monitoring_df: the monitoring dataframe
        hadm_id: the hadm_id of the patient
        reference_date_time: the reference date time
        equivalent_measurement_labels: the measurement labels to be considered
        allowed_values: the allowed values for the measurement labels
    """
    available_measurements = monitoring_df[
        (monitoring_df.hadm_id == hadm_id) & (monitoring_df.charttime <= reference_date_time) & (
            monitoring_df.label.isin(equivalent_measurement_labels)) & (monitoring_df.value.isin(allowed_values))]

    if len(available_measurements) == 0:
        return None

    most_recent_measurement_date_time = available_measurements.sort_values(by=['charttime'], ascending=False).iloc[
        0].charttime

    most_recent_measurements = monitoring_df[
        (monitoring_df.hadm_id == hadm_id) & (monitoring_df.charttime == most_recent_measurement_date_time)
        & (monitoring_df.label.isin(equivalent_measurement_labels)) & (monitoring_df.value.isin(allowed_values))]

    return most_recent_measurements


def flatten(list):
    return [item for sublist in list for item in sublist]


def most_recent_measurement_score(monitoring_df: pd.DataFrame, hadm_id: int, reference_date_time: str,
                                  measurement_labels_by_preference: str, graded_allowed_values: List[List]) -> int:
    """
    Method to extract the most recent measurement score from the monitoring dataframe
    For a given hadm_id and reference_date_time, the most recent measurement score is extracted, by finding the most recent measurement (occurring before) allowing only a given set of values.

    Args:
        monitoring_df: the monitoring dataframe
        hadm_id: the hadm_id of the patient
        reference_date_time: the reference date time
        measurement_labels_by_preference: the measurement labels to be considered, ordered by preference (i.e. the first label is preferred over the second label)
        graded_allowed_values: a list of lists of allowed value where the index in the first list corresponds to the score

    Returns:
        the score (int)
    """

    all_allowed_values = flatten(graded_allowed_values)

    most_recent_measurements = get_most_recent_measurements(monitoring_df, hadm_id, reference_date_time,
                                                            measurement_labels_by_preference, all_allowed_values)

    if most_recent_measurements is None:
        return None

    # select label occurring at most recent measurment by preference of label order given by measurement_labels_by_preference
    available_labels = most_recent_measurements.label.values
    selected_label = [label for label in measurement_labels_by_preference if label in available_labels][0]

    # select the value of the selected label
    measurement_value = most_recent_measurements[most_recent_measurements.label == selected_label].iloc[0].value

    # transform measurement value into a graded NIHSS item score
    for i, allowed_values in enumerate(graded_allowed_values):
        if measurement_value in allowed_values:
            return i


def get_score_from_mimic_admission_database(mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                                            score_label: str) -> int:
    """
    Method to extract a score from the mimic admission nihss database
        If extraction is not possible, 0 is returned
    """
    if hadm_id not in mimic_admission_nihss_df.hadm_id.values:
        return 0

    admission_score = mimic_admission_nihss_df[mimic_admission_nihss_df.hadm_id == hadm_id][score_label].iloc[0]

    if pd.isna(admission_score):
        return 0
    else:
        return int(admission_score)


def get_level_of_consciousness_1a(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                                  reference_date_time: str) -> pd.DataFrame:
    """
    Method to extract the level of consciousness (NIHSS 1a) from the monitoring dataframe
    1a. Level of Consciousness
         - 0 = Alert; keenly responsive.
         - 1 = Not alert; but arousable by minor stimulation to obey, answer, or respond.
         - 2 = Not alert; requires repeated stimulation to attend, or is obtunded and requires strong or painful stimulation to make movements (not stereotyped).
         - 3 = Responds only with reflex motor or autonomic effects or totally unresponsive, flaccid, and areflexic.

    For a given hadm_id and reference_date_time, the level of consciousness is extracted, by finding the most recent LOC measurement (occurring before).

    """
    consciousness_labels = ['Level of Consciousness', 'Level of Conscious',
                            'Richmond-RAS Scale', 'Riker-SAS Scale', 'GCS Total', 'Ramsey SedationScale',
                            'PAR-Consciousness']

    # 0 = Alert; keenly responsive.
    loc_1a_0_equivalents = ['Alert', 'Fully awake', '15', '14', '+1 Anxious, apprehensive, but not aggressive',
                            '+4 Combative, violent, danger to staff'
                            '+2 Frequent nonpurposeful movement, fights ventilator'
                            '+3 Pulls or removes tube(s) or catheter(s); aggressive', 'Calm/Cooperative', 'Agitated',
                            'Very Agitated', 'Danger Agitation',
                            'Awake, Anxious', 'Awake, Oriented', 'Awake, Commands']
    # 1 = Not alert; but arousable by minor stimulation to obey, answer, or respond.
    loc_1a_1_equivalents = ['Arouse to Stimulation', 'Arouse to Voice', 'Arousable on calling', '13', '12', '11', '10',
                            '9',
                            '-1 Awakens to voice (eye opening/contact) > 10 sec' ' 0  Alert and calm',
                            '-2 Light sedation, briefly awakens to voice (eye opening/contact) < 10 sec', 'Sedated',
                            'Asleep, Brisk', 'Dozing Intermit'
                            ]
    # 2 = Not alert; requires repeated stimulation to attend, or is obtunded and requires strong or painful stimulation to make movements (not stereotyped).
    loc_1a_2_equivalents = ['Lethargic', 'Arouse to Pain', 'Arouse to Pain', '8', '7',
                            '-3 Moderate sedation, movement or eye opening; No eye contact',
                            '-4 Deep sedation, no response to voice, but movement or eye opening to physical stimulation',
                            'Very Sedated', 'Asleep, Sluggish']

    # 3 = Responds only with reflex motor or autonomic effects or totally unresponsive, flaccid, and areflexic.
    loc_1a_3_equivalents = ['Unresponsive', '6', '5', '4', '3', 'Not responding',
                            '-5 Unarousable, no response to voice or physical stimulation', 'Unarousable']

    graded_equivalents = [loc_1a_0_equivalents, loc_1a_1_equivalents, loc_1a_2_equivalents, loc_1a_3_equivalents]

    most_recent_score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, consciousness_labels,
                                                      graded_equivalents)

    if most_recent_score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '1a_LOC')
    else:
        return most_recent_score


def get_level_of_orientation_1b(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                                reference_date_time: str) -> pd.DataFrame:
    '''
    Method to extract the level of orientation (NIHSS 1b) from the monitoring dataframe
    1b. Level of Orientation
        LOC Questions: The patient is asked the month and his/her age.
            - 0 = Answers both questions correctly.
            - 1 = Answers one question correctly.
            - 2 = Answers neither question correctly.
        Patients unable to speak because of endotracheal intubation, orotracheal trauma, severe dysarthria from any cause, language barrier, or any other problem not secondary to aphasia are given a 1.

    For a given hadm_id and reference_date_time, the level of orientation is extracted, by finding the most recent measurement (occurring before).
    '''

    orientation_labels = ['Orientation', 'Orientation to Place', 'Orientation to Time', 'Orient/Clouding Sensory']

    # 0 = Answers both questions correctly.
    loc_1b_0_equivalents = ['Oriented x 3', 'Oriented x3', 'Both', 'Year, month, & day' 'Year & month',
                            'Oriented and can do serial conditions']

    # 1 = Answers one question correctly.
    loc_1b_1_equivalents = ['Oriented x 2', 'Oriented x 1', 'Oriented x2', 'Oriented x1', 'Hospital', 'BIDMC', 'Year',
                            'Disoriented by date for < 2 days', 'Disoriented by date for > 2 days',
                            'Unable to Assess']

    # 2 = Answers neither question correctly.
    loc_1b_2_equivalents = ['Disoriented', 'None', 'Cannot do additions and is uncertain of date']

    graded_equivalents = [loc_1b_0_equivalents, loc_1b_1_equivalents, loc_1b_2_equivalents]

    most_recent_score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, orientation_labels,
                                                      graded_equivalents)

    if most_recent_score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '1b_LOCQuestions')
    else:
        return most_recent_score


def get_level_of_command_response_1c(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                                     reference_date_time: str) -> pd.DataFrame:
    '''
    Method to extract the level of command response (NIHSS 1c) from the monitoring dataframe

    1c. LOC Commands
        - 0 = Performs both tasks correctly.
        - 1 = Performs one task correctly.
        - 2 = Performs neither task correctly.
    For a given hadm_id and reference_date_time, the level of command response is extracted, by finding the most recent measurement (occurring before).
    '''

    command_response_labels = ['Follows Commands', 'Commands Response']

    # 0 = Responds appropriately to all three commands.
    loc_1c_0_equivalents = ['Consistently']

    # 1 = Responds appropriately to one command.
    loc_1c_1_equivalents = ['Inconsistently']

    # 2 = Responds appropriately to none of the commands.
    loc_1c_2_equivalents = ['None', 'No']

    grading_allowed_values = [loc_1c_0_equivalents, loc_1c_1_equivalents, loc_1c_2_equivalents]

    most_recent_score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time,
                                                      command_response_labels, grading_allowed_values)

    if most_recent_score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '1c_LOCCommands')
    else:
        return most_recent_score


def get_best_gaze_2(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                    reference_date_time: str):
    """
    Grading:
    0 = Normal.
    1 = Partial gaze palsy; gaze is abnormal in one or both eyes, but forced deviation or total gaze paresis is not present.
    2 = Forced deviation, or total gaze paresis not overcome by the oculocephalic maneuver.

    No good data in monitoring dataframe, therefore, if available, this value is extracted from the mimic nihss admission database
    """

    return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '2_BestGaze')


def get_visual_fields_3(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                        reference_date_time: str):
    """
    3. Visual: Visual fields (upper and lower quadrants) are tested by confrontation
         0 = No visual loss.
         1 = Partial hemianopia.
         2 = Complete hemianopia.
         3 = Bilateral hemianopia
    """

    visual_field_labels = ['Visual Field Cut']

    # 0 = No visual loss.
    visual_field_0_equivalents = ['No']

    # 1 = Partial hemianopia./ 2 = Complete hemianopia.
    visual_field_X_equivalents = ['Left', 'Right']

    grading_allowed_values = [visual_field_0_equivalents, visual_field_X_equivalents]

    most_recent_score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, visual_field_labels,
                                                      grading_allowed_values)

    if most_recent_score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '3_Visual')
    elif most_recent_score == 1:
        admission_score = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '3_Visual')
        if admission_score == 0:
            # the visual field loss is new
            return 2
        # the visual field loss is not new, hence grade as upon admission
        else:
            return admission_score
    else:
        return most_recent_score


def get_facial_palsy_4(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                       reference_date_time: str):
    """
    Facial Palsy
         0 = Normal symmetrical movements.
         1 = Minor paralysis (flattened nasolabial fold, asymmetry on smiling).
         2 = Partial paralysis (total or near-total paralysis of lower face).
         3 = Complete paralysis of one or both sides (absence of facial movement in the upper and lower face).
    """
    facial_palsy_labels = ['Facial Droop', 'Face Droop']

    # 0 = Normal symmetrical movements.
    facial_palsy_0_equivalents = ['No']

    # All other levels
    facial_palsy_X_equivalents = ['Left', 'Right', 'LEFT', 'left']

    grading_allowed_values = [facial_palsy_0_equivalents, facial_palsy_X_equivalents]

    most_recent_score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, facial_palsy_labels,
                                                      grading_allowed_values)

    if most_recent_score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '4_FacialPalsy')

    elif most_recent_score == 1:
        admission_score = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '4_FacialPalsy')
        if admission_score == 0:
            # the facial palsy is new
            return 2
        # the facial palsy is not new, hence grade as upon admission
        else:
            return admission_score
    else:
        return most_recent_score


def get_motor_arms_5(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                     reference_date_time: str):
    """
    Motor Arm L & R
        0 = No drift; limb holds 90 (or 45) degrees for full 10 seconds.
        1 = Drift; limb holds 90 (or 45) degrees, but drifts down before full 10 seconds; does not hit bed or other support.
        2 = Some effort against gravity; limb cannot get to or maintain (if cued) 90 (or 45) degrees, drifts down to bed, but has some effort against gravity.
        3 = No effort against gravity; limb falls.
        4 = No movement.

    The following items will not be scored: 'Posturing', 'Other/Remarks', 'Moves on Bed'

    """
    motor_ru_labels = ['RU Strength/Movement', 'Strength R Arm']
    motor_lu_labels = ['LU Strength/Movement', 'Strength L Arm']

    # 0 = No drift; limb holds 90 (or 45) degrees for full 10 seconds.
    motor_upper_limb_0_equivalents = ['Full resistance', 'Some resistance', 'Normal Strength', 'Normal strength',
                                      'Lifts and Holds']

    # 1 = Drift; limb holds 90 (or 45) degrees, but drifts down before full 10 seconds; does not hit bed or other support.
    motor_upper_limb_1_equivalents = ['Lifts against gravity, no resistance']

    # 2 = Some effort against gravity; limb cannot get to or maintain (if cued) 90 (or 45) degrees, drifts down to bed, but has some effort against gravity.
    motor_upper_limb_2_equivalents = ['Lifts/falls back', 'Lifts/Falls Back']

    # 3 = No effort against gravity; limb falls.
    motor_upper_limb_3_equivalents = ['Movement, but not against gravity']

    # 4 = No movement.
    motor_upper_limb_4_equivalents = ['No movement', 'No Movement']

    grading_allowed_values = [motor_upper_limb_0_equivalents, motor_upper_limb_1_equivalents,
                              motor_upper_limb_2_equivalents, motor_upper_limb_3_equivalents,
                              motor_upper_limb_4_equivalents]

    # Left upper limb
    score_lu = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, motor_lu_labels,
                                             grading_allowed_values)
    if score_lu is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        score_lu = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '5a_LeftArm')

    # Right upper limb
    score_ru = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, motor_ru_labels,
                                             grading_allowed_values)
    if score_ru is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        score_ru = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '5b_RightArm')

    return score_ru + score_lu


def get_motor_legs_6(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                     reference_date_time: str):
    """
    Motor Leg L & R
        0 = No drift; leg holds 30-degree position for full 5 seconds.
        1 = Drift; leg falls by the end of the 5-second period but does
        not hit bed.
        2 = Some effort against gravity; leg falls to bed by 5
        seconds, but has some effort against gravity.
        3 = No effort against gravity; leg falls to bed immediately.
        4 = No movement.

    The following items will not be scored: 'Posturing', 'Other/Remarks', 'Moves on Bed'

    """

    motor_rl_labels = ['RL Strength/Movement', 'Strength R Leg']
    motor_ll_labels = ['LL Strength/Movement', 'Strength L Leg']

    # 0 = No drift; leg holds 30-degree position for full 5 seconds.
    motor_lower_limb_0_equivalents = ['Full resistance', 'Some resistance', 'Normal Strength', 'Normal strength',
                                      'Lifts and Holds']

    # 1 = Drift; leg falls by the end of the 5-second period but does not hit bed.
    motor_lower_limb_1_equivalents = ['Lifts against gravity, no resistance']

    # 2 = Some effort against gravity; leg falls to bed by 5 seconds, but has some effort against gravity.
    motor_lower_limb_2_equivalents = ['Lifts/falls back', 'Lifts/Falls Back']

    # 3 = No effort against gravity; leg falls to bed immediately.
    motor_lower_limb_3_equivalents = ['Movement, but not against gravity']

    # 4 = No movement.
    motor_lower_limb_4_equivalents = ['No movement', 'No Movement']

    grading_allowed_values = [motor_lower_limb_0_equivalents, motor_lower_limb_1_equivalents,
                              motor_lower_limb_2_equivalents, motor_lower_limb_3_equivalents,
                              motor_lower_limb_4_equivalents]

    # Left lower limb
    score_ll = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, motor_ll_labels,
                                             grading_allowed_values)
    if score_ll is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        score_ll = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '6a_LeftLeg')

    # Right lower limb
    score_rl = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, motor_rl_labels,
                                             grading_allowed_values)
    if score_rl is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        score_rl = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '6b_RightLeg')

    return score_rl + score_ll


def get_ataxia_7(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                 reference_date_time: str):
    """
    7. Limb Ataxia
         0 = Absent.
         1 = Present in one limb.
         2 = Present in two limbs
    """
    ataxia_labels = ['Ataxia']
    # 0 = Absent.
    ataxia_0_equivalents = ['No']
    # X, Present
    ataxia_X_equivalents = ['Yes']

    grading_allowed_values = [ataxia_0_equivalents, ataxia_X_equivalents]

    score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, ataxia_labels,
                                          grading_allowed_values)

    if score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '7_LimbAtaxia')
    if score == 1:
        admission_score = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '7_LimbAtaxia')
        if admission_score != 0:
            return admission_score
    return score


def get_sensory_8(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                  reference_date_time: str):
    """
    8. Sensory
        0 = Normal; no sensory loss.
        1 = Mild-to-moderate sensory loss; patient feels pinprick is
        less sharp or is dull on the affected side; or there is a
        loss of superficial pain with pinprick, but patient is aware
        of being touched.
        2 = Severe to total sensory loss; patient is not aware of
        being touched in the face, arm, and leg. Or if Coma or if bilateral
    """
    sensory_l_labels = ['LUE Sensation', 'LLE Sensation', 'LLE [Sensation]', 'LUE [Sensation]']
    sensory_r_labels = ['RUE Sensation', 'RLE Sensation', 'RLE [Sensation]', 'RUE [Sensation]']
    overall_sensory_labels = ['Braden Sensory Perception', 'Braden SensoryPercep']

    # 0 = Normal; no sensory loss.
    sensory_0_equivalents = ['Intact']
    # 1 = Mild-to-moderate sensory loss; patient feels pinprick is less sharp or is dull on the affected side; or there is a loss of superficial pain with pinprick, but patient is aware of being touched.
    sensory_1_equivalents = ['Impaired']
    # 2 = Severe to total sensory loss; patient is not aware of being touched in the face, arm, and leg. Or if Coma or if bilateral
    sensory_2_equivalents = ['Absent']

    grading_allowed_values = [sensory_0_equivalents, sensory_1_equivalents, sensory_2_equivalents]
    # get sensory on left side
    score_l = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, sensory_l_labels,
                                            grading_allowed_values)
    # get sensory on right side
    score_r = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, sensory_r_labels,
                                            grading_allowed_values)

    # if both sides are None, check overall sensory
    if score_l is None and score_r is None:
        overall_sensory_0_equivalents = ['No Impairment']
        overall_sensory_1_equivalents = ['Slight Impairment', 'Sl. Limited', 'Very Limited']
        overall_sensory_2_equivalents = ['Completely Limited', 'Comp. Limited']
        overall_grading_allowed_values = [overall_sensory_0_equivalents, overall_sensory_1_equivalents,
                                          overall_sensory_2_equivalents]
        score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, overall_sensory_labels,
                                              overall_grading_allowed_values)
        if score is None:
            # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
            return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '8_Sensory')
        return score

    # if one side is None, return the other side
    if score_l is None:
        return score_r
    if score_r is None:
        return score_l

    # if both sides are not null, return two
    if score_l != 0 and score_r != 0:
        return 2

    # otherwise, return sum (one of them is 0)
    return score_l + score_r


def get_language_9(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                   reference_date_time: str):
    """
    9. Best Language:
        0 = No aphasia; normal.
        1 = Mild-to-moderate aphasia;
        2 = Severe aphasia;
        3 = Mute, global aphasia; no usable speech or auditory comprehension. Or coma.
    """

    language_labels = ['Speech']

    # 0 = No aphasia; normal.
    language_0_equivalents = ['Normal']
    # 1 = Mild-to-moderate aphasia; & 2 = Severe aphasia;
    language_1_and_2_equivalents = ['Aphasic']
    # 3 = Mute, global aphasia; no usable speech or auditory comprehension. Or coma.
    language_3_equivalents = ['Mute']

    grading_allowed_values = [language_0_equivalents, language_1_and_2_equivalents, language_3_equivalents]

    score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, language_labels,
                                          grading_allowed_values)

    if score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '9_BestLanguage')

    if score == 0:
        return score

    if score == 1:
        admission_score = get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '9_BestLanguage')
        if admission_score != 0:
            return admission_score
        return 1

    if score == 2:
        return 3


def get_dysarthria_10(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                      reference_date_time: str):
    """
    10. Dysarthria
        0 = Normal.
        1 = Mild-to-moderate dysarthria; patient slurs at least some words and, at worst, can be understood with some difficulty.
        2 = Severe dysarthria; patient's speech is so slurred as to be unintelligible in the absence of or out of proportion to any dysphasia, or is mute/anarthric.
        If intubated UN
    """
    dysarthria_labels = ['Speech', 'Slurred Speech']

    # 0 = Normal.
    dysarthria_0_equivalents = ['Normal', 'No']

    # 1 = Mild-to-moderate dysarthria; patient slurs at least some words and, at worst, can be understood with some difficulty.
    dysarthria_1_equivalents = ['Yes', 'Slurred', 'Dysarthric']

    # 2 = Severe dysarthria; patient's speech is so slurred as to be unintelligible in the absence of or out of proportion to any dysphasia, or is mute/anarthric.
    dysarthria_2_equivalents = ['Mute', 'Garbled', 'Aphasic']

    grading_allowed_values = [dysarthria_0_equivalents, dysarthria_1_equivalents, dysarthria_2_equivalents]

    score = most_recent_measurement_score(monitoring_df, hadm_id, reference_date_time, dysarthria_labels,
                                          grading_allowed_values)

    if score is None:
        # If the most recent score is None, the admission database is queried, if no measurement is found there, 0 is returned.
        return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '10_Dysarthria')

    return score


def get_extinction_11(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
                      reference_date_time: str):
    """
    11. Extinction and Inattention
        0 = No abnormality.
        1 = Visual, tactile, auditory, spatial, or personal inattention or extinction to bilateral simultaneous stimulation in one of the sensory modalities.
        2 = Profound hemi-inattention or extinction to more than one modality; does not recognize own hand or orients to only one side of space.

    No continuous measurement is available for this item. The admission database is queried.
    """

    return get_score_from_mimic_admission_database(mimic_admission_nihss_df, hadm_id, '11_ExtinctionInattention')


def get_nihss(monitoring_df: pd.DataFrame, mimic_admission_nihss_df: pd.DataFrame, hadm_id: int,
              reference_date_time: str):
    """
    NIHSS is the sum of the 11 items.
    """

    return np.sum([
        get_level_of_consciousness_1a(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_level_of_orientation_1b(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_level_of_command_response_1c(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_best_gaze_2(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_visual_fields_3(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_facial_palsy_4(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_motor_arms_5(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_motor_legs_6(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_ataxia_7(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_sensory_8(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_language_9(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_dysarthria_10(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time),
        get_extinction_11(monitoring_df, mimic_admission_nihss_df, hadm_id, reference_date_time)
    ])
