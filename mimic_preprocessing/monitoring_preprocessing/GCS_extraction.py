import pandas as pd


def get_GCS(monitoring_df: pd.DataFrame, hadm_id: int, reference_date_time: str):
    """
    Extracts the Glasgow Coma Scale (GCS) from the monitoring dataframe
    If no pre-computed GCS is found, it is computed from the component scores
    """

    # if 'GCS Total' already exists, return that
    if len(monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'] == 'GCS Total') & (monitoring_df['charttime'] == reference_date_time)
           & (~monitoring_df['value'].isna())]) > 0:
        return monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'] == 'GCS Total') & (monitoring_df['charttime'] == reference_date_time)]['value'].values[0]

    # otherwise, compute it from the component scores
    motor_components = ['GCS - Motor Response', 'Motor Response']
    verbal_components = ['GCS - Verbal Response', 'Verbal Response']
    eye_components = ['GCS - Eye Opening', 'Eye Opening']

    motor_1_equivalents = ['1 No Response', 'No response']
    motor_2_equivalents = ['2 Abnorm extensn', 'Abnormal extension']
    motor_3_equivalents = ['3 Abnorm flexion', 'Abnormal Flexion']
    motor_4_equivalents = ['4 Flex-withdraws', 'Flex-withdraws']
    motor_5_equivalents = ['5 Localizes Pain', 'Localizes Pain']
    motor_6_equivalents = ['6 Obeys Commands', 'Obeys Commands']

    verbal_1_equivalents = ['1 No Response', '1.0 ET/Trach', 'No Response-ETT', 'No response', 'No Response']
    verbal_2_equivalents = ['2 Incomp sounds', 'Incomprehensible sounds']
    verbal_3_equivalents = ['3 Inapprop words', 'Inappropriate Words']
    verbal_4_equivalents = ['4 Confused', 'Confused']
    verbal_5_equivalents = ['5 Oriented', 'Oriented']

    eye_1_equivalents = ['1 No Response', 'No Response', 'None']
    eye_2_equivalents = ['2 To Pain', 'To Pain']
    eye_3_equivalents = ['3 To Speech', 'To Speech']
    eye_4_equivalents = ['4 Spontaneously', 'Spontaneously']

    # get the motor component
    # first search at the reference date time, then search at all time points prior to the reference date time and select the most recent
    motor_df = monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'].isin(motor_components)) & (monitoring_df['charttime'] == reference_date_time) & (~monitoring_df['value'].isna())]
    if len(motor_df) == 0:
        motor_df = monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'].isin(motor_components)) & (monitoring_df['charttime'] < reference_date_time) & (~monitoring_df['value'].isna())]
    if len(motor_df) == 0:
        motor_score = 6 # if no motor score is found, assume the patient is responsive
    else:
        motor_response = motor_df.sort_values(by='charttime', ascending=False).iloc[0].value

        # transform motor component to motor score
        if motor_response in motor_1_equivalents:
            motor_score = 1
        elif motor_response in motor_2_equivalents:
            motor_score = 2
        elif motor_response in motor_3_equivalents:
            motor_score = 3
        elif motor_response in motor_4_equivalents:
            motor_score = 4
        elif motor_response in motor_5_equivalents:
            motor_score = 5
        elif motor_response in motor_6_equivalents:
            motor_score = 6
        else:
            raise ValueError('Motor response not recognized')

    # get the verbal component
    verbal_df = monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'].isin(verbal_components)) & (monitoring_df['charttime'] == reference_date_time) & (~monitoring_df['value'].isna())]
    if len(verbal_df) == 0:
        verbal_df = monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'].isin(verbal_components)) & (monitoring_df['charttime'] < reference_date_time) & (~monitoring_df['value'].isna())]
    if len(verbal_df) == 0:
        verbal_score = 5
    else:
        verbal_response = verbal_df.sort_values(by='charttime', ascending=False).iloc[0].value
        # transform verbal component to verbal score
        if verbal_response in verbal_1_equivalents:
            verbal_score = 1
        elif verbal_response in verbal_2_equivalents:
            verbal_score = 2
        elif verbal_response in verbal_3_equivalents:
            verbal_score = 3
        elif verbal_response in verbal_4_equivalents:
            verbal_score = 4
        elif verbal_response in verbal_5_equivalents:
            verbal_score = 5
        else:
            raise ValueError('Verbal response not recognized')

    # get the eye component
    eye_df = monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'].isin(eye_components)) & (monitoring_df['charttime'] == reference_date_time) & (~monitoring_df['value'].isna())]
    if len(eye_df) == 0:
        eye_df = monitoring_df[(monitoring_df['hadm_id'] == hadm_id) & (monitoring_df['label'].isin(eye_components)) & (monitoring_df['charttime'] < reference_date_time) & (~monitoring_df['value'].isna())]
    if len(eye_df) == 0:
        eye_score = 4
    else:
        eye_response = eye_df.sort_values(by='charttime', ascending=False).iloc[0].value
        # transform eye component to eye score
        if eye_response in eye_1_equivalents:
            eye_score = 1
        elif eye_response in eye_2_equivalents:
            eye_score = 2
        elif eye_response in eye_3_equivalents:
            eye_score = 3
        elif eye_response in eye_4_equivalents:
            eye_score = 4
        else:
            raise ValueError('Eye response not recognized')

    return motor_score + verbal_score + eye_score


