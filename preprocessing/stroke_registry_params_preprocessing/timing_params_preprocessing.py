import pandas as pd


def preprocess_timing_params(stroke_registry_df: pd.DataFrame) -> pd.DataFrame:
    stroke_registry_df = stroke_registry_df.copy()
    stroke_registry_df['patient_id'] = stroke_registry_df['Case ID'].apply(lambda x: x[8:-4])
    stroke_registry_df['EDS_last_4_digits'] = stroke_registry_df['Case ID'].apply(lambda x: x[-4:])
    stroke_registry_df['case_admission_id'] = stroke_registry_df['patient_id'].astype(str) \
                                              + stroke_registry_df['EDS_last_4_digits'].astype(str) \
                                              + '_' + pd.to_datetime(stroke_registry_df['Arrival at hospital'],
                                                                     format='%Y%m%d').dt.strftime('%d%m%Y').astype(str)
    stroke_registry_df['begin_date'] = pd.to_datetime(stroke_registry_df['Arrival at hospital'],
                                                      format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                       stroke_registry_df['Arrival time']

    stroke_registry_df['onset_datetime'] = pd.to_datetime(
        pd.to_datetime(stroke_registry_df['Onset date'], format='%Y%m%d').dt.strftime('%d-%m-%Y') \
        + ' ' + stroke_registry_df['Onset time'], format='%d-%m-%Y %H:%M')

    stroke_registry_df['onset_to_admission_min'] = (pd.to_datetime(stroke_registry_df['begin_date'],
                                                                      format='%d.%m.%Y %H:%M') - pd.to_datetime(
        stroke_registry_df['onset_datetime'], format='%d-%m-%Y %H:%M:%S')).dt.total_seconds() / 60

    ## Categorize admission timings
    # Categories: 'onset_unknown', intra_hospital, '<270min', '271-540min', '541-1440min', '>1440min'

    stroke_registry_df['categorical_onset_to_admission_time'] = pd.cut(
        stroke_registry_df['onset_to_admission_min'],
        bins=[-float("inf"), 270, 540, 1440, float("inf")],
        labels=['<270min', '271-540min', '541-1440min', '>1440min'])

    stroke_registry_df['categorical_onset_to_admission_time'] = stroke_registry_df[
        'categorical_onset_to_admission_time'].cat.add_categories('intra_hospital')
    stroke_registry_df['categorical_onset_to_admission_time'] = stroke_registry_df[
        'categorical_onset_to_admission_time'].cat.add_categories('onset_unknown')

    stroke_registry_df.loc[stroke_registry_df[
                                  'Referral'] == 'In-hospital event', 'categorical_onset_to_admission_time'] = 'intra_hospital'

    stroke_registry_df.loc[stroke_registry_df[
                                  'Time of symptom onset known'] == 'no', 'categorical_onset_to_admission_time'] = 'onset_unknown'

    # add variable to account for wake-up strokes
    stroke_registry_df['wake_up_stroke'] = stroke_registry_df['Time of symptom onset known'].apply(
        lambda x: True if x == 'wake up' else False)

    timing_columns = ['categorical_onset_to_admission_time', 'wake_up_stroke']
    timing_df = stroke_registry_df[timing_columns + ['case_admission_id', 'begin_date']]

    assert timing_df.isna().sum().sum() == 0, 'Missing values in timing data.'

    timing_df = pd.melt(timing_df, id_vars=['case_admission_id', 'begin_date'], var_name='sample_label')

    return timing_df
