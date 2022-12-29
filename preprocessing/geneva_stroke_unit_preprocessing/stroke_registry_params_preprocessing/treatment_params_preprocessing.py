import pandas as pd


def treatment_params_preprocessing(stroke_registry_df: pd.DataFrame) -> pd.DataFrame:
    stroke_registry_df = stroke_registry_df.copy()

    stroke_registry_df['onset_datetime'] = pd.to_datetime(
        pd.to_datetime(stroke_registry_df['Onset date'], format='%Y%m%d').dt.strftime('%d-%m-%Y') \
        + ' ' + pd.to_datetime(stroke_registry_df['Onset time'], format='%H:%M',
                                                       infer_datetime_format=True).dt.strftime('%H:%M'),
        format='%d-%m-%Y %H:%M')
    stroke_registry_df['IVT_datetime'] = pd.to_datetime(
        pd.to_datetime(stroke_registry_df['IVT start date'], format='%Y%m%d').dt.strftime('%d-%m-%Y') \
        + ' ' + pd.to_datetime(stroke_registry_df['IVT start time'], format='%H:%M',
                                                       infer_datetime_format=True).dt.strftime('%H:%M'),
        format='%d-%m-%Y %H:%M')
    stroke_registry_df['groin_puncture_datetime'] = pd.to_datetime(
        pd.to_datetime(stroke_registry_df['Date of groin puncture'], format='%Y%m%d').dt.strftime('%d-%m-%Y') \
        + ' ' + pd.to_datetime(stroke_registry_df['Time of groin puncture'], format='%H:%M',
                                                       infer_datetime_format=True).dt.strftime('%H:%M'),
        format='%d-%m-%Y %H:%M')

    stroke_registry_df['onset_to_IVT_min'] = (pd.to_datetime(stroke_registry_df['IVT_datetime'],
                                                                format='%d-%m-%Y %H:%M:%S') - pd.to_datetime(
        stroke_registry_df['onset_datetime'], format='%d-%m-%Y %H:%M:%S')).dt.total_seconds() / 60
    stroke_registry_df['onset_to_groin_min'] = (pd.to_datetime(stroke_registry_df['groin_puncture_datetime'],
                                                                  format='%d-%m-%Y %H:%M:%S') - pd.to_datetime(
        stroke_registry_df['onset_datetime'], format='%d-%m-%Y %H:%M:%S')).dt.total_seconds() / 60

    ## Categorizing IVT treatment
    # Categories:    'no_IVT', '<90min', '91-270min', '271-540min', '>540min'
    stroke_registry_df['categorical_IVT'] = pd.cut(stroke_registry_df['onset_to_IVT_min'],
                                                      bins=[-float("inf"), 90, 270, 540, float("inf")],
                                                      labels=['<90min', '91-270min', '271-540min', '>540min'])

    stroke_registry_df['categorical_IVT'] = stroke_registry_df['categorical_IVT'].cat.add_categories('no_IVT')
    stroke_registry_df['categorical_IVT'].fillna('no_IVT', inplace=True)

    # For patients  with unknown IVT timing, replace NaN with mode
    stroke_registry_df.loc[(stroke_registry_df['categorical_IVT'] == 'no_IVT')
                              & (stroke_registry_df['IVT with rtPA'] != 'no'), 'categorical_IVT'] = \
    stroke_registry_df[(stroke_registry_df['categorical_IVT'] != 'no_IVT')]['categorical_IVT'].mode()[0]

    ## Categorizing IAT treatment
    # Categories: 'no_IAT', '<270min', '271-540min', '>540min'
    stroke_registry_df['categorical_IAT'] = pd.cut(stroke_registry_df['onset_to_groin_min'],
                                                      bins=[-float("inf"), 270, 540, float("inf")],
                                                      labels=['<270min', '271-540min', '>540min'])
    stroke_registry_df['categorical_IAT'] = stroke_registry_df['categorical_IAT'].cat.add_categories('no_IAT')
    stroke_registry_df['categorical_IAT'].fillna('no_IAT', inplace=True)
    # For patients  with unknown IAT timing, replace NaN with mode
    stroke_registry_df.loc[(stroke_registry_df['categorical_IAT'] == 'no_IAT')
                              & (stroke_registry_df['IAT'] == 'yes'), 'categorical_IAT'] = \
    stroke_registry_df[(stroke_registry_df['categorical_IAT'] != 'no_IAT')]['categorical_IAT'].mode()[0]

    treatment_columns = ['categorical_IVT', 'categorical_IAT']
    treatment_df = stroke_registry_df[treatment_columns + ['case_admission_id', 'sample_date']]

    # verify that there is no missing data
    assert treatment_df.isna().sum().sum() == 0, 'Missing treatment data in stroke registry dataframe'

    treatment_df = pd.melt(treatment_df, id_vars=['case_admission_id', 'sample_date'], var_name='sample_label')

    return treatment_df

