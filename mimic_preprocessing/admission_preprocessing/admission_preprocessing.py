import pandas as pd
import os
import numpy as np

from preprocessing.stroke_registry_params_preprocessing.admission_params_preprocessing import \
    restrict_variable_to_possible_ranges


def preprocess_admission(admission_notes_data_path:str, admission_table_path:str, verbose:bool = False) -> pd.DataFrame:
    """
    Preprocess the admission data manually extracted from admission / discharge notes

    Target output variables:
    1. [] age
    2. [] Sex
    3. [] Referral
    4. [ ] Prestroke disability (Rankin)
    6. [ ] Antihypert. drugs pre-stroke
    6. [ ] Lipid lowering drugs pre-stroke
    7. [ ] Antiplatelet drugs
    8. [ ] Anticoagulants
    9. [ ] MedHist Hypertension
    10. [ ] MedHist Diabetes
    11. [ ] MedHist Hyperlipidemia
    12. [ ] MedHist Smoking
    13. [ ] MedHist Atrial Fibr.
    14. [ ] MedHist CHD
    15. [ ] MedHist PAD
    16. [ ] MedHist cerebrovascular_event
    18. [ ] categorical_onset_to_admission_time
    18. [ ] wake_up_stroke
    1. [ ] categorical_IVT
    2. [ ] categorical_IAT


    """
    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(''))),
                                              'preprocessing/possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)


    # Preprocessing admission table data
    admission_table_df = pd.read_csv(admission_table_path)
    admission_table_df = admission_table_df[['subject_id', 'hadm_id', 'icustay_id', 'dob', 'admittime', 'age', 'gender', 'admission_location']]
    admission_table_df.drop_duplicates(inplace=True)

    if verbose:
        print('Preprocessing age')
    # for patients with age at 300; set to 90 (this is an artifact of the MIMIC database, to anonymize the data)
    admission_table_df.loc[admission_table_df['age'] > 250, 'age'] = 90
    admission_table_df, excluded_age_df = restrict_variable_to_possible_ranges(admission_table_df,
                                                                              'age', possible_value_ranges,
                                                                               verbose=verbose)

    admission_table_df.rename(columns={'gender': 'Sex'}, inplace=True)
    # encode 'Sex' to ['Female', 'Male']
    admission_table_df.loc[admission_table_df.Sex == 'F', 'Sex'] = 'Female'
    admission_table_df.loc[admission_table_df.Sex == 'M', 'Sex'] = 'Male'

    # transform referral to target variables: ['Emergency service (144)', 'Self referral or GP', 'Other hospital']
    admission_table_df.rename(columns={'admission_location': 'Referral'}, inplace=True)
    admission_table_df.loc[
        (admission_table_df['Referral'] == 'TRANSFER FROM HOSP/EXTRAM')
        | (admission_table_df['Referral'] == 'CLINIC REFERRAL/PREMATURE'), 'Referral'] = 'Other hospital'
    admission_table_df.loc[
        (admission_table_df['Referral'] == 'PHYS REFERRAL/NORMAL DELI'), 'Referral'] = 'Self referral or GP'
    admission_table_df.loc[
        (admission_table_df['Referral'] == 'EMERGENCY ROOM ADMIT') |
        (admission_table_df['Referral'] == 'TRANSFER FROM SKILLED NUR'), 'Referral'] = 'Emergency service (144)'

    if len(admission_table_df.Referral.unique()) != 3:
        raise ValueError('Referral variable has more than 3 unique values')

    admission_table_df = admission_table_df[['hadm_id', 'icustay_id', 'admittime', 'age', 'Sex', 'Referral']]
    admission_table_df = pd.melt(admission_table_df, id_vars=['hadm_id', 'icustay_id', 'admittime'],
                                var_name='sample_label')

    # Preprocessing admission notes data
    admission_data_df = pd.read_excel(admission_notes_data_path)

    # restrict to patients admitted to ICU with stroke as primary reason and with onset to admission < 7 d
    admission_data_df = admission_data_df[admission_data_df['admitted to ICU for stroke'] == 'y']
    admission_data_df = admission_data_df[admission_data_df['onset to ICU admission > 7d'] == 'n']

    med_hist_columns = ['Antihypert. drugs pre-stroke',
                        'Lipid lowering drugs pre-stroke',
                        'Antiplatelet drugs',
                        'Anticoagulants',
                        'MedHist Hypertension',
                        'MedHist Diabetes',
                        'MedHist Hyperlipidemia',
                        'MedHist Smoking',
                        'MedHist Atrial Fibr.',
                        'MedHist CHD',
                        'MedHist PAD']

    # transform values in med_hist_columns from ['y', 'n'] to ['yes', 'no']
    for col in med_hist_columns:
        admission_data_df[col] = admission_data_df[col].apply(lambda x: 'yes' if x == 'y' else x)
        admission_data_df[col] = admission_data_df[col].apply(lambda x: 'no' if x == 'n' else x)

    admission_data_df['MedHist cerebrovascular_event'] = admission_data_df['MedHist cerebrovascular_event'].apply(lambda x: 'True' if x == 'y' else x)
    admission_data_df['MedHist cerebrovascular_event'] = admission_data_df['MedHist cerebrovascular_event'].apply(lambda x: 'False' if x == 'n' else x)

    admission_data_df['wake up stroke'] = admission_data_df['wake up stroke'].apply(lambda x: 'True' if x == 'yes' else x)
    admission_data_df['wake up stroke'] = admission_data_df['wake up stroke'].apply(lambda x: 'False' if x == 'no' else x)

    # rename variables to DPI variables naming
    admission_data_df.rename(columns={'prestroke mRS': 'Prestroke disability (Rankin)',
                                      'admission NIHSS': 'NIHSS',
                                      'wake up stroke': 'wake_up_stroke'}, inplace=True)

    # preprocess timings
    date_format = '%Y-%m-%d %H:%M:%S'
    admission_data_df['onset_to_admission_min'] = (pd.to_datetime(admission_data_df['admittime'], format=date_format) -
                                                  pd.to_datetime(admission_data_df['stroke onset time']
                                                                 .replace(to_replace=r"unk(nown|own)", value=np.nan, regex=True),
                                                                format=date_format)).dt.total_seconds() / 60

    # verify that no onset_to_admission_min is over 7 days
    if len(admission_data_df[admission_data_df['onset_to_admission_min'] > 7 * 24 * 60]) > 0:
        raise ValueError('onset_to_admission_min is over 7 days')
    # verify that no onset_to_admission_min is negative
    if len(admission_data_df[admission_data_df['onset_to_admission_min'] < 0]) > 0:
        raise ValueError('onset_to_admission_min is negative')

    # Categorize admission timings
    # Categories: 'onset_unknown', intra_hospital, '<270min', '271-540min', '541-1440min', '>1440min'
    admission_data_df['categorical_onset_to_admission_time'] = pd.cut(
        admission_data_df['onset_to_admission_min'],
        bins=[-float("inf"), 270, 540, 1440, float("inf")],
        labels=['<270min', '271-540min', '541-1440min', '>1440min'])

    admission_data_df['categorical_onset_to_admission_time'] = admission_data_df[
        'categorical_onset_to_admission_time'].cat.add_categories('onset_unknown')
    admission_data_df.loc[admission_data_df.onset_to_admission_min.isna(), 'categorical_onset_to_admission_time'] = 'onset_unknown'

    # Preprocess procedures / treatments
    # IVT
    admission_data_df['IVT'] = ~admission_data_df['IVT time'].isna()

    admission_data_df['onset_to_IVT_min'] = (pd.to_datetime(admission_data_df['IVT time']
                                                                 .replace(to_replace=r"y", value=np.nan, regex=True),
                                                                format=date_format) -
                                            pd.to_datetime(admission_data_df['stroke onset time']
                                                           .replace(to_replace=r"unk(nown|own)", value=np.nan, regex=True),
                                                           format=date_format)).dt.total_seconds() / 60

    # verify that there are no negative IVT values
    if len(admission_data_df[admission_data_df['onset_to_IVT_min'] < 0]) > 0:
        raise ValueError('onset_to_IVT_min is negative')

    ## Categorizing IVT treatment
    # Categories:    'no_IVT', '<90min', '91-270min', '271-540min', '>540min'
    admission_data_df['categorical_IVT'] = pd.cut(admission_data_df['onset_to_IVT_min'],
                                                   bins=[-float("inf"), 90, 270, 540, float("inf")],
                                                   labels=['<90min', '91-270min', '271-540min', '>540min'])

    # For patients  with unknown IVT timing, replace NaN with mode
    admission_data_df.loc[(admission_data_df['categorical_IVT'].isna())
                              & (admission_data_df['IVT'] == True), 'categorical_IVT'] = \
                                    admission_data_df['categorical_IVT'].mode()[0]

    admission_data_df['categorical_IVT'] = admission_data_df['categorical_IVT'].cat.add_categories('no_IVT')
    admission_data_df['categorical_IVT'].fillna('no_IVT', inplace=True)

    # check that there is no occurrence with IVT time == y and categorical_IVT == no_IVT
    if len(admission_data_df[(admission_data_df['IVT time'] == 'y') & (admission_data_df['categorical_IVT'] == 'no_IVT')]) > 0:
        raise ValueError('IVT time == y and categorical_IVT == no_IVT')

    # IAT
    admission_data_df['IAT'] = ~admission_data_df['IAT time'].isna()

    admission_data_df['onset_to_IAT_min'] = (pd.to_datetime(admission_data_df['IAT time']
                                                                 .replace(to_replace=r"y", value=np.nan, regex=True),
                                                                format=date_format) -
                                            pd.to_datetime(admission_data_df['stroke onset time']
                                                           .replace(to_replace=r"unk(nown|own)", value=np.nan, regex=True),
                                                           format=date_format)).dt.total_seconds() / 60

    # verify that there are no negative IAT values
    if len(admission_data_df[admission_data_df['onset_to_IAT_min'] < 0]) > 0:
        raise ValueError('onset_to_IAT_min is negative')

    ## Categorizing IAT treatment
    # Categories: 'no_IAT', '<270min', '271-540min', '>540min'
    admission_data_df['categorical_IAT'] = pd.cut(admission_data_df['onset_to_IAT_min'],
                                                   bins=[-float("inf"), 270, 540, float("inf")],
                                                   labels=['<270min', '271-540min', '>540min'])

    # For patients  with unknown IAT timing, replace NaN with mode
    admission_data_df.loc[(admission_data_df['categorical_IAT'].isna())
                              & (admission_data_df['IAT'] == True), 'categorical_IAT'] = \
                                    admission_data_df['categorical_IAT'].mode()[0]

    admission_data_df['categorical_IAT'] = admission_data_df['categorical_IAT'].cat.add_categories('no_IAT')
    admission_data_df['categorical_IAT'].fillna('no_IAT', inplace=True)

    # check that there is no occurrence with IAT time == y and categorical_IAT == no_IAT
    if len(admission_data_df[(admission_data_df['IAT time'] == 'y') & (admission_data_df['categorical_IAT'] == 'no_IAT')]) > 0:
        raise ValueError('IAT time == y and categorical_IAT == no_IAT')

    id_columns = ['hadm_id',
    'icustay_id',
    'admittime'
     ]

    variable_columns = [
    'NIHSS',
    'Prestroke disability (Rankin)',
    'wake_up_stroke',
    'Antihypert. drugs pre-stroke',
    'Lipid lowering drugs pre-stroke',
    'Antiplatelet drugs',
    'Anticoagulants',
    'MedHist Hypertension',
    'MedHist Diabetes',
    'MedHist Hyperlipidemia',
    'MedHist Smoking',
    'MedHist Atrial Fibr.',
    'MedHist CHD',
    'MedHist PAD',
    'MedHist cerebrovascular_event',
    'categorical_onset_to_admission_time',
    'categorical_IVT',
    'categorical_IAT'
    ]

    admission_data_df = admission_data_df[id_columns + variable_columns]
    admission_data_df = pd.melt(admission_data_df, id_vars=id_columns,
                                var_name='sample_label')


    # restrict admission table to hadm_ids in the admission_data_df
    admission_table_df = admission_table_df[admission_table_df['hadm_id'].isin(admission_data_df['hadm_id'])]
    # append admission_table_df to admission_data_df
    admission_data_df = pd.concat([admission_data_df, admission_table_df], axis=0)

    return admission_data_df

