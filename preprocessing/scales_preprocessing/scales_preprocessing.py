import pandas as pd
import os
import numpy as np

from preprocessing.utils import create_ehr_case_identification_column


def restrict_variable_to_possible_ranges(df, variable_name, possible_value_ranges, verbose=False):
    """
    Restricts a variable to the possible ranges in the possible_value_ranges dataframe.
    """
    variable_range = possible_value_ranges[possible_value_ranges['variable_label'] == variable_name]
    variable_range = variable_range.iloc[0]
    clean_df = df.copy()
    # set score to np.nan if outside of range
    clean_df.loc[(df['scale'] == variable_name) & (df['score'] < variable_range['Min']), 'score'] = np.nan
    clean_df.loc[(df['scale'] == variable_name) & (df['score'] > variable_range['Max']), 'score'] = np.nan
    if verbose:
        print(f'Excluding {clean_df.score.isna().sum()} observations because out of range')
    excluded_df = df[clean_df.score.isna()]
    clean_df = clean_df.dropna()
    return clean_df, excluded_df

def preprocess_scales(scales_df, eds_df, verbose=False):
    """
    Preprocesses the scales dataframe.
    eds_df is necessary as patient_id in scales_df was overwritten by eds_final_patient_id
    :param scales_df:
    :param eds_df:
    :param verbose:
    :return:
    """

    scales_df['original_patient_id'] = scales_df['patient_id'].apply(lambda x: eds_df[eds_df['eds_final_patient_id'] == x]['patient_id'].iloc[0])

    scales_df['case_admission_id'] = create_ehr_case_identification_column(scales_df)

    columns_to_drop = ['nr', 'patient_id', 'eds_end_4digit', 'eds_manual', 'DOB', 'begin_date',
                       'end_date', 'death_date', 'death_hosp', 'eds_final_id',
                       'eds_final_begin', 'eds_final_end', 'eds_final_patient_id', 'original_patient_id',
                       'eds_final_birth', 'eds_final_death', 'eds_final_birth_str',
                       'date_from', 'date_to', 'patient_id_manual', 'stroke_onset_date', 'Referral', 'match_by']
    scales_df.drop(columns_to_drop, axis=1, inplace=True)
    possible_value_ranges_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            'possible_ranges_for_variables.xlsx')
    possible_value_ranges = pd.read_excel(possible_value_ranges_file)

    glasgow_equivalents = ['Glasgow + pupilles', 'Glasgow + pupilles + sensibilité/motricité', 'Glasgow',
                           'Glasgow  urgence',
                           'Neurologie - Glasgow']
    scales_df.loc[scales_df['scale'].isin(glasgow_equivalents), 'scale'] = 'Glasgow Coma Scale'

    NIHSS_equivalents = ['NIHSS - National Institute oh Health Stroke Scale',
                         'NIHSS - National Institute of Health Stroke Scale']
    scales_df.loc[scales_df['scale'].isin(NIHSS_equivalents), 'scale'] = 'NIHSS'

    pain_scale_equivalents = ['Douleur - b - Echelle numérique', 'Douleur - a - EVA', 'Douleur - c - Echelle verbale']
    scales_df.loc[scales_df['scale'].isin(pain_scale_equivalents), 'scale'] = 'pain scale'
    # drop rows with scale = 'Douleur - h - CPOT' as not comparable with other scales
    scales_df.drop(scales_df[scales_df['scale'].str.contains('CPOT')].index, inplace=True)

    # convert score to float
    scales_df['score'] = pd.to_numeric(scales_df['score'], errors='coerce')

    if verbose:
        print('Preprocessing NIHSS')
    cleaned_scales_df, _ = restrict_variable_to_possible_ranges(scales_df, 'NIHSS',
                                                                                possible_value_ranges, verbose=verbose)
    if verbose:
        print('Glasgow Coma Scale')
    cleaned_scales_df, _ = restrict_variable_to_possible_ranges(cleaned_scales_df,
                                                                                  'Glasgow Coma Scale',
                                                                                  possible_value_ranges, verbose=verbose)
    return cleaned_scales_df

