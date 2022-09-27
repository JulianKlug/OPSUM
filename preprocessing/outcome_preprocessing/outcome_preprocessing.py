import pandas as pd
import numpy as np
import os

from preprocessing.patient_selection.restrict_to_patient_selection import restrict_to_patient_selection
from preprocessing.utils import create_registry_case_identification_column

outcome_columns = ["Symptomatic ICH",
"Symptomatic ICH date",
"Recurrent stroke",
"Recurrent stroke date",
"Orolingual angioedema",
"Death in hospital",
"Death at hospital date",
"Death at hospital time",
"Death at hospital cause",
"Epileptic seizure in hospital",
"Epileptic seizure in hospital date",
"Decompr. craniectomy",
"Decompr. craniectomy date",
"CEA",
"CEA date",
"CAS",
"CAS date",
"Other endovascular revascularization",
"Other surgical revascularization",
"Other surgical revascularization date",
"Other surgical revascularization spec",
"PFO closure",
"PFO closure date",
"Discharge destination",
"Discharge date",
"Discharge time",
"Duration of hospital stay (days)",
"3M date",
"3M mode",
"3M mRS",
"3M NIHSS","3M Stroke",
"3M Stroke date",
"3M ICH", '3M ICH date', '3M Death', '3M Death date', '3M Death cause',
       '3M Epileptic seizure', '3M Epileptic seizure date', '3M delta mRS']


def preprocess_outcomes(stroke_registry_data_path, patient_selection_path, verbose:bool=True):
    stroke_registry_df = pd.read_excel(stroke_registry_data_path)

    stroke_registry_df['patient_id'] = stroke_registry_df['Case ID'].apply(lambda x: x[8:-4])
    stroke_registry_df['EDS_last_4_digits'] = stroke_registry_df['Case ID'].apply(lambda x: x[-4:])

    stroke_registry_df['case_admission_id'] = create_registry_case_identification_column(stroke_registry_df)
    restricted_stroke_registry_df = restrict_to_patient_selection(stroke_registry_df, patient_selection_path, restrict_to_event_period=False,
                                                                  verbose=verbose)

    # if death in hospital, set mRs to 6
    restricted_stroke_registry_df.loc[restricted_stroke_registry_df['Death in hospital'] == 'yes', '3M mRS'] = 6
    # if 3M Death and 3M mRS nan, set mrs to 6
    restricted_stroke_registry_df.loc[(restricted_stroke_registry_df['3M Death'] == 'yes') &
                                        (restricted_stroke_registry_df['3M mRS'].isna()), '3M mRS'] = 6

    restricted_stroke_registry_df['3M delta mRS'] = restricted_stroke_registry_df['3M mRS'] - restricted_stroke_registry_df[
        'Prestroke disability (Rankin)']

    # if death in hospital set 3M Death to yes
    restricted_stroke_registry_df.loc[restricted_stroke_registry_df['Death in hospital'] == 'yes', '3M Death'] = 'yes'
    # if 3M mRs == 6, set 3M Death to yes
    restricted_stroke_registry_df.loc[restricted_stroke_registry_df['3M mRS'] == 6, '3M Death'] = 'yes'
    # if 3M mRs not nan and not 6, set 3M Death to no
    restricted_stroke_registry_df.loc[(restricted_stroke_registry_df['3M mRS'] != 6) &
                                      (~restricted_stroke_registry_df['3M mRS'].isna())
                                    &(restricted_stroke_registry_df['3M Death'].isna()), '3M Death'] = 'no'


    outcome_df = restricted_stroke_registry_df[["case_admission_id"] + outcome_columns]

    # restrict to plausible ranges
    outcome_df.loc[outcome_df['3M delta mRS'] < 0, '3M delta mRS'] = 0
    outcome_df.loc[outcome_df['Duration of hospital stay (days)'] > 365, 'Duration of hospital stay (days)'] = np.nan

    # add binarised outcomes if 3M mRS is not nan
    outcome_df['3M mRS 0-1'] = np.where(outcome_df['3M mRS'].isna(), np.nan, np.where(outcome_df['3M mRS'] <= 1, 1, 0))
    outcome_df['3M mRS 0-2'] = np.where(outcome_df['3M mRS'].isna(), np.nan, np.where(outcome_df['3M mRS'] <= 2, 1, 0))

    # binarise 3M Death and Death in hospital
    outcome_df.loc[outcome_df['3M Death'] == 'yes', '3M Death'] = 1
    outcome_df.loc[outcome_df['3M Death'] == 'no', '3M Death'] = 0
    outcome_df.loc[outcome_df['Death in hospital'] == 'yes', 'Death in hospital'] = 1
    outcome_df.loc[outcome_df['Death in hospital'] == 'no', 'Death in hospital'] = 0


    assert outcome_df['3M mRS 0-2'].value_counts().sum() == outcome_df['3M mRS'].value_counts().sum(), "Number of 3M mRS 0-2 not equal to 3M mRS"
    assert outcome_df['3M mRS 0-1'].value_counts().sum() == outcome_df['3M mRS'].value_counts().sum(), "Number of 3M mRS 0-1 not equal to 3M mRS"

    return outcome_df
