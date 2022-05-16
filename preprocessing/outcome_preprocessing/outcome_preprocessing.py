import pandas as pd
import numpy as np
import os

from preprocessing.patient_selection.restrict_to_patient_selection import restrict_to_patient_selection

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

    stroke_registry_df['case_admission_id'] = stroke_registry_df['patient_id'].astype(str) \
                                              + stroke_registry_df['EDS_last_4_digits'].astype(str) \
                                              + '_' + pd.to_datetime(stroke_registry_df['Arrival at hospital'],
                                                                     format='%Y%m%d').dt.strftime('%d%m%Y').astype(str)
    restricted_stroke_registry_df = restrict_to_patient_selection(stroke_registry_df, patient_selection_path,
                                                                  verbose=verbose)

    restricted_stroke_registry_df['3M delta mRS'] = restricted_stroke_registry_df['3M mRS'] - restricted_stroke_registry_df[
        'Prestroke disability (Rankin)']

    outcome_df = restricted_stroke_registry_df[["case_admission_id"] + outcome_columns]

    # restrict to plausible ranges
    outcome_df.loc[outcome_df['3M delta mRS'] < 0, '3M delta mRS'] = 0
    outcome_df.loc[outcome_df['Duration of hospital stay (days)'] > 365, 'Duration of hospital stay (days)'] = np.nan

    # add binarised outcomes
    outcome_df['3M mRS 0-1'] = outcome_df['3M mRS'] <= 1
    outcome_df['3M mRS 0-2'] = outcome_df['3M mRS'] <= 2

    return outcome_df
