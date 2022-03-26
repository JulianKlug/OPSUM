import pandas as pd
import numpy as np
import os

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


def preprocess_outcomes(outcome_df, patient_selection_df):
    outcome_df['patient_id'] = outcome_df['Case ID'].apply(lambda x: x[8:-4])
    outcome_df['EDS_last_4_digits'] = outcome_df['Case ID'].apply(lambda x: x[-4:])

    patient_selection_df['case_id'] = patient_selection_df['patient_id'].astype(str) + patient_selection_df[
        'EDS_last_4_digits'].astype(str)

    # TODO use restrict to patient slection function
    selected_full_data_df = outcome_df[
        outcome_df['Case ID'].apply(lambda x: x[8:]).isin(patient_selection_df['case_id'].tolist())]

    selected_full_data_df['begin_date'] = pd.to_datetime(selected_full_data_df['Arrival at hospital'],
                                                         format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                          selected_full_data_df['Arrival time']

    selected_full_data_df['patient_admission_id'] = selected_full_data_df['patient_id'].astype(str) + \
                                                    selected_full_data_df['EDS_last_4_digits'].astype(str) + '_' + \
                                                    selected_full_data_df['begin_date'].apply(
                                                        lambda bd: ''.join(bd.split(' ')[0].split('.')))

    selected_full_data_df['3M delta mRS'] = selected_full_data_df['3M mRS'] - selected_full_data_df[
        'Prestroke disability (Rankin)']

    outcome_df = selected_full_data_df[["patient_admission_id"] + outcome_columns]

    # restrict to plausible ranges
    outcome_df.loc[outcome_df['3M delta mRS'] < 0, '3M delta mRS'] = 0
    outcome_df.loc[outcome_df['Duration of hospital stay (days)'] > 365, 'Duration of hospital stay (days)'] = np.nan

    return outcome_df
