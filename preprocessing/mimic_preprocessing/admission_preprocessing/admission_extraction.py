import pandas as pd
import psycopg2
import os
import argparse



def extract_admission():
    """
    Extracts admission events from the MIMIC database and returns a dataframe
    Applies selection criteria
    """
    # information used to create a database connection
    sqluser = 'postgres'
    sqlpassword = 'postgres'
    dbname = 'mimic'
    schema_name = 'mimiciii'
    con = psycopg2.connect(dbname=dbname, user=sqluser, password=sqlpassword, port=5000, host='localhost')

    # the below statement is prepended to queries to ensure they select from the right schema
    query_schema = f'set search_path to {schema_name};'

    selection_query_path = '../data_extraction/patient_selection_query.sql'
    # load in the text of the query
    with open(selection_query_path) as fp:
        patient_selection_query = ''.join(fp.readlines())

    query = query_schema + """
    WITH sel_admission as
    (
    """ + patient_selection_query + """
    SELECT selection.subject_id, selection.hadm_id, selection.icustay_id, selection.dob, selection.admittime,
           selection.age, pat.gender, admissions.diagnosis, admissions.admission_type, admissions.admission_location, chart.itemid, chart.value as chart_value

    FROM selection
    
    INNER JOIN patients pat
      ON selection.subject_id = pat.subject_id
    
    INNER JOIN admissions admissions
      ON selection.hadm_id = admissions.hadm_id
    
    LEFT JOIN chartevents as chart
        ON selection.hadm_id = chart.hadm_id
        AND (chart.itemid = 225059 OR chart.itemid = 225811)

    WHERE selection.exclusion_discharge_diagnosis = 0
        AND selection.exclusion_first_stay = 0
        AND selection.exclusion_age = 0
        AND selection.exclusion_los = 0
        AND selection.exclusion_non_urgent = 0
        AND selection.exclusion_admission_diagnosis = 0
    )

    SELECT sel_admission.subject_id, sel_admission.hadm_id, sel_admission.icustay_id, sel_admission.dob, sel_admission.admittime,
            sel_admission.age, sel_admission.gender, sel_admission.admission_type, sel_admission.diagnosis, sel_admission.admission_location,
            sel_admission.itemid, d_items.label, sel_admission.chart_value
        
    FROM sel_admission
    LEFT JOIN d_items as d_items
        on sel_admission.itemid = d_items.itemid


    """

    return pd.read_sql_query(query, con)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', type=str)
    output_dir = parser.parse_args().output_dir

    admission_df = extract_admission()
    admission_df.to_csv(os.path.join(output_dir, 'admission_df.csv'), index=False)
