import pandas as pd
import psycopg2
import os
import argparse



def extract_procedures():
    """
    Extracts procedures events from the MIMIC database and returns a dataframe
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
    WITH sel_proceduress as
    (
    """ + patient_selection_query + """
    SELECT selection.subject_id, selection.hadm_id, selection.icustay_id, selection.admittime,
            procedures.icd9_code, in_cv.itemid as citemid, in_cv.charttime as charttime, in_mv.itemid as mitemid, in_mv.starttime as starttime,
            proc_mv.itemid as pmitemid, proc_mv.starttime as proc_starttime
    
    FROM selection
    
    LEFT JOIN procedures_icd as procedures
      ON selection.hadm_id = procedures.hadm_id
      AND (procedures.icd9_code::integer = 3974 OR procedures.icd9_code::integer = 9910)
    
    LEFT JOIN inputevents_cv as in_cv
        ON selection.hadm_id = in_cv.hadm_id
        AND (in_cv.itemid = 227056 OR in_cv.itemid = 2666 OR in_cv.itemid = 2667 OR in_cv.itemid = 42575 OR in_cv.itemid = 221319)
    
    LEFT JOIN inputevents_mv as in_mv
        ON selection.hadm_id = in_mv.hadm_id
        AND (in_mv.itemid = 227056 OR in_mv.itemid = 2666 OR in_mv.itemid = 2667 OR in_mv.itemid = 42575 OR in_mv.itemid = 221319)
    
    LEFT JOIN procedureevents_mv as proc_mv
        ON selection.hadm_id = proc_mv.hadm_id
        AND (proc_mv.itemid = 225427 OR proc_mv.itemid = 225462)

    WHERE selection.exclusion_discharge_diagnosis = 0
        AND selection.exclusion_first_stay = 0
        AND selection.exclusion_age = 0
        AND selection.exclusion_los = 0
        AND selection.exclusion_non_urgent = 0
        AND selection.exclusion_procedures_diagnosis = 0
    )

    SELECT sel_procedures.subject_id, sel_procedures.hadm_id, sel_procedures.icustay_id, sel_procedures.admittime,
        sel_procedures.icd9_code, sel_procedures.charttime, sel_procedures.starttime, proc_starttime, d_icd_procedures.long_title, d_items.label, pmitemid, mitemid, citemid
    
    FROM sel_procedures
    
    LEFT JOIN d_icd_procedures
        on sel_procedures.icd9_code = d_icd_procedures.icd9_code
    
    LEFT JOIN d_items as d_items
        ON sel_procedures.citemid = d_items.itemid
        OR sel_procedures.mitemid = d_items.itemid
        OR sel_procedures.pmitemid = d_items.itemid
    """

    return pd.read_sql_query(query, con)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', type=str)
    output_dir = parser.parse_args().output_dir

    procedures_df = extract_procedures()
    procedures_df.to_csv(os.path.join(output_dir, 'procedures_df.csv'), index=False)
