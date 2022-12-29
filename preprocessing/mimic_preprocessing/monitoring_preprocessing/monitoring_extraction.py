import pandas as pd
import psycopg2
import os
import argparse



def extract_monitoring(output_dir:str):
    """
    Extracts monitoring events from the MIMIC database and returns a dataframe
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
    WITH sel_monitoring as
    (
    """ + patient_selection_query + """
    SELECT selection.subject_id, selection.hadm_id, selection.icustay_id, selection.admittime,
            chart.itemid,	chart.charttime,	chart.storetime,	chart.value,	chart.valuenum,	chart.valueuom

    FROM selection

    INNER JOIN chartevents as chart
        on selection.hadm_id = chart.hadm_id

    WHERE selection.exclusion_discharge_diagnosis = 0
        AND selection.exclusion_first_stay = 0
        AND selection.exclusion_age = 0
        AND selection.exclusion_los = 0
        AND selection.exclusion_non_urgent = 0
        AND selection.exclusion_admission_diagnosis = 0
    )

    SELECT sel_monitoring.subject_id, sel_monitoring.hadm_id, sel_monitoring.icustay_id, sel_monitoring.admittime,
        sel_monitoring.itemid, d_items.label,	sel_monitoring.charttime,	sel_monitoring.storetime,	sel_monitoring.value,	sel_monitoring.valuenum,	sel_monitoring.valueuom
    FROM sel_monitoring
    LEFT JOIN d_items as d_items
        on sel_monitoring.itemid = d_items.itemid


    """

    return pd.read_sql_query(query, con)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', type=str)
    output_dir = parser.parse_args().output_dir

    monitoring_df = extract_monitoring(output_dir)
    monitoring_df.to_csv(os.path.join(output_dir, 'monitoring_df.csv'), index=False)
