import pandas as pd
import numpy as np

def remove_french_accents_and_cedillas_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes(include=[np.object]).columns
    df[cols] = df[cols].apply(
        lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

    return df


def create_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits) as well as admission date
    case_identification_column = df['patient_id'].astype(str) \
                                 + df['eds_end_4digit'].astype(str) \
                                 + '_' + df['begin_date'].apply(lambda bd: ''.join(bd.split(' ')[0].split('.')))
    return case_identification_column
