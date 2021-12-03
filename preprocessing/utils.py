import pandas as pd
import numpy as np

def remove_french_accents_and_cedillas_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes(include=[np.object]).columns
    df[cols] = df[cols].apply(
        lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

    return df
