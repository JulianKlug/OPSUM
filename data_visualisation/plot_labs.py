from matplotlib.dates import DateFormatter
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_patient_lab(patiend_admission_id, dosage_label, lab_df):
    temp = lab_df[(lab_df['patient_admission_id'] == patiend_admission_id) & (
        lab_df['dosage_label'].isin([dosage_label]))].copy()
    temp['value'] = pd.to_numeric(temp['value'], errors='coerce')
    temp['sample_date'] = pd.to_datetime(temp['sample_date'], format='%d.%m.%Y %H:%M')
    temp = temp.dropna(subset=['value'])
    ax = sns.scatterplot(x='sample_date', y='value', data=temp, hue='value', legend=False)
    # Define the date format
    date_form = DateFormatter("%d-%m-%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.tick_params(axis="x", rotation=45)

    return ax
