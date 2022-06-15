
# Data import

Read in data from a CSV file as string to avoid loosing leading zeros:
`pd.read_csv(os.path.join(data_path, f), delimiter=';', encoding='utf-8', dtype=str)`

Binarisation of continuous outcome containing NaN values: 
- This: `outcomes_df['cont_outcome'] = outcomes_df['cont_outcome'] <= 2`will result in NaN values to be considered as false.
- Alternative: ``outcome_df['cont_outcome'] = np.where(outcome_df['cont_outcome'].isna(), np.nan, np.where(outcome_df['cont_outcome'] <= 2, 1, 0))``
