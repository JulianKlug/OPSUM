
# Data import

Read in data from a CSV file as string to avoid loosing leading zeros:
`pd.read_csv(os.path.join(data_path, f), delimiter=';', encoding='utf-8', dtype=str)`