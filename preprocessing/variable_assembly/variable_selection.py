import pandas as pd


def restrict_to_selected_variables(df:pd.DataFrame, variable_selection_path:str):
    """
    Restricts the dataframe to the selected variables.
    :param df: the dataframe to be restricted
    :param variable_selection_path: the path to the variable selection file
    :return: the dataframe restricted to the selected variables
    """
    selected_variables = pd.read_excel(variable_selection_path)['included']
    df = df.drop(df[~df.sample_label.isin(selected_variables)].index)

    return df