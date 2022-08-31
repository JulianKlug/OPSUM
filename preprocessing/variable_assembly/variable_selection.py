import pandas as pd


def restrict_to_selected_variables(df:pd.DataFrame, variable_selection_path:str, enforce:bool = False):
    """
    Restricts the dataframe to the selected variables.
    :param df: the dataframe to be restricted
    :param variable_selection_path: the path to the variable selection file
    :param enforce: if True, all variables from the variable selection file must be present in the dataframe
    :return: the dataframe restricted to the selected variables
    """
    selected_variables = pd.read_excel(variable_selection_path)['included']
    df = df.drop(df[~df.sample_label.isin(selected_variables)].index)

    if enforce:
        missing_variables = set(selected_variables).difference(set(df.sample_label.unique()))
        if len(missing_variables) > 0:
            raise ValueError(f'The following variables are missing from the dataframe: {missing_variables}')

    return df