import pandas as pd
from tqdm import tqdm


def reverse_normalisation(normalised_df: pd.DataFrame, normalisation_parameters_df: pd.DataFrame,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Reverse normalisation by multiplying with std and adding mean
    :param normalised_df: dataframe after normalisation
    :param normalisation_parameters_df: dataframe with mean and std for every normalised variable
    :param verbose:
    :return:
    """
    reversed_normalised_df = normalised_df.copy()
    if verbose:
        print(f'Reversing normalisation...')

    for variable in tqdm(normalisation_parameters_df.variable.unique()):
        if variable not in reversed_normalised_df.sample_label.unique():
            print(f'{variable} is not present in Dataframe')
            continue

        temp = reversed_normalised_df[
            reversed_normalised_df.sample_label == variable].value.copy()
        std = normalisation_parameters_df[normalisation_parameters_df.variable == variable].original_std.iloc[0]
        mean = normalisation_parameters_df[normalisation_parameters_df.variable == variable].original_mean.iloc[0]
        temp = (temp * std) + mean
        reversed_normalised_df.loc[
            reversed_normalised_df.sample_label == variable, 'value'] = temp

    return reversed_normalised_df


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--normalised_df_path', type=str, required=True)
    parser.add_argument('-p', '--normalisation_parameters_df_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    args = parser.parse_args()

    normalised_df = pd.read_csv(args.normalised_df_path)
    normalisation_parameters_df = pd.read_csv(args.normalisation_parameters_df_path)
    reversed_normalised_df = reverse_normalisation(normalised_df, normalisation_parameters_df)
    reversed_normalised_df.to_csv(os.path.join(args.output_path, 'reverse_normalised_df.csv'), index=False)

    