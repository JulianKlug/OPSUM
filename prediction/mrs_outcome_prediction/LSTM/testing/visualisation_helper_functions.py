


def reverse_normalisation_for_subj(norm_subj_df, normalisation_parameters_df):
    subj_df = norm_subj_df.copy()
    for variable in normalisation_parameters_df.variable.unique():
        if variable not in subj_df.columns:
            continue

        temp = subj_df[variable].copy()
        std = normalisation_parameters_df[normalisation_parameters_df.variable == variable].original_std.iloc[0]
        mean = normalisation_parameters_df[normalisation_parameters_df.variable == variable].original_mean.iloc[0]
        temp = (temp * std) + mean
        subj_df[variable] = temp

    return subj_df