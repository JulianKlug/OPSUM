import json
import os
import pickle

import pandas as pd
from tqdm import tqdm

from prediction.outcome_prediction.Transformer.testing.test_model_from_trained_folds import \
    evaluate_model_from_trained_folds
from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.utils.utils import save_json, ensure_dir
from preprocessing.preprocessing_tools.normalisation.reverse_normalisation import reverse_normalisation


def test_model_on_subgroups(features_path, labels_path, normalisation_parameters_path, model_weights_dir, model_config_path, outcome, output_dir,
                            covid_subgroup_path = None, imaging_subgroup_path = None,
                            seed=42, test_size=0.2, n_splits=5, use_gpu=False, verbose: bool = False):
    """
    Test models from trained folds on test data split by subgroups.
    Subgroups:
        - First hour max NIHSS: <= 5 or over
        - Presenting mrs: 0-2 or 3-5
        - Age: <= 70 or over
        - Sex: m/f
        - Acute treatment: no treatment, only IVT, IAT +/- IVT

    Parameters:
        features_path: str
            Path to features file.
        labels_path: str
            Path to labels file.
        model_weights_dir: str
            Path to directory containing model weights for each trained fold
        model_config_path: str
            Path to model config file.
        outcome: str
            Outcome to predict.
        output_dir: str
            Path to output directory.
        covid_subgroup_path: str
            Path to csv file containing covid subgroup case / patient ids.
        imaging_subgroup_path: str
            Path to csv file containing patient ids of patients with imaging (from restricted to imaging extraction)
        seed: int
            Random seed.
        test_size: float
            Fraction of data to use for testing.
        n_splits: int
            Number of folds to use for cross-validation.
        use_gpu: bool
    """

    # Save testing args
    testing_args_df = pd.DataFrame(locals(), index=[0])
    testing_args_df['testing_mode'] = 'test_model_from_all_trained_folds'
    testing_args_df.to_csv(os.path.join(output_dir, 'testing_args.csv'), sep=',', index=False)

    # load data
    pids, train_data, test_data, train_splits, test_features_lookup_table = load_data(features_path, labels_path,
                                                                                      outcome, test_size, n_splits,
                                                                                      seed)

    pid_train, pid_test = pids
    train_X_np, train_y_np = train_data
    test_X_np, test_y_np = test_data

    # save patient ids used for testing / training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.tsv'),
        sep='\t', index=False)
    pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_test.tsv'),
        sep='\t', index=False)

    save_json(test_features_lookup_table,
              os.path.join(output_dir, 'test_lookup_dict.json'))

    # load normalisation parameters
    normalisation_parameters_df = pd.read_csv(normalisation_parameters_path)

    # Create a table with the features at timestep 0
    baseline_t0_test_X_np = test_X_np[:, 0, :]
    baseline_t0_test_X_df = pd.DataFrame(baseline_t0_test_X_np, columns=test_features_lookup_table['sample_label'])
    baseline_t0_test_X_df = baseline_t0_test_X_df.reset_index().rename(columns={'index': 'pidx'}).melt(id_vars='pidx',
                                                                                                       var_name='sample_label',
                                                                                                       value_name='value')
    non_norm_baseline_t0_test_X_df = reverse_normalisation(baseline_t0_test_X_df, normalisation_parameters_df)

    # load model config
    model_config = json.load(open(model_config_path, 'r'))

    # Define subgroups
    # every subgroup will be associated to the indices of the respective patients in the test set
    defined_subgroups = {
        'NIHSS': [],
        'mrs': [],
        'age': [],
        'sex': [],
        'treatment': []
    }

    # patient indices (!= patient_id)
    all_pidx = set(non_norm_baseline_t0_test_X_df.pidx.unique())

    # Define COVID subgroup
    if covid_subgroup_path is not None:
        covid_subgroup_df = pd.read_csv(covid_subgroup_path, dtype=str)
        # indices of covid patients in test set
        covid_pos_pidx = [test_features_lookup_table['case_admission_id'][cid] for cid in covid_subgroup_df.case_admission_id.unique()
                        if cid in test_features_lookup_table['case_admission_id'].keys()]
        covid_neg_pidx = all_pidx - set(covid_pos_pidx)
        defined_subgroups['covid'] = [('covid_positive', list(covid_pos_pidx)), ('covid_negative', list(covid_neg_pidx))]

    # Define with imaging subgroup
    if imaging_subgroup_path is not None:
        imaging_subgroup_df = pd.read_csv(imaging_subgroup_path, dtype=str)
        # indices of patients with imaging data available in test set
        with_imaging_pidx = [test_features_lookup_table['case_admission_id'][cid] for cid in imaging_subgroup_df.case_admission_id.unique()
                        if cid in test_features_lookup_table['case_admission_id'].keys()]
        without_imaging_pidx = all_pidx - set(with_imaging_pidx)
        defined_subgroups['with_imaging'] = [('with_imaging_available', list(with_imaging_pidx)),
                                             ('without_imaging_available', list(without_imaging_pidx))]

    # Define NIHSS subgroups
    minor_stroke_pidx = set(non_norm_baseline_t0_test_X_df[(non_norm_baseline_t0_test_X_df.sample_label == 'max_NIHSS') & (non_norm_baseline_t0_test_X_df.value <= 5)].pidx.unique())
    severe_stroke_pidx = all_pidx - minor_stroke_pidx
    defined_subgroups['NIHSS'].append(('NIHSS <= 5', list(minor_stroke_pidx)))
    defined_subgroups['NIHSS'].append(('NIHSS > 5', list(severe_stroke_pidx)))

    # Define mrs subgroups
    pidx_mrs3_to_5 = set(list(non_norm_baseline_t0_test_X_df[(
                                                                     non_norm_baseline_t0_test_X_df.sample_label == 'prestroke_disability_(rankin)_3.0') & (
                                                                     non_norm_baseline_t0_test_X_df.value == 1)].pidx.unique())
                         + list(non_norm_baseline_t0_test_X_df[
                                                                    (non_norm_baseline_t0_test_X_df.sample_label == 'prestroke_disability_(rankin)_4.0') & (
                                                                     non_norm_baseline_t0_test_X_df.value == 1)].pidx.unique())
                         + list(non_norm_baseline_t0_test_X_df[
                                                                    (non_norm_baseline_t0_test_X_df.sample_label == 'prestroke_disability_(rankin)_5.0') & (
                                                                            non_norm_baseline_t0_test_X_df.value == 1)].pidx.unique()))
    pidx_mrs0_to_2 = all_pidx - pidx_mrs3_to_5
    defined_subgroups['mrs'].append(('mRS 0-2', list(pidx_mrs0_to_2)))
    defined_subgroups['mrs'].append(('mRS 3-5', list(pidx_mrs3_to_5)))

    # Define age subgroups
    pidx_age_under_70 = set(non_norm_baseline_t0_test_X_df[(non_norm_baseline_t0_test_X_df.sample_label == 'age') & (
                non_norm_baseline_t0_test_X_df.value <= 70)].pidx.unique())
    pidx_age_over_70 = all_pidx - pidx_age_under_70
    defined_subgroups['age'].append(('age <= 70', list(pidx_age_under_70)))
    defined_subgroups['age'].append(('age > 70', list(pidx_age_over_70)))

    # Define sex subgroups
    pidx_sex_male = set(non_norm_baseline_t0_test_X_df[(non_norm_baseline_t0_test_X_df.sample_label == 'sex_male') & (
                non_norm_baseline_t0_test_X_df.value == 1)].pidx.unique())
    pidx_sex_female = all_pidx - pidx_sex_male
    defined_subgroups['sex'].append(('male', list(pidx_sex_male)))
    defined_subgroups['sex'].append(('female', list(pidx_sex_female)))

    # Define treatment subgroups
    pidx_with_IAT = set(non_norm_baseline_t0_test_X_df[
                            (non_norm_baseline_t0_test_X_df.sample_label == 'categorical_iat_no_iat') & (
                                        non_norm_baseline_t0_test_X_df.value == 0)].pidx.unique())
    pidx_with_IVT = set(non_norm_baseline_t0_test_X_df[
                            (non_norm_baseline_t0_test_X_df.sample_label == 'categorical_ivt_no_ivt') & (
                                        non_norm_baseline_t0_test_X_df.value == 0)].pidx.unique())
    pidx_with_only_IVT = pidx_with_IVT - pidx_with_IAT
    pidx_with_no_ttt = all_pidx - pidx_with_IAT - pidx_with_IVT

    defined_subgroups['treatment'].append(('IAT (with_or_without IVT)', list(pidx_with_IAT)))
    defined_subgroups['treatment'].append(('IVT only', list(pidx_with_IVT)))
    defined_subgroups['treatment'].append(('no treatment', list(pidx_with_no_ttt)))

    ## Iterate through subgroups and evaluate performance
    for subgroup in tqdm(defined_subgroups.keys()):
        if verbose:
            print('Evaluating subgroup: {}'.format(subgroup))

        # create directory for subgroup output
        subgroup_output_dir = os.path.join(output_dir, subgroup)
        ensure_dir(subgroup_output_dir)

        subgroup_results_df = pd.DataFrame()
        # iterate through subgroup splits
        for subgroup_split_name, subgroup_pidx in defined_subgroups[subgroup]:
            if verbose:
                print('Evaluating subgroup split: {}'.format(subgroup_split_name))

            # create directory for subgroup split output
            subgroup_split_output_dir = os.path.join(subgroup_output_dir, subgroup_split_name.replace(' ', '_').replace('<=', 'under_equal').replace('>', 'over'))
            ensure_dir(subgroup_split_output_dir)

            subgroup_split_X_np = test_X_np[subgroup_pidx, :, :]
            subgroup_split_y_np = test_y_np[subgroup_pidx]

            subgroup_split_results_df = evaluate_model_from_trained_folds(subgroup_split_X_np, subgroup_split_y_np, train_splits,
                                              model_config, model_weights_dir, outcome, subgroup_split_output_dir, use_gpu=use_gpu)
            subgroup_split_results_df['subgroup_split'] = subgroup_split_name
            subgroup_results_df = subgroup_results_df.append(subgroup_split_results_df)
        subgroup_results_df['subgroup'] = subgroup
        subgroup_results_df.to_csv(os.path.join(subgroup_output_dir, 'subgroup_results.csv'), index=False)


if __name__ == '__main__':
    # Example usage
    # -f /Users/jk1/.../preprocessed_features_01012023_233050.csv -l /Users/jk1/../preprocessed_outcomes_01012023_233050.csv -n /Users/jk1/../normalisation_parameters.csv
    # -w /Users/jk1/../trained_models -c /Users/jk1/../hyperopt_selected_transformer_20230402_184459.json
    # -o "3M mRS 0-2" -O /Users/jk1/../subgroup_testing_output
    # -i /Users/jk1/../imaging_subset_cids.csv -cov /Users/jk1/../opsum_covid_subset.csv

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_path', type=str, required=True)
    parser.add_argument('-l', '--labels_path', type=str, required=True)
    parser.add_argument('-n', '--normalisation_parameters_path', type=str, required=True)
    parser.add_argument('-w', '--model_weights_dir', type=str, required=True)
    parser.add_argument('-c', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--outcome', type=str, required=True)
    parser.add_argument('-O', '--output_dir', type=str, required=True)
    parser.add_argument('-i', '--imaging_subgroup_path', type=str, required=False, default=None)
    parser.add_argument('-cov', '--covid_subgroup_path', type=str, required=False, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    test_model_on_subgroups(**vars(args))




