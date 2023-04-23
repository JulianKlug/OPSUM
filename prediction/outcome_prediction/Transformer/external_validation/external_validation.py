import json
import os
import pickle

import pandas as pd
from tqdm import tqdm

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.outcome_prediction.Transformer.testing.test_transformer_model import test_transformer_model
from prediction.outcome_prediction.data_loading.data_loader import load_data, load_external_data
from prediction.utils.utils import save_json, ensure_dir


def external_validation_from_trained_folds(training_features_path, training_labels_path, external_features_path, external_labels_path,  model_weights_dir, model_config_path, outcome, output_dir,
                                  seed=42, test_size=0.2, n_splits=5, use_gpu=False):
    """
    Test models from trained folds on test data.
    """
    # got dict of input args
    testing_args_df = pd.DataFrame(locals(), index=[0])
    testing_args_df['testing_mode'] = 'test_model_from_all_trained_folds'
    testing_args_df.to_csv(os.path.join(output_dir, 'testing_args.csv'), sep=',', index=False)

    # load training data
    pids, train_data, _, train_splits, _ = load_data(training_features_path, training_labels_path, outcome, test_size, n_splits, seed)

    # load external test data
    test_X_np, test_y_np, test_features_lookup_table = load_external_data(external_features_path, external_labels_path, outcome)

    pid_train, pid_test = pids
    train_X_np, train_y_np = train_data

    # save patient ids used for training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.tsv'),
        sep='\t', index=False)


    save_json(test_features_lookup_table,
              os.path.join(output_dir, 'test_lookup_dict.json'))

    # load model config
    model_config = json.load(open(model_config_path, 'r'))

    # define model
    ff_factor = 2
    ff_dim = ff_factor * model_config['model_dim']
    pos_encode_factor = 1

    model_architecture = OPSUMTransformer(
        input_dim=84,
        num_layers=int(model_config['num_layers']),
        model_dim=int(model_config['model_dim']),
        dropout=int(model_config['dropout']),
        ff_dim=int(ff_dim),
        num_heads=int(model_config['num_head']),
        num_classes=1,
        max_dim=500,
        pos_encode_factor=pos_encode_factor
    )

    trained_model_base_name = '_'.join([dir for dir in os.listdir(model_weights_dir)
                         if (os.path.isdir(os.path.join(model_weights_dir, dir))) & (dir.startswith('checkpoints'))
                         ][0].split('_')[:-1])

    overall_results_df = pd.DataFrame()
    for split_idx in tqdm(range(len(train_splits))):
        fold_X_train, _, fold_y_train, _ = train_splits[split_idx]

        # load model corresponding to fold
        fold_model_dir = os.path.join(model_weights_dir, f'{trained_model_base_name}_{split_idx}')
        if not os.path.exists(fold_model_dir):
            raise ValueError(f'No model found for fold {split_idx}.')
        fold_trained_model_paths = os.listdir(fold_model_dir)
        if len(fold_trained_model_paths) > 1:
            raise ValueError(f'More than one model found in {fold_model_dir}.')
        fold_trained_model_path = os.path.join(fold_model_dir, fold_trained_model_paths[0])

        trained_model = LitModel.load_from_checkpoint(checkpoint_path=fold_trained_model_path, model=model_architecture,
                                                      lr=model_config['lr'],
                                                      wd=model_config['weight_decay'],
                                                      train_noise=model_config['train_noise'])

        # (bootstrapped_ground_truth, bootstrapped_predictions), (y, y_pred_test)
        fold_result_df, bootstrapped_gt_and_pred, overall_gt_and_pred = test_transformer_model(trained_model, fold_X_train,
                                                                                             fold_y_train, test_X_np,
                                                                                             test_y_np, outcome,
                                                                                             model_config, fold_trained_model_path,
                                                                                             use_gpu=use_gpu)
        fold_result_df['fold'] = split_idx
        overall_results_df = pd.concat([overall_results_df, fold_result_df])

        # save bootstrapped ground truth and predictions
        pickle.dump(bootstrapped_gt_and_pred, open(os.path.join(output_dir, f'fold_{split_idx}_bootstrapped_gt_and_pred.pkl'), 'wb'))
        pickle.dump(overall_gt_and_pred, open(os.path.join(output_dir, f'fold_{split_idx}_test_gt_and_pred.pkl'), 'wb'))

    overall_results_df.to_csv(os.path.join(output_dir, 'overall_results.csv'), sep=',', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', '--training_features_path', type=str, required=True)
    parser.add_argument('-ef', '--external_features_path', type=str, required=True)
    parser.add_argument('-tl', '--training_labels_path', type=str, required=True)
    parser.add_argument('-el','--external_labels_path', type=str, required=True)
    parser.add_argument('-w', '--model_weights_dir', type=str, required=True)
    parser.add_argument('-c', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--outcome', type=str, required=True)
    parser.add_argument('-O', '--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('-g','--use_gpu', type=bool, default=False)

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    external_validation_from_trained_folds(**vars(args))





