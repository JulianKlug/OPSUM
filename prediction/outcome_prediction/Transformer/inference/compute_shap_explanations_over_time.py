import argparse
import json
import os
import time

import shap
import pickle
import numpy as np
from captum.attr import GradientShap
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch as ch
import pytorch_lightning as pl

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.outcome_prediction.Transformer.utils.utils import prepare_dataset, DictLogger
#from prediction.utils.shap_helper_functions import check_shap_version_compatibility

# Shap values require very specific versions
#check_shap_version_compatibility()


def compute_shap_explanations_over_time(train_data, test_data, model_weights_path:str, n_time_steps:int, model_config:dict,
                                        n_samples_background=100, library:str = 'shap', use_gpu=False):
    """
    Compute SHAP values for all timesteps for all patients in test data.
    Args:
        train_data: tuple of (X_train, y_train)
        test_data: tuple of (X_test, y_test)
        model_weights_path: path to the model weights
        n_time_steps: total number of time steps
        model_config: model configuration
        n_samples_background: number of samples to use as background for SHAP
        library: which library to use for SHAP computation (shap or captum)
        use_gpu: whether to use GPU

    Returns:
        predictions: numpy array of shape (n_time_steps, n_patients)
    """

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

    trained_model = LitModel.load_from_checkpoint(checkpoint_path=model_weights_path, model=model_architecture,
                                                  lr=model_config['lr'],
                                                  wd=model_config['weight_decay'],
                                                  train_noise=model_config['train_noise'])

    X_train, y_train = train_data
    X_test, y_test = test_data

    shap_values_over_ts = []
    for ts in tqdm(range(n_time_steps)):

        # Prepare datasets (use only first ts time steps)
        modified_time_steps = ts + 1
        X_train_with_first_n_ts = X_train[:, 0:modified_time_steps, :]
        X_test_with_first_n_ts = X_test[:, 0:modified_time_steps, :]
        print('Using GPU: ', use_gpu)
        train_dataset, test_dataset = prepare_dataset((X_train_with_first_n_ts, X_test_with_first_n_ts, y_train, y_test),
                                                      balanced=model_config['balanced'],
                                                      rescale=True,
                                                      use_gpu=use_gpu)

        # Prepare background dataset (use all training data in batch size)
        train_loader = DataLoader(train_dataset, batch_size=X_train.shape[0], shuffle=True, drop_last=True)

        batch = next(iter(train_loader))
        train_sample, _ = batch
        background = train_sample[:n_samples_background]

        test_loader = DataLoader(test_dataset, batch_size=1024)
        batch = next(iter(test_loader))
        test_samples, _ = batch

        if library == 'shap':
            # Initialize DeepExplainer
            explainer = shap.DeepExplainer(trained_model.model.to(background.device), background)

            shap_values = explainer.shap_values(test_samples)
            shap_values_over_ts.append(shap_values)
        elif library == 'captum':
            print(f'Using captum library for SHAP computation for time step {ts}, {test_samples.shape}, {background.shape}')

            gradient_shap = GradientShap(trained_model.model.to(background.device))
            print(f'Devices: {background.device}, {test_samples.device}')
           
            attributions_gs = gradient_shap.attribute(test_samples,
                                                      baselines=background,
                                                      target=(-1, 0))
            shap_values_over_ts.append(attributions_gs.cpu().numpy())
        else:
            raise ValueError(f'Library {library} not supported.')

    return shap_values_over_ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer model - compute shap explanations over time.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--model_weights_path', type=str, required=True)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--outcome', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_time_steps', type=int, default=72)
    parser.add_argument('--n_samples_background', type=int, default=100)
    parser.add_argument('--library', type=str, default='shap')
    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()


    pids, train_data, test_data, train_splits, test_features_lookup_table = load_data(args.features_path, args.labels_path, args.outcome, args.test_size, args.n_splits, args.seed)

    # load model config
    model_config = json.load(open(args.model_config_path, 'r'))

    fold_X_train, _, fold_y_train, _ = train_splits[int(model_config['best_cv_fold'])]

    # Time execution
    start_time = time.time()
    shap_values_over_ts = compute_shap_explanations_over_time((fold_X_train, fold_y_train), test_data, args.model_weights_path, args.n_time_steps, model_config,
                                                              n_samples_background=args.n_samples_background, library=args.library,
                                                              use_gpu=args.use_gpu)
    print('Time elapsed: {:.2f} min'.format((time.time() - start_time) / 60))

    with open(os.path.join(args.output_dir, 'transformer_explainer_shap_values_over_ts.pkl'), 'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)



