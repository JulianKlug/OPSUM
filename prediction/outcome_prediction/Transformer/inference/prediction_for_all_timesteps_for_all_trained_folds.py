import argparse
import json
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch as ch
import pytorch_lightning as pl

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.outcome_prediction.Transformer.utils.utils import prepare_dataset, DictLogger


def prediction_for_all_timesteps(train_data, test_data, model_weights_path:str, n_time_steps:int, model_config:dict, use_gpu=False):
    """
    Predicts the outcome for all timesteps for all patients in data.
    Args:
        data: torch dataset of shape (n_patients, n_time_steps, n_features)
        model_weights_path: path to the model weights
        n_time_steps: total number of time steps
        n_channels: number of channels
        config: model configuration

    Returns:
        predictions: numpy array of shape (n_time_steps, n_patients)
    """
    if use_gpu:
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'
    logger = DictLogger(0)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1000,
                         gradient_clip_val=model_config['grad_clip_value'], logger=logger)



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

    pred_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        modified_time_steps = ts + 1

        X_test_with_first_n_ts = X_test[:, 0:modified_time_steps, :]
        train_dataset, test_dataset = prepare_dataset((X_train, X_test_with_first_n_ts, y_train, y_test),
                                                      balanced=model_config['balanced'],
                                                      rescale=True,
                                                      use_gpu=use_gpu)

        test_loader = DataLoader(test_dataset, batch_size=1024)
        if ts == 0:
            y_pred = ch.sigmoid(trainer.predict(trained_model, test_loader)[0])
        else:
            y_pred = ch.sigmoid(trainer.predict(trained_model, test_loader)[0])[:, -1]


        pred_over_ts.append(np.squeeze(y_pred))

    return np.array(pred_over_ts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer model for predicting outcome')
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--model_weights_dir', type=str, required=True)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--outcome', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_time_steps', type=int, default=72)
    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()

    pids, train_data, test_data, train_splits, test_features_lookup_table = load_data(args.features_path, args.labels_path, args.outcome, args.test_size, args.n_splits, args.seed)
    # load model config
    model_config = json.load(open(args.model_config_path, 'r'))

    for fold_idx in range(args.n_splits):
        fold_X_train, _, fold_y_train, _ = train_splits[int(fold_idx)]

        fold_model_weights_dir = os.path.join(args.model_weights_dir, f'checkpoints_opsum_{os.basename(args.model_weights_dir)}_cv_{fold_idx}')
        model_weights_paths = [os.path.join(fold_model_weights_dir, f) for f in os.listdir(fold_model_weights_dir) if f.endswith('.ckpt')]
        if len(model_weights_paths) == 1:
            model_weights_path = model_weights_paths[0]
        else:
            raise ValueError(f'Found {len(model_weights_paths)} model weights files in {fold_model_weights_dir}')

        output_dir = fold_model_weights_dir

        predictions = prediction_for_all_timesteps((fold_X_train, fold_y_train), test_data, model_weights_path, args.n_time_steps, model_config, args.use_gpu)

        # Save predictions as pickle
        with open(os.path.join(output_dir, f'predictions_over_timesteps_from_fold_{fold_idx}.pkl'), 'wb') as f:
            pickle.dump(predictions, f)


