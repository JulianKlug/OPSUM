import json
import os

import pandas as pd
import torch as ch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.outcome_prediction.Transformer.architecture import OPSUM_encoder_decoder
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitEncoderDecoderModel
from prediction.utils.utils import ensure_dir


def predict_n_next_steps(input_data, n_steps, model, trainer, use_gpu):
    """
    Predict the next n_steps using the model and trainer.
    :param input_data: The input data to predict on.
    :param n_steps: The number of steps to predict.
    :param model: The model to use for prediction.
    :param trainer: The trainer to use for prediction.
    :return: The predictions for the next n_steps.
    """
    y_placeholder = ch.zeros((input_data.shape[0], 1))

    predictions = []
    for i in tqdm(range(n_steps)):
        # first predictions only relies on past date
        if i == 0:
            input_np = input_data
        else:
            # append last prediction to input
            input_np = np.concatenate([input_np, np.expand_dims(predictions[-1], axis=1)], axis=1)

        if use_gpu:
            input_dataset = TensorDataset(ch.from_numpy(input_np).cuda(), y_placeholder.cuda())
        else:
            input_dataset = TensorDataset(ch.from_numpy(input_np), y_placeholder)

        input_loader = DataLoader(input_dataset, batch_size=1024)


        y_pred = np.array(trainer.predict(model, input_loader)[0][:, -1])

        # append prediction to list
        predictions.append(y_pred)

    predictions_np = np.concatenate([np.expand_dims(predictions[i], axis=1) for i in range(len(predictions))], axis=1)
    return predictions_np


def encoder_decoder_predict(data_path:str, model_path:str, model_config_path:str, split_idx:int=None,
            predict_n_time_steps:int=None, n_time_steps=None, use_gpu:bool=False):
    """
    Predict the next predict_n_time_steps using the encoder-decoder model.
    :param data_path: The path to the data.
    :param model_path: The path to the model.
    :param model_config_path: The path to the model config.
    :param predict_n_time_steps: The number of steps to predict.
    :param n_time_steps: total number of time steps in the data.
    :param use_gpu: Whether to use GPU for prediction.
    :return: The predictions for the next predict_n_time_steps.
    """
    
    if model_config_path.endswith('.json'):
        model_config = json.load(open(model_config_path))
    else:
        model_config = pd.read_csv(model_config_path)
        model_config = model_config.to_dict(orient='records')[0]

    if split_idx is None:
        split_idx = model_config['best_cv_fold']

    splits = ch.load(os.path.join(data_path))
    full_X_train, full_X_val, y_train, y_val = splits[split_idx]

    # prepare input data
    X_train = full_X_train[:, :, :, -1].astype('float32')
    X_val = full_X_val[:, :, :, -1].astype('float32')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)


    # prepare model
    accelerator = 'gpu' if use_gpu else 'cpu'

    ff_factor = 2
    ff_dim = ff_factor * model_config['model_dim']
    pos_encode_factor = 1

    input_dim = X_val.shape[-1]

    logger = DictLogger(0)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1000,
                        gradient_clip_val=model_config['grad_clip_value'], logger=logger)

    model_architecture = OPSUM_encoder_decoder(
                input_dim=input_dim,
                num_layers=int(model_config['num_layers']),
                num_decoder_layers=int(model_config['num_decoder_layers']),
                model_dim=int(model_config['model_dim']),
                dropout=int(model_config['dropout']),
                ff_dim=int(ff_dim),
                num_heads=int(model_config['num_head']),
                pos_encode_factor=pos_encode_factor,
                n_tokens=1,
                max_dim=5000,
                layer_norm_eps=1e-05)

    trained_model = LitEncoderDecoderModel.load_from_checkpoint(checkpoint_path=model_path, model=model_architecture,
                                                lr=model_config['lr'],
                                                wd=model_config['weight_decay'],
                                                train_noise=model_config['train_noise'],
                                                lr_warmup_steps=model_config['n_lr_warm_up_steps'],
                                                loss_function=model_config['loss_function'])
    
    # compute predictions
    pred_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        modified_time_steps = ts+1
        # predict recursively for predict_n_time_steps time steps
        X_val_with_first_n_ts = X_val[:, 0:modified_time_steps, :]

         # predict recursively for predict_n_time_steps time steps
        predictions_np = predict_n_next_steps(X_val_with_first_n_ts, predict_n_time_steps, trained_model, trainer, use_gpu=use_gpu)
        pred_over_ts.append(predictions_np)

    pred_over_ts_np = np.squeeze(pred_over_ts)

    return pred_over_ts_np


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-mc', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=False, default=None)
    parser.add_argument('-ts', '--predict_n_time_steps', type=int, required=False, default=6)
    parser.add_argument('-nts', '--n_time_steps', type=int, required=False, default=72)
    parser.add_argument('-g', '--use_gpu', action='store_true', required=False, default=False)

    args = parser.parse_args()

    pred_over_ts = encoder_decoder_predict(
        data_path=args.data_path,
        model_path=args.model_path,
        model_config_path=args.model_config_path,
        predict_n_time_steps=args.predict_n_time_steps,
        n_time_steps=args.n_time_steps,
        use_gpu=args.use_gpu
    )

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.model_path), 'predictions')
        ensure_dir(output_dir)
    else:
        output_dir = args.output_dir

    ch.save(pred_over_ts, os.path.join(output_dir, 'encoder_decoder_predictions.pt'))


if __name__ == '__main__':
    main()