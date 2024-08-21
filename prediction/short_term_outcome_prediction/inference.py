import os
import torch as ch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pytorch_lightning as pl
import pandas as pd
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.utils.utils import ensure_dir


def predict(data_path:str, model_path:str, model_config_path:str,
            imbalance_factor:float=62, n_time_steps=None, use_gpu:bool=False):
    splits = ch.load(os.path.join(data_path))
    model_config = pd.read_csv(model_config_path).to_dict(orient='records')[0]

    full_X_train, full_X_val, y_train, y_val = splits[model_config['best_cv_fold']]

    # prepare input data
    X_train = full_X_train[:, :, :, -1].astype('float32')
    X_val = full_X_val[:, :, :, -1].astype('float32')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)

    accelerator = 'gpu' if use_gpu else 'cpu'

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

    trained_model = LitModel.load_from_checkpoint(checkpoint_path=model_path, model=model_architecture,
                                                  lr=model_config['lr'],
                                                  wd=model_config['weight_decay'],
                                                  train_noise=model_config['train_noise'],
                                                  imbalance_factor=ch.tensor(imbalance_factor))

    if n_time_steps is None:
        n_time_steps = X_val.shape[1]

    pred_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        modified_time_steps = ts + 1

        X_val_with_first_n_ts = X_val[:, 0:modified_time_steps, :]
        y_placeholder = ch.zeros((X_val_with_first_n_ts.shape[0], 1))
        if use_gpu:
            val_dataset = TensorDataset(ch.from_numpy(X_val_with_first_n_ts).cuda(), y_placeholder.cuda())
        else:
            val_dataset = TensorDataset(ch.from_numpy(X_val_with_first_n_ts), y_placeholder)

        val_loader = DataLoader(val_dataset, batch_size=1024)
        if ts == 0:
            y_pred = np.array(ch.sigmoid(trainer.predict(trained_model, val_loader)[0]))
        else:
            y_pred = np.array(ch.sigmoid(trainer.predict(trained_model, val_loader)[0])[:, -1])

        pred_over_ts.append(np.squeeze(y_pred))

    return pred_over_ts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-mc', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=False, default=None)
    parser.add_argument('-ts', '--n_time_steps', type=int, required=False, default=None)
    parser.add_argument('-g', '--use_gpu', action='store_true', required=False, default=False)

    args = parser.parse_args()

    pred_over_ts = predict(data_path=args.data_path, model_path=args.model_path, model_config_path=args.model_config_path,
                            n_time_steps=args.n_time_steps, use_gpu=args.use_gpu)

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.model_path), 'predictions')
        ensure_dir(output_dir)
    else:
        output_dir = args.output_dir

    ch.save(pred_over_ts, os.path.join(output_dir, 'predictions.pt'))


if __name__ == '__main__':
    main()