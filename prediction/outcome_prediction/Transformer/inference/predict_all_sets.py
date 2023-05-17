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


def predict_all_sets(train_data, validation_data, test_data, model_weights_path:str, model_config:dict, use_gpu=False):
    """
    Predict probability for train, validation and test set.
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
    X_validation, y_validation = validation_data
    X_test, y_test = test_data

    train_dataset, test_dataset = prepare_dataset((X_train, X_test, y_train, y_test),
                                                  balanced=model_config['balanced'],
                                                  rescale=True,
                                                  use_gpu=use_gpu)
    _, validation_dataset = prepare_dataset((X_train, X_validation, y_train, y_validation),
                                            balanced=model_config['balanced'],
                                            rescale=True,
                                            use_gpu=use_gpu)

    train_loader = DataLoader(train_dataset, batch_size=X_train.shape[0])
    validation_loader = DataLoader(validation_dataset, batch_size=X_validation.shape[0])
    test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0])

    raw_predictions_train = trainer.predict(trained_model, train_loader)[0][:, -1]
    sigm_predictions_train = ch.sigmoid(raw_predictions_train)

    raw_predictions_validation = trainer.predict(trained_model, validation_loader)[0][:, -1]
    sigm_predictions_validation = ch.sigmoid(raw_predictions_validation)

    raw_predictions_test = trainer.predict(trained_model, test_loader)[0][:, -1]
    sigm_predictions_test = ch.sigmoid(raw_predictions_test)

    return (raw_predictions_test, sigm_predictions_test, y_test), (raw_predictions_validation, sigm_predictions_validation, y_validation), (raw_predictions_train, sigm_predictions_train, y_train)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer model for predicting outcome')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_path', type=str, required=True)
    parser.add_argument('-l', '--labels_path', type=str, required=True)
    parser.add_argument('-w', '--model_weights_path', type=str, required=True)
    parser.add_argument('-c', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--outcome', type=str, required=True)
    parser.add_argument('-O', '--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()


    pids, train_data, test_data, train_splits, test_features_lookup_table = load_data(args.features_path, args.labels_path, args.outcome, args.test_size, args.n_splits, args.seed)

    # load model config
    model_config = json.load(open(args.model_config_path, 'r'))

    fold_X_train, fold_X_val, fold_y_train, fold_y_val = train_splits[int(model_config['best_cv_fold'])]

    test_predictions, val_predictions, train_predictions = predict_all_sets((fold_X_train, fold_y_train), (fold_X_val, fold_y_val), test_data,
                                   args.model_weights_path, model_config, args.use_gpu)

    output_dir = args.output_dir
    # Save predictions as pickle
    with open(os.path.join(output_dir, 'test_predictions.pkl'), 'wb') as f:
        pickle.dump(test_predictions, f)

    with open(os.path.join(output_dir, 'val_predictions.pkl'), 'wb') as f:
        pickle.dump(val_predictions, f)

    with open(os.path.join(output_dir, 'train_predictions.pkl'), 'wb') as f:
        pickle.dump(train_predictions, f)


