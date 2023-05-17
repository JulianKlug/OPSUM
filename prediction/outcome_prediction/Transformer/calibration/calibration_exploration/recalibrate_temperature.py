import os
import torch
import json
from torch.utils.data import DataLoader

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.outcome_prediction.Transformer.utils.utils import prepare_dataset
from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.utils.calibration_tools import ModelWithTemperature


def recalibrate_temperature(model_config_path, model_weights_path, outcome, features_path, labels_path, out_dir,
                            test_size=0.2, n_splits=5, seed=42, use_gpu=False):
    """
    Applies temperature scaling to a trained model, then saves a temperature scaled version.
    Args:
        model_config_path (str): path to model config file
        model_weights_path (str): path to model weights file
        outcome (str): outcome to predict
        features_path (str): path to features file
        labels_path (str): path to labels file
        out_dir (str): path to output directory
        test_size (float): proportion of data to use for testing
        n_splits (int): number of cross validation folds
        seed (int): random seed
        use_gpu (bool): whether to use GPU
    Returns: Void
    """
    # Load model state dict
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

    trained_model = LitModel.load_from_checkpoint(checkpoint_path=model_weights_path, model=model_architecture,
                                                  lr=model_config['lr'],
                                                  wd=model_config['weight_decay'],
                                                  train_noise=model_config['train_noise'])


    # Load validation data
    pids, train_data, test_data, train_splits, test_features_lookup_table = load_data(features_path, labels_path,
                                                                                      outcome, test_size, n_splits,
                                                                                      seed)
    fold_X_train, fold_X_val, fold_y_train, fold_y_val = train_splits[int(model_config['best_cv_fold'])]

    train_dataset, val_dataset = prepare_dataset((fold_X_train, fold_X_val, fold_y_train, fold_y_val),
                                                 balanced=model_config['balanced'],
                                                 rescale=True,
                                                 use_gpu=use_gpu)

    val_loader = DataLoader(val_dataset, batch_size=1024)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(trained_model.model, use_gpu=use_gpu, verbose=True, target_class=-1)

    # Tune the model temperature, and save the results
    model.set_temperature(val_loader)
    model_filename = os.path.join(out_dir, 'model_with_temperature.pth')
    torch.save(model.state_dict(), model_filename)


if __name__ == '__main__':
    """
    Applies temperature scaling to a trained model.
    """
    model_config_path = '/Users/jk1/temp/opsum_prediction_output/transformer/3M_mrs02/transformer_20230402_184459_test_set_evaluation/hyperopt_selected_transformer_20230402_184459.json'
    model_weights_path = '/Users/jk1/temp/opsum_prediction_output/transformer/3M_mrs02/transformer_20230402_184459_test_set_evaluation/trained_models/checkpoints_opsum_transformer_20230402_184459_cv_2/opsum_transformer_epoch=14_val_auroc=0.9222.ckpt'
    outcome = '3M mRS 0-2'
    features_path = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/preprocessed_features_01012023_233050.csv'
    labels_path = '/Users/jk1/temp/opsum_prepro_output/gsu_prepro_01012023_233050/preprocessed_outcomes_01012023_233050.csv'
    out_dir = '/Users/jk1/temp/opsum_prediction_output/transformer/3M_mrs02/recalibration'

    recalibrate_temperature(model_config_path, model_weights_path, outcome, features_path, labels_path, out_dir)