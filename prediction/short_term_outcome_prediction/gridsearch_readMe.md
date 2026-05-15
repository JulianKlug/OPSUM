# OPSUM - Short-term outcome: GridSearch Configuration Guide

This document provides an explanation of the parameters used in the OPSUM grid search configurations and includes example configuration files for different model types.

## Overview

OPSUM supports hyperparameter optimization for three types of transformer models:
- **Encoder-only**: Binary classification for outcome prediction
- **Encoder-Decoder**: Time-series prediction
- **Time-to-Event**: Regression for predicting time until an event occurs

All grid searches use [Optuna](https://optuna.org/) for hyperparameter optimization.

## Common Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `n_trials` | Number of trials to run in the optimization | `1000` |
| `batch_size` | Training batch size | `[416]` |
| `num_layers` | Number of transformer layers | `[2, 3, 4]` |
| `model_dim` | Dimension of model embeddings | `[256, 512, 1024]` |
| `train_noise` | Noise added during training for regularization | `[1e-5, 1e-3, 1e-2]` |
| `weight_decay` | L2 regularization parameter | `[1e-5, 1e-3, 5e-4]` |
| `dropout` | Dropout probability | `[0.1, 0.3, 0.5]` |
| `num_head` | Number of attention heads | `[16, 32]` |
| `lr` | Learning rate | `[1e-5, 1e-4, 1e-3]` |
| `n_lr_warm_up_steps` | Number of warm-up steps for learning rate | `[0, 100]` |
| `grad_clip_value` | Gradient clipping value | `[0.25, 0.5, 0.75, 1]` |
| `early_stopping_step_limit` | Number of steps without improvement before stopping | `[10]` |
| `max_epochs` | Maximum number of epochs to train | `50` or `100` |

## Encoder-only Specific Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `target_interval` | Whether to use target interval | `true` |
| `restrict_to_first_event` | Whether to restrict to the first event | `false` |
| `scheduler` | Learning rate scheduler type | `["exponential", "cosine"]` |
| `imbalance_factor` | Weight for imbalanced classes | `62` |
| `loss_function` | Type of loss function | `["focal", "bce", "aploss"]` |
| `alpha` | Alpha parameter for focal loss | `[0.25, 0.5, 0.6]` |
| `gamma` | Gamma parameter for focal loss | `[2.0, 3.0, 4.0]` |
| `tau` | Tau parameter for AP loss | `[0.01, 0.1, 1.0, 10.0]` |
| `oversampling_ratio` | Ratio for oversampling minority class | `[1, 10, 50]` |

## Encoder-Decoder Specific Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `num_decoder_layers` | Number of transformer decoder layers | `[6]` |
| `pos_encode_factor` | Factor for positional encoding | `[0.1, 1]` |
| `target_timeseries_length` | Length of target time series | `1` |
| `loss_function` | Type of loss function | `['weighted_mse']` |

## Time-to-Event Specific Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `restrict_to_first_event` | Whether to restrict to the first event | `false` |
| `classification_threshold` | Threshold for classification (in hours) | `6` |
| `loss_function` | Type of loss function | `["log_cosh"]` |

## Example Configuration Files

### Example for Encoder-only Model

```json
{
  "n_trials": 100,
  "target_interval": true,
  "restrict_to_first_event": false,
  "batch_size": [416],
  "num_layers": [3],
  "model_dim": [512],
  "train_noise": [1e-4],
  "weight_decay": [1e-4],
  "dropout": [0.4],
  "num_head": [16],
  "lr": [1e-5],
  "n_lr_warm_up_steps": [100],
  "grad_clip_value": [0.5],
  "early_stopping_step_limit": [10],
  "scheduler": ["cosine"],
  "imbalance_factor": 62,
  "loss_function": ["focal"],
  "alpha": [0.5],
  "gamma": [2.0],
  "tau": [0.1],
  "oversampling_ratio": [10],
  "max_epochs": 100
}
```

### Example for Encoder-Decoder Model

```json
{
  "n_trials": 50,
  "batch_size": [416],
  "num_layers": [6],
  "num_decoder_layers": [6],
  "model_dim": [1024],
  "train_noise": [1e-4],
  "weight_decay": [1e-4],
  "dropout": [0.3],
  "num_head": [16],
  "pos_encode_factor": [0.5],
  "lr": [0.0005],
  "n_lr_warm_up_steps": [0],
  "grad_clip_value": [0.1],
  "early_stopping_step_limit": [10],
  "imbalance_factor": 62,
  "max_epochs": 50,
  "target_timeseries_length": 1,
  "loss_function": ["weighted_mse"]
}
```

### Example for Time-to-Event Model

```json
{
  "n_trials": 50,
  "restrict_to_first_event": false,
  "batch_size": [416],
  "num_layers": [6],
  "model_dim": [1024],
  "train_noise": [1e-4],
  "weight_decay": [1e-4],
  "dropout": [0.3],
  "num_head": [16],
  "lr": [0.0005],
  "n_lr_warm_up_steps": [0],
  "grad_clip_value": [0.1],
  "early_stopping_step_limit": [10],
  "max_epochs": 50,
  "classification_threshold": 6,
  "loss_function": ["log_cosh"]
}
```

## Running Grid Search

To run a grid search, use the appropriate script with a configuration file:

```bash
python prediction/short_term_outcome_prediction/gridsearch_transformer_encoder.py \
  -d /path/to/data_splits.pt \
  -o /path/to/output_folder \
  -c /path/to/config.json \
  -g 1
```

Additional options:
- `-g 1`: Use GPU (0 for CPU)
- `-spwd PASSWORD`: Redis storage password
- `-sport PORT`: Redis storage port
- `-shost HOST`: Redis storage host

For encoder-decoder models, additional parameters may be required:
- `-nd /path/to/normalisation_data.pt`: Path to normalisation data
- `-od /path/to/outcome_data.pt`: Path to outcome data
