# CLAUDE.md


## Project Overview
Evaluate models at zero-shot prediction of early neurological deterioration in the next 6 hours

models to evaluate: 
- MIRA

### Plan
- build data loader: transform data to MIRA input format
- build inference script
- evaluate zero shot inference of MIRA on validation data of all splits and report metrics 

### Data
- validation data path: /mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth
- data structure: splits with training and validation data (input data and labels)

### Ressources
- MIRA repo: https://github.com/microsoft/MIRA
- MIRA model on HF: https://huggingface.co/MIRA-Mode/MIRA


# Rest of overall OPSUM Project
## Running Models

### Local hyperparameter search (single GPU)
```bash
python prediction/short_term_outcome_prediction/gridsearch_transformer_encoder.py \
  -d /path/to/data_splits.pt \
  -o /path/to/output \
  -c /path/to/config.json \
  -g 1
```

Options: `-g 0` for CPU, `-g 1` for GPU

### Distributed SLURM cluster execution
```bash
python prediction/short_term_outcome_prediction/cluster/master_launcher.py \
  -d /path/to/data_splits.pth \
  -o /path/to/output \
  -c /path/to/config.json \
  -n 10 \
  -spwd <redis_password> \
  -sport <redis_port> \
  -shost <redis_host> \
  -f  # Optional: enable Optuna dashboard
```

Model type flags: `-tte` (time-to-event), `-dec` (encoder-decoder), `-xgb` (XGBoost)

### Running tests
```bash
pytest tests/timeseries_decomposition_tests.py
```

## Architecture

### Key Directories
- `preprocessing/geneva_stroke_unit_preprocessing/` - EHR data preprocessing pipeline
- `prediction/short_term_outcome_prediction/` - Main short-term prediction models (active development)
- `prediction/outcome_prediction/` - Long-term prediction models (Transformer, LSTM, XGBoost)
- `prediction/utils/` - Shared utilities (loss functions, scoring, calibration)

### Model Types
1. **Transformer Encoder** (`gridsearch_transformer_encoder.py`) - Binary classification
2. **Transformer Encoder-Decoder** (`gridsearch_transformer_encoder_decoder.py`) - Time-series forecasting
3. **Time-to-Event** (`gridsearch_transformer_encoder_time_to_event.py`) - Regression predicting time until event
4. **XGBoost** (`gridsearch_aggregate_xgb.py`) - Tree-based baseline on aggregated features

### Data Flow
1. `data_splits.py` - Splits data by patient ID, generates K-fold cross-validation
2. `timeseries_decomposition.py` - Decomposes time series into labeled subsequences
3. Training scripts use PyTorch Lightning wrapper (`prediction/outcome_prediction/Transformer/lightning_wrapper.py`)
4. Optuna handles hyperparameter optimization with Redis for distributed storage

### Core Classes
- `prediction/outcome_prediction/Transformer/architecture.py` - Transformer implementation (Encoder, MultiHeadedAttention, PositionalEncoding)
- `prediction/utils/loss_functions.py` - Focal Loss, SoftAPLoss, Weighted MSE
- `prediction/short_term_outcome_prediction/timeseries_decomposition.py` - BucketBatchSampler, subsequence labeling

### Configuration
Hyperparameter configs are JSON files. See `prediction/short_term_outcome_prediction/gridsearch_readMe.md` for parameter documentation. Key parameters:
- `n_trials` - Optuna trials
- `batch_size`, `num_layers`, `model_dim`, `num_head` - Model architecture
- `loss_function` - "focal", "bce", "aploss" (encoder), "weighted_mse" (decoder), "log_cosh" (time-to-event)
- `imbalance_factor` - Class weight for imbalanced data

### Evaluation Metrics
Binary classification: AUROC, AUPRC, MCC, Accuracy
Time-to-event: MSE/MAE
Calibration: Temperature scaling, Brier score
