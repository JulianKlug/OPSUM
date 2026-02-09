# MIRA Zero-Shot Evaluation Progress

## Objective
Evaluate the pretrained MIRA model (Microsoft's Medical Time Series Foundation Model) at zero-shot prediction of early neurological deterioration in the next 6 hours.

## Approach
**Forecasting-based classification**:
1. Use MIRA to forecast the next 6 timesteps for max_NIHSS feature
2. Apply reverse scaling and normalization to get actual NIHSS values
3. Classify as deterioration if delta_NIHSS (max_NIHSS_forecast - min_NIHSS_historical) >= 4
4. Evaluate using AUROC, AUPRC, MCC, and Accuracy metrics

## Implementation Status

### Completed Tasks
1. **Created conda environment** (`mira`) with dependencies:
   - Python 3.10
   - PyTorch, transformers, huggingface_hub
   - torchdiffeq (for ODE solver)
   - pandas==1.4.4, numpy==1.23.5 (for compatibility with pickled data)
   - MIRA repository cloned from GitHub

2. **Created modules**:
   - `mira_data_loader.py` - Data loading and transformation
   - `mira_inference.py` - MIRA model loading and forecasting
   - `mira_evaluation.py` - Evaluation and metrics computation
   - `run_mira_evaluation.py` - Main CLI script

3. **Fixed issues**:
   - ODE underflow error: Disabled ODE extrapolation by modifying config before loading
   - NaN values for short sequences: Fixed normalization to handle single-element sequences
   - pandas compatibility: Used older pandas version for pickled data

4. **Ran evaluation** on fold 0 (single fold, no cross-validation)

## Data Files
- **Data splits**: `/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth`
- **Normalisation params**: `/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/logs_30012026_154047/normalisation_parameters.csv`
- **Outcome labels**: `/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/preprocessed_outcomes_short_term_30012026_154047.csv`

## Results (Fold 0)

### Configuration
- Forecast horizon: 6 hours
- Timesteps: 72
- Validation samples: 430 patients
- Total predictions: 30,960
- Positive samples: 89 (0.29% - severe class imbalance)

### Overall Metrics
| Metric | Value |
|--------|-------|
| AUROC | 0.614 |
| AUPRC | 0.006 |
| MCC | 0.018 |
| Accuracy | 93.3% |

### Median Per-Timestep Metrics
| Metric | Value |
|--------|-------|
| Median AUROC | 0.747 |
| Median AUPRC | 0.017 |
| Median MCC | -0.013 |
| Median Accuracy | 92.6% |

### Analysis
- AUROC of 0.614 is better than random (0.5) but modest for zero-shot prediction
- Very low AUPRC (0.006) reflects severe class imbalance
- High accuracy is misleading due to class imbalance
- Per-timestep performance varies significantly (some achieve AUROC=1.0, others near 0.05)
- Median AUROC (0.747) is more representative than mean

## Output Files
Results saved to: `/home/klug/opsum/prediction/short_term_outcome_prediction/zero_shot/mira_evaluation_results_20260209_115618/`
- `all_folds_overall_validation_results.csv` - Overall metrics
- `cv_fold_0/validation_scores_per_timestep.csv` - Per-timestep metrics
- `cv_fold_0/validation_scores_over_time.png` - Visualization
- `cv_fold_0/overall_validation_predictions.csv` - Raw predictions

## How to Run

### Single fold evaluation
```bash
source ~/utils/miniconda3/bin/activate mira
cd /home/klug/opsum/prediction/short_term_outcome_prediction/zero_shot
PYTHONPATH=./MIRA:$PYTHONPATH python run_mira_evaluation.py --use_gpu
```

### Cross-validation (all 5 folds)
```bash
source ~/utils/miniconda3/bin/activate mira
cd /home/klug/opsum/prediction/short_term_outcome_prediction/zero_shot
PYTHONPATH=./MIRA:$PYTHONPATH python run_mira_evaluation.py --use_gpu --use_cross_validation
```

### CLI Options
```
-d, --data_path           Path to data splits .pth file
-n, --normalisation_path  Path to normalisation parameters CSV
-o, --outcome_path        Path to outcomes CSV
--output_path             Output directory for results
--use_gpu                 Use GPU if available
--use_cross_validation    Run on all 5 CV folds
--n_time_steps            Number of timesteps (default: 72)
--n_forecast_steps        Forecast horizon in hours (default: 6)
--batch_size              Batch size for inference (default: 64)
```

## Key Code Changes

### mira_inference.py - ODE Disable Fix
```python
def load_mira_model(model_name: str = "MIRA-Mode/MIRA", device: str = "cuda", disable_ode: bool = True):
    if disable_ode:
        # Load config first and disable ODE before model initialization
        config = MIRAConfig.from_pretrained(model_name)
        config.use_terminal_ode = False
        model = MIRAForPrediction.from_pretrained(model_name, config=config)
        model.use_terminal_ode = False
        model.ode_extrapolation_block = None
    else:
        model = MIRAForPrediction.from_pretrained(model_name)

    model = model.to(device)
    model.eval()
    return model
```

### mira_inference.py - Single Element Normalization Fix
```python
def normalize_sequence(values: torch.Tensor):
    mean = values.mean(dim=1, keepdim=True)

    # Handle single-element sequences (std would be 0 or undefined)
    if values.shape[1] == 1:
        std = torch.ones_like(mean)
    else:
        std = values.std(dim=1, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)

    std = std + 1e-8  # Small epsilon for numerical stability
    normalized = (values - mean) / std
    return normalized, mean, std
```

## Next Steps
1. Run full cross-validation across all 5 folds
2. Compare results with trained encoder-decoder model
3. Consider alternative classification approaches (e.g., different thresholds, ensemble methods)
4. Investigate why some timesteps have very poor performance

## Notes
- Evaluation on CPU takes ~5.5 hours per fold (72 timesteps × ~5 min per timestep)
- MIRA model has 455M parameters with 12 transformer layers
- MIRA uses CT-RoPE (Continuous-Time Rotary Positional Encoding) for time encoding
- ODE extrapolation was disabled due to numerical instability with short sequences
