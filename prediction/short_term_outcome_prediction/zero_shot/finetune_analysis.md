# MIRA Finetuning Trial Analysis

**Date**: 2026-02-10
**Results dir**: `mira_finetune_results_20260210_070349/`

## Setup

- **Approach**: Frozen MIRA backbone (4096-dim hidden state) with a trainable linear classification head (4096 -> 256 -> 1)
- **Task**: Binary classification of early neurological deterioration (next 6h)
- **Input**: Single-channel `max_NIHSS` time series (up to 72 timesteps)
- **Loss**: BCEWithLogitsLoss with positive class weighting
- **Config**: LR=0.001, batch_size=32, cosine annealing, patience=5, dropout=0.1

## Results (first 3 folds)

| Fold | Best Epoch | Val AUROC | Val AUPRC | Val MCC | Val Accuracy | Val Loss |
|------|-----------|-----------|-----------|---------|--------------|----------|
| 0    | 1         | 0.652     | 0.0048    | 0.0     | 99.71%       | 5.73     |
| 1    | 1         | 0.625     | 0.0042    | 0.0     | 99.71%       | 7.05     |
| 2    | (incomplete - only best_model.pt saved, no history/results CSV) |||||

## Key Observations

### 1. Model is not learning
Early stopping triggered at epoch 6 on both completed folds. Best AUROC was at epoch 1 in both cases. Performance degraded throughout training:
- Fold 0 AUROC: 0.652 -> 0.563 over 6 epochs
- Fold 1 AUROC: 0.625 -> 0.612 over 6 epochs

### 2. Extreme class imbalance dominates
Accuracy is ~99.71%, meaning positive cases (neurological deterioration) make up only ~0.3% of samples. The model predicts all-negative (MCC = 0.0 across all epochs). AUPRC values (~0.004-0.005) are barely above random baseline (~0.003).

### 3. Some discriminative signal exists
AUROC values (0.62-0.65) are above chance, suggesting the MIRA backbone extracts some useful features from max_NIHSS, but the model cannot translate this into useful predictions.

### 4. Training is unstable
Validation loss oscillates significantly (e.g., fold 0: 5.73 -> 6.76 -> 6.27 -> 5.86 -> 5.88 -> 5.82). The frozen backbone means only the classification head (~1M params) is being optimized.

## Root Causes

- **Sample construction inflates imbalance**: Each patient generates 72 samples (one per timestep), most negative. This creates a ~99.7% negative dataset. Positive events at specific future timesteps are very sparse.
- **Single-feature input**: Only `max_NIHSS` is used, limiting the information available.
- **Frozen backbone**: MIRA was pretrained for general time-series forecasting, not clinical classification. With `unfreeze_last_n_layers=0`, the representation may not suit this task.
- **High learning rate**: 0.001 is aggressive for a linear probe on frozen features.

## Recommendations for Next Trial

1. **Subsample negatives or aggregate per-patient**: Instead of 72 samples per patient, use per-patient or per-window prediction to reduce extreme class imbalance.
2. **Unfreeze some backbone layers**: Try `unfreeze_last_n_layers=2-4` so the representation can adapt.
3. **Use a different pooling strategy**: Mean pooling over non-padded tokens instead of last-token pooling.
4. **Lower the learning rate**: Try 1e-4 or 1e-5 for a linear probe on frozen features.
5. **Consider focal loss** instead of weighted BCE for extreme imbalance.
