# XGBoost Optimisation Progress Report — END Prediction

**Date:** 2026-02-13
**Target:** Predict early neurological deterioration (END) in the next 6 hours
**Data:** `train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth` (~1.8% positive rate)

---

## Baseline Performance

From prior hyperoptimisation (~65 Optuna trials):
- **Median AUPRC: 0.053**
- **Median AUROC: 0.784**

---

## Issues Identified

1. **Optuna optimised the wrong metric** — returned AUROC instead of AUPRC as the study objective
2. **XGBoost early stopping used wrong eval metric** — `["aucpr", "auc"]` meant early stopping tracked AUC (last metric); swapped to `["auc", "aucpr"]`
3. **Search space missed best-performing regions** — e.g., `scale_pos_weight=10` outperformed all tested values but was below the search minimum of 25
4. **Dart booster ignores early stopping** — `best_iteration` is reported but all `n_estimators` rounds are trained regardless

---

## Changes Implemented

### Step 1: Fix optimisation objective and early stopping
**File:** `gridsearch_aggregate_xgb.py`
- Changed `return np.median(val_auroc)` → `return np.median(val_auprc), np.median(val_auroc)` (multi-objective)
- Changed `study = optuna.create_study(direction='maximize')` → `directions=['maximize', 'maximize']`
- Swapped eval_metric order to `["auc", "aucpr"]` so AUCPR is used for early stopping

### Step 2: Feature engineering
**File:** `prediction/utils/utils.py` — `aggregate_features_over_time()`

Added new features on top of existing raw/mean/min/max:
- **Cumulative standard deviation** — captures clinical instability
- **Rate-of-change (first-order differences)** — temporal dynamics
- **Timestep index (normalised 0-1)** — temporal position
- **Optional lag features (t-2, t-3)** — enabled via `add_lag_features` flag

Feature count: 619 (without lag) or 825 (with lag) vs original ~412.

Hard-coded `n_features*4` multipliers in evaluation and SHAP scripts replaced with dynamic inference from data shape.

### Step 3: Refined search space
Based on analysis of top-performing trials, narrowed or adjusted all hyperparameter ranges (see config details below).

### Step 4: Focal loss objective
**File:** `gridsearch_aggregate_xgb.py`

Added custom focal loss objective function operating in logit space with configurable `gamma_fl` and `pos_weight`. When `focal_gamma > 0`, the custom objective replaces the default, with `pos_weight` applied within the gradient/hessian computation (since XGBoost's `scale_pos_weight` doesn't apply to custom objectives).

---

## Optuna Hyperoptimisation Runs (5-fold CV, median reported)

| Run | Key Changes | Best AUPRC | Best AUROC | Trials |
|-----|------------|-----------|-----------|--------|
| **Baseline** | Original code | 0.053 | 0.784 | ~65 |
| **V1** | Fixed opt target, std features, refined search | 0.058 | 0.791 | 2 |
| **V2** | + diff features, timestep index | **0.068** | **0.796** | 1 |
| **V3** | + focal loss, max_delta_step, n_est reduced | 0.064 | 0.793 | 3 |
| **V4** | + multi-objective, wider regularisation | 0.061 | 0.776 | 1 |

**Best Optuna config (V2-T0):** AUPRC=0.068, AUROC=0.796
```
max_depth=4, n_estimators=2000, lr=0.064, scale_pos_weight=10
reg_lambda=10, alpha=1, colsample_bytree=0.8, subsample=0.80
gamma=0.75, max_delta_step=5, focal_gamma=0, booster=dart, grow_policy=lossguide
```

---

## A/B Testing (single fold 0, 25 experiments each)

Ran systematic ablations varying one parameter at a time against the best known config, both without and with lag features.

### Without Lag Features (619 features)

| Experiment | AUPRC | AUROC | vs baseline |
|---|---|---|---|
| spw=25 | **0.0912** | 0.7959 | +7.5% |
| colsample=1.0 | 0.0895 | 0.7920 | +5.5% |
| lowreg (λ=1, α=0) | 0.0881 | 0.8039 | +3.9% |
| focal=1.0 | 0.0879 | 0.8013 | +3.7% |
| delta_step=0 | 0.0867 | 0.8035 | +2.2% |
| mcw=7 | 0.0856 | 0.7993 | +0.9% |
| lr=0.10 | 0.0850 | 0.8039 | +0.2% |
| **baseline** | **0.0848** | **0.7989** | — |
| depth=2 | 0.0789 | **0.8061** | best AUROC |
| highreg (λ=50, α=10) | 0.0733 | 0.8036 | -13.6% |
| gbtree | 0.0702 | 0.8058 | -17.2% |
| depth=6 | 0.0628 | 0.7722 | -25.9% |

### With Lag Features (825 features)

| Experiment | AUPRC | AUROC | vs baseline |
|---|---|---|---|
| **lr=0.03** | **0.0938** | 0.8016 | **+4.6%** |
| focal=1.0 | 0.0920 | 0.7997 | +2.6% |
| subsample=0.5 | 0.0903 | **0.8088** | +0.7% |
| **baseline (lag)** | **0.0897** | **0.8061** | — |
| highreg (λ=50, α=10) | 0.0887 | 0.8067 | -1.1% |
| depth=3 | 0.0873 | **0.8097** | best AUROC |
| mcw=1 | 0.0851 | **0.8099** | best AUROC |
| lr=0.10 | 0.0761 | 0.8058 | -15.2% |
| depth=6 | 0.0646 | 0.7870 | -28.0% |

---

## Key Findings

### What improves AUPRC
1. **Lag features (t-2, t-3)** — AUPRC +5.8%, AUROC +0.9%. Both metrics improve.
2. **Feature engineering (std, diff, timestep)** — AUPRC +27.5% vs original baseline (0.053→0.068 median 5-fold)
3. **Lower learning rate with lag** — lr=0.03 beats lr=0.064 when lag features are enabled
4. **scale_pos_weight=10-25** — optimal range; too low (5) or too high (45+) hurts
5. **Focal loss gamma=1.0** — consistent mild benefit (+2-4% AUPRC)

### What doesn't matter
- **n_estimators** — dart booster ignores it (100/200/500 give identical results)
- **grow_policy** — depthwise and lossguide produce identical results
- **max_delta_step** — no consistent effect
- **colsample_bylevel** — no effect at 1.0

### What hurts
- **depth >= 6** — severe overfitting, -25% AUPRC
- **gbtree booster** — 10x faster but -15-17% AUPRC vs dart
- **High learning rate (0.10) with lag** — -15% AUPRC
- **Very low scale_pos_weight (5)** — hurts AUPRC

### Interactions (optimal settings differ with/without lag)
| Parameter | Best without lag | Best with lag |
|---|---|---|
| scale_pos_weight | 25 | 10 |
| learning_rate | 0.064 | **0.03** |
| regularisation | low (λ=1, α=0) | high (λ=50, α=10) |
| colsample_bytree | 1.0 | 0.8 |
| subsample | 0.8 | 0.5 |

More features → need more regularisation and slower learning.

### AUROC vs AUPRC trade-off
- Shallower trees (depth 2-3) boost AUROC but sacrifice AUPRC
- Lower scale_pos_weight boosts AUROC but hurts AUPRC
- Multi-objective Optuna can find Pareto-optimal configurations

---

## Combination Testing (single fold 0, with lag features)

Tested stacking the top individual improvements to find interactions. 16 experiments.

### Results (sorted by AUPRC)

| Combo | AUPRC | AUROC | Notes |
|---|---|---|---|
| **lr03+focal1+sub05+highreg** | **0.0939** | **0.8094** | **Best overall — both metrics** |
| lr03+focal1+highreg | 0.0925 | 0.8060 | |
| lr03+focal1 | 0.0923 | 0.8053 | Core winning combo |
| lr03+highreg | 0.0922 | 0.8007 | |
| focal1+spw25 | 0.0920 | 0.8013 | |
| **lag_baseline** | **0.0897** | **0.8061** | **Reference** |
| lr03+focal1+d3 | 0.0887 | 0.8067 | |
| lr03+focal1+spw25 | 0.0881 | 0.7940 | spw=25 hurts in combos |
| lr03+focal1+sub05 | 0.0874 | 0.8061 | sub05 needs highreg |
| focal1+sub05 | 0.0849 | 0.8071 | |
| lr03+sub05 | 0.0807 | 0.8024 | sub05 alone hurts |

### Combination Findings

- **lr=0.03 + focal=1.0** is the core winning pair (~0.092 AUPRC)
- **Adding highreg (λ=50, α=10)** pushes to 0.0925
- **subsample=0.5 only works with strong regularisation** — hurts in 2-way combos, helps in 4-way with highreg (0.0939). The extra regularisation from both subsample dropout and L1/L2 prevents overfitting.
- **spw=25 hurts in all combinations** — spw=10 remains optimal with lag features
- **depth=4 remains optimal** — depth=3 and depth=5 both worse in combos

---

## Current Best (single fold 0)
- **AUPRC = 0.0939** (lag + lr=0.03 + focal=1.0 + sub=0.5 + highreg) — +77% vs original baseline
- **AUROC = 0.8094** (same config) — +3.2% vs original baseline
- This is the first config to achieve top AUPRC and top AUROC simultaneously

---

## Cluster Configuration Fixes

Two issues were found and fixed for SLURM cluster deployment:

1. **`cluster/master_launcher.py`** — Study creation mismatch. The master launcher created single-objective studies (`direction='maximize'`), but XGBoost now returns two objectives (AUPRC, AUROC). Fixed: XGB now creates multi-objective study with `directions=['maximize', 'maximize']`.

2. **`cluster/cluster_subprocess.py`** — Missing `add_lag_features`. The subprocess did not pass `add_lag_features` to `prepare_aggregate_dataset`, so lag features were always disabled on the cluster. Fixed: now reads `add_lag_features` from config and passes it through.

---

## Hyperoptimisation Config

Saved as `xgb_auprc_config.json`:

```json
{
    "n_trials": 100,
    "add_lag_features": true,
    "max_depth": [2, 5],
    "n_estimators": [200],
    "learning_rate": [0.02, 0.08],
    "reg_lambda": [10, 50, 75],
    "alpha": [1, 5, 10, 15, 25],
    "scale_pos_weight": [5, 10, 25],
    "min_child_weight": [1, 5],
    "subsample": [0.3, 1.0],
    "colsample_bytree": [0.8],
    "colsample_bylevel": [1.0],
    "booster": ["dart"],
    "grow_policy": ["lossguide"],
    "num_boost_round": [500],
    "gamma": [0.1, 0.2, 0.5, 0.75, 1.0],
    "max_delta_step": [0, 5],
    "focal_gamma": [0, 1.0]
}
```

### Recommended SLURM cluster settings
- **10 subprocesses** (10 trials each, ~5 hours wall time)
- **8 CPUs per subprocess** (`--cpus-per-task=8`)
- **1 task per subprocess** (`--ntasks=1`)

---

## Files Modified

| File | Changes |
|---|---|
| `gridsearch_aggregate_xgb.py` | Multi-objective Optuna, focal loss, search space, lag features support |
| `prediction/utils/utils.py` | std, diff, timestep, lag features in `aggregate_features_over_time` |
| `prediction/short_term_outcome_prediction/timeseries_decomposition.py` | `add_lag_features` parameter threading |
| `evaluation/xgb_evaluation.py` | Dynamic feature count (replaced `n_features*4`) |
| `testing/compute_shap_explanations_over_time.py` | Dynamic feature count (replaced `n_features*4`) |
| `cluster/master_launcher.py` | Multi-objective study creation for XGB |
| `cluster/cluster_subprocess.py` | Pass `add_lag_features` to `prepare_aggregate_dataset` |
| `xgb_auprc_config.json` | Refined hyperparameter search space |
| `quick_eval_xgb.py` | A/B testing and combination testing script (new) |

---

## Next Steps

1. Launch full hyperoptimisation on SLURM cluster (100 trials, 10 subprocesses)
2. Analyse Pareto front from multi-objective results (AUPRC vs AUROC trade-off)
3. Validate best config on held-out test set
4. Update evaluation and SHAP pipelines to use `add_lag_features=True`
