# Publication Figures for END Prediction

Scripts for generating publication-quality figures comparing XGBoost vs Logistic Regression for early neurological deterioration (END) prediction.

## Prerequisites

Use the `opsum` conda environment:

```bash
conda activate opsum
```

## Standalone Figures

All scripts follow the same CLI pattern. Run from the repository root:

```bash
XGB_DIR=/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/test_results/
LR_DIR=/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/lr_test_results/
OUT_DIR=/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/figures/
```

### ROC Curve Comparison

```bash
PYTHONPATH=/home/klug/opsum python prediction/short_term_outcome_prediction/figures/plot_roc_curves.py \
    --xgb_dir $XGB_DIR --lr_dir $LR_DIR -o $OUT_DIR
```

### Precision-Recall Curve Comparison

```bash
PYTHONPATH=/home/klug/opsum python prediction/short_term_outcome_prediction/figures/plot_pr_curves.py \
    --xgb_dir $XGB_DIR --lr_dir $LR_DIR -o $OUT_DIR
```

### AUROC Over Time (72h)

```bash
PYTHONPATH=/home/klug/opsum python prediction/short_term_outcome_prediction/figures/plot_auroc_over_time.py \
    --xgb_dir $XGB_DIR --lr_dir $LR_DIR -o $OUT_DIR
```

### Calibration Plot

```bash
PYTHONPATH=/home/klug/opsum python prediction/short_term_outcome_prediction/figures/plot_calibration.py \
    --xgb_dir $XGB_DIR --lr_dir $LR_DIR -o $OUT_DIR
```

### SHAP Beeswarm Plot

Requires SHAP values computed beforehand (see [SHAP Computation](#shap-computation) below).

```bash
MODEL_DIR=/mnt/data1/klug/output/opsum/short_term_outcomes/end_with_imaging/ss_xgb_eval/
SHAP_DIR=$MODEL_DIR/shap_explanations_over_time
TEST_DATA=/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/test_data_early_neurological_deterioration_ts0.8_rs42_ns5.pth
CAT_ENC=/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/logs_30012026_154047/categorical_variable_encoding.csv
FEAT_NAMES=/home/klug/opsum/preprocessing/preprocessing_tools/feature_name_to_english_name_correspondence.xlsx

PYTHONPATH=/home/klug/opsum python prediction/short_term_outcome_prediction/figures/plot_shap_beeswarm.py \
    --shap_path $SHAP_DIR/tree_explainer_shap_values_over_ts.pkl \
    --test_data_path $TEST_DATA \
    --cat_encoding_path $CAT_ENC \
    --feature_names_path $FEAT_NAMES \
    --n_top_features 10 \
    --add_lag_features --add_rolling_features \
    -o $OUT_DIR
```

## SHAP Computation

Compute SHAP explanations over time for the final XGBoost model:

```bash
PYTHONPATH=/home/klug/opsum python prediction/short_term_outcome_prediction/testing/compute_shap_explanations_over_time.py \
    --final \
    -d $TEST_DATA \
    -m $MODEL_DIR \
    -o $SHAP_DIR
```

This produces:
- `tree_explainer_shap_values_over_ts.pkl` — SHAP values per timestep
- `shap_feature_names.pkl` — ordered list of aggregated feature names

## Embedding in Combined Figures

Each script exposes a reusable plotting function that accepts a matplotlib `Axes` object, for use in multi-panel publication figures:

```python
from plot_config import setup_theme, PUB_TICK, PUB_LABEL
from plot_roc_curves import plot_roc
from plot_pr_curves import plot_pr
from plot_auroc_over_time import plot_auroc_time
from plot_calibration import plot_calibration

setup_theme()
fig, axes = plt.subplots(2, 2)

plot_roc(axes[0, 0], xgb_dir, lr_dir, tick_size=PUB_TICK, label_size=PUB_LABEL)
plot_pr(axes[0, 1], xgb_dir, lr_dir, tick_size=PUB_TICK, label_size=PUB_LABEL)
plot_auroc_time(axes[1, 0], xgb_dir, lr_dir, tick_size=PUB_TICK, label_size=PUB_LABEL)
plot_calibration(axes[1, 1], xgb_dir, lr_dir, tick_size=PUB_TICK, label_size=PUB_LABEL)
```

## Output

Each script saves two files per figure:
- `.svg` (1200 dpi) — for editing and submission
- `.tiff` (900 dpi) — for journal requirements
