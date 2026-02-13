"""Quick XGB evaluation script for A/B testing hyperparameter choices.

Runs a fixed config on a single fold for fast iteration.
Use --all-folds for full 5-fold CV validation of promising configs.

Usage:
    python quick_eval_xgb.py -d <data_splits_path> [--lag] [--fold 0] [--all-folds]
"""
import time
from functools import partial
import numpy as np
import torch as ch

from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

from prediction.short_term_outcome_prediction.timeseries_decomposition import prepare_aggregate_dataset
from prediction.short_term_outcome_prediction.gridsearch_aggregate_xgb import focal_loss_objective


def evaluate_config(datasets, config, name=""):
    val_auprc = []
    val_auroc = []
    t0 = time.time()

    device = "cuda" if ch.cuda.is_available() else "cpu"

    focal_gamma = config.pop('focal_gamma', 0)
    scale_pos_weight = config.pop('scale_pos_weight', 10)
    early_stopping_rounds = config.pop('early_stopping_rounds', 50)

    xgb_params = dict(config)
    xgb_params['device'] = device
    xgb_params['reg_alpha'] = xgb_params.pop('alpha', 1)

    if focal_gamma > 0:
        xgb_params['objective'] = partial(focal_loss_objective, gamma_fl=focal_gamma, pos_weight=scale_pos_weight)
        xgb_params['scale_pos_weight'] = 1
    else:
        xgb_params['scale_pos_weight'] = scale_pos_weight

    for i, (X_train, X_val, y_train, y_val) in enumerate(datasets):
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds,
                  eval_metric=["auc", "aucpr"],
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1].astype('float32')
        auprc = average_precision_score(y_val, y_prob)
        auroc = roc_auc_score(y_val, y_prob)
        val_auprc.append(auprc)
        val_auroc.append(auroc)

    elapsed = time.time() - t0
    med_auprc = np.median(val_auprc)
    med_auroc = np.median(val_auroc)
    print(f"  {name:30s}  AUPRC={med_auprc:.4f}  AUROC={med_auroc:.4f}  ({elapsed:.0f}s)")
    return med_auprc, med_auroc


# ── Best known config (V2-T0) ──────────────────────────────────────────
BEST_CONFIG = dict(
    max_depth=4,
    n_estimators=200,
    learning_rate=0.064,
    reg_lambda=10,
    alpha=1,
    scale_pos_weight=10,
    min_child_weight=3,
    subsample=0.80,
    colsample_bytree=0.8,
    colsample_bylevel=1.0,
    booster="dart",
    grow_policy="lossguide",
    gamma=0.75,
    max_delta_step=5,
    focal_gamma=0,
    early_stopping_rounds=50,
)


def run_experiments(datasets, tag=""):
    experiments = {}

    # ── Baseline: best known config ──
    experiments['baseline'] = dict(BEST_CONFIG)

    # ── n_estimators ──
    c = dict(BEST_CONFIG); c['n_estimators'] = 100
    experiments['n_est=100'] = c
    c = dict(BEST_CONFIG); c['n_estimators'] = 500
    experiments['n_est=500'] = c

    # ── scale_pos_weight ──
    c = dict(BEST_CONFIG); c['scale_pos_weight'] = 5
    experiments['spw=5'] = c
    c = dict(BEST_CONFIG); c['scale_pos_weight'] = 25
    experiments['spw=25'] = c
    c = dict(BEST_CONFIG); c['scale_pos_weight'] = 45
    experiments['spw=45'] = c

    # ── focal loss ──
    c = dict(BEST_CONFIG); c['focal_gamma'] = 1.0
    experiments['focal=1.0'] = c
    c = dict(BEST_CONFIG); c['focal_gamma'] = 2.0
    experiments['focal=2.0'] = c

    # ── max_depth ──
    c = dict(BEST_CONFIG); c['max_depth'] = 2
    experiments['depth=2'] = c
    c = dict(BEST_CONFIG); c['max_depth'] = 3
    experiments['depth=3'] = c
    c = dict(BEST_CONFIG); c['max_depth'] = 6
    experiments['depth=6'] = c
    c = dict(BEST_CONFIG); c['max_depth'] = 8
    experiments['depth=8'] = c

    # ── booster ──
    c = dict(BEST_CONFIG); c['booster'] = 'gbtree'
    experiments['gbtree'] = c

    # ── regularization ──
    c = dict(BEST_CONFIG); c['reg_lambda'] = 50; c['alpha'] = 10
    experiments['highreg'] = c
    c = dict(BEST_CONFIG); c['reg_lambda'] = 1; c['alpha'] = 0
    experiments['lowreg'] = c

    # ── grow_policy ──
    c = dict(BEST_CONFIG); c['grow_policy'] = 'depthwise'
    experiments['depthwise'] = c

    # ── subsample ──
    c = dict(BEST_CONFIG); c['subsample'] = 0.5
    experiments['subsample=0.5'] = c

    # ── colsample_bytree ──
    c = dict(BEST_CONFIG); c['colsample_bytree'] = 0.5
    experiments['colsample=0.5'] = c
    c = dict(BEST_CONFIG); c['colsample_bytree'] = 1.0
    experiments['colsample=1.0'] = c

    # ── max_delta_step ──
    c = dict(BEST_CONFIG); c['max_delta_step'] = 0
    experiments['delta_step=0'] = c
    c = dict(BEST_CONFIG); c['max_delta_step'] = 10
    experiments['delta_step=10'] = c

    # ── learning rate ──
    c = dict(BEST_CONFIG); c['learning_rate'] = 0.03
    experiments['lr=0.03'] = c
    c = dict(BEST_CONFIG); c['learning_rate'] = 0.10
    experiments['lr=0.10'] = c

    # ── min_child_weight ──
    c = dict(BEST_CONFIG); c['min_child_weight'] = 1
    experiments['mcw=1'] = c
    c = dict(BEST_CONFIG); c['min_child_weight'] = 7
    experiments['mcw=7'] = c

    print(f"\n{'='*60}")
    print(f"  {tag} — {len(experiments)} experiments")
    print(f"{'='*60}")
    print(f"  {'Experiment':30s}  {'AUPRC':>7s}  {'AUROC':>7s}  {'Time':>5s}")
    print(f"  {'-'*60}")

    results = {}
    for name, config in experiments.items():
        results[name] = evaluate_config(datasets, dict(config), name=name)

    # Summary sorted by AUPRC
    print(f"\n  SUMMARY (sorted by AUPRC):")
    for name, (auprc, auroc) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = " <-- baseline" if name == 'baseline' else ""
        print(f"    {name:30s}  AUPRC={auprc:.4f}  AUROC={auroc:.4f}{marker}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('--lag', action='store_true', help='Also test with lag features')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to use (default: 0)')
    parser.add_argument('--all-folds', action='store_true', help='Use all 5 folds instead of single fold')
    args = parser.parse_args()

    splits = ch.load(args.data_splits_path)

    # Prepare single fold (fast) or all folds
    if args.all_folds:
        print("Preparing ALL folds (lag=False)...")
        datasets = [prepare_aggregate_dataset(x, rescale=True, target_time_to_outcome=6,
                                              target_interval=1, restrict_to_first_event=0,
                                              add_lag_features=False) for x in splits]
    else:
        print(f"Preparing fold {args.fold} only (lag=False)...")
        datasets = [prepare_aggregate_dataset(splits[args.fold], rescale=True, target_time_to_outcome=6,
                                              target_interval=1, restrict_to_first_event=0,
                                              add_lag_features=False)]

    print(f"Data ready: {datasets[0][0].shape[1]} features, {len(datasets)} fold(s)\n")

    run_experiments(datasets, tag="NO LAG FEATURES")

    if args.lag:
        if args.all_folds:
            print("\nPreparing ALL folds (lag=True)...")
            datasets_lag = [prepare_aggregate_dataset(x, rescale=True, target_time_to_outcome=6,
                                                      target_interval=1, restrict_to_first_event=0,
                                                      add_lag_features=True) for x in splits]
        else:
            print(f"\nPreparing fold {args.fold} only (lag=True)...")
            datasets_lag = [prepare_aggregate_dataset(splits[args.fold], rescale=True, target_time_to_outcome=6,
                                                      target_interval=1, restrict_to_first_event=0,
                                                      add_lag_features=True)]

        print(f"Data ready: {datasets_lag[0][0].shape[1]} features, {len(datasets_lag)} fold(s)\n")
        run_experiments(datasets_lag, tag="WITH LAG FEATURES")
