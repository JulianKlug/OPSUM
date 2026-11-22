"""
Admission-only baseline models vs the time-series XGB END model (reviewer response).

    same patients as the XGB END model
        training = fold 0 train + val (= retrain_on_all_data), CV = the same 5 folds
        internal test (GSU held-out), external test (MIMIC)

    admission table (admission_features.py)
        |-- variable selection on training patients (variable_selection.py)
        |-- models: lr_literature, lr_selected, lr_selected_no_imaging, xgb_baseline
        |           hyperparameters: median validation AUROC over the 5 folds; threshold: Youden J on out-of-fold
        '-- tasks
              hourly   same rows / labels as XGB END (END in next 6 h at each hour), features = admission + ts
              patient  one row per patient, label = any END after hour 0;
                       XGB END patient score = max hourly probability before the first END

    -> metrics with patient-bootstrap 95% CI, paired comparison vs XGB END
"""
import argparse
import os
from enum import Enum

import numpy as np
import pandas as pd
import torch as ch
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from prediction.short_term_outcome_prediction.evaluation.baseline_model.admission_features import \
    TIMESTEP, IMAGING_LABELS, extract_admission_features, load_admission_imaging, count_late_imaging, \
    fit_binary_rescaling, apply_binary_rescaling, hourly_labels, patient_labels, predictable_events, \
    literature_features
from prediction.short_term_outcome_prediction.evaluation.baseline_model.model_comparison import \
    paired_cluster_bootstrap, delong_test
from prediction.short_term_outcome_prediction.evaluation.baseline_model.variable_selection import \
    univariable_table, choose_l1_C, stability_selection, select_final_variables, fit_unpenalised_lr, wald_table, \
    MAX_ITER
from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.concern_timing_analysis import \
    predict_hourly, load_thresholds, PRIMARY_THRESHOLD
from prediction.short_term_outcome_prediction.evaluation.first_event.first_event_evaluation import \
    _metrics_with_ci, _counts, restrict_to_first_event, Cohort, SEED
from prediction.utils.utils import ensure_dir

LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
XGB_GRID = [{'max_depth': d, 'min_child_weight': w} for d in (2, 3, 4) for w in (1, 10)]
XGB_FIXED = dict(learning_rate=0.05, n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
                 tree_method='hist', eval_metric='auc', use_label_encoder=False, random_state=SEED)
XGB_EARLY_STOPPING = 50
TS_SPLINE_KNOTS = 5
MIN_EVENTS_PER_VARIABLE = 10


class Task(str, Enum):
    HOURLY = 'hourly'
    PATIENT = 'patient'


class Model(str, Enum):
    XGB_END = 'xgb_end'  # time-series reference
    LR_LITERATURE = 'lr_literature'
    LR_SELECTED = 'lr_selected'
    LR_SELECTED_NO_IMAGING = 'lr_selected_no_imaging'
    XGB_BASELINE = 'xgb_baseline'


BASELINE_MODELS = [Model.LR_LITERATURE, Model.LR_SELECTED, Model.LR_SELECTED_NO_IMAGING, Model.XGB_BASELINE]
TEST_COHORTS = [Cohort.INTERNAL_TEST, Cohort.EXTERNAL_TEST]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_training(splits_path: str) -> tuple:
    """All training patients (fold 0 train + val, as retrain_on_all_data) and the validation cids of each fold."""
    splits = ch.load(splits_path, weights_only=False)
    X_train, X_val, y_train, y_val = splits[0]
    X = np.concatenate([X_train, X_val])
    fold_val_cids = [set(split[1][:, 0, 0, 0]) for split in splits]
    return X, pd.concat([y_train, y_val], ignore_index=True), fold_val_cids


def fold_indices(cids: np.ndarray, fold_val_cids: list) -> list:
    """(train_idx, val_idx) per fold for rows identified by cids."""
    folds = []
    for val_cids in fold_val_cids:
        is_val = np.isin(cids, list(val_cids))
        folds.append((np.where(~is_val)[0], np.where(is_val)[0]))
    return folds


def task_frame(admission: pd.DataFrame, hourly: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Rows of the task: hourly -> admission features joined to each (cid, ts); patient -> one row per cid."""
    if task == Task.HOURLY:
        return hourly[['case_admission_id', TIMESTEP, 'y_true']].merge(
            admission, left_on='case_admission_id', right_index=True, how='left')
    labels = patient_labels(hourly).rename('y_true')
    return admission.join(labels, how='inner').rename_axis('case_admission_id').reset_index()


def task_columns(features: list, task: Task) -> list:
    return features + [TIMESTEP] if task == Task.HOURLY else features


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_lr(C: float, task: Task) -> Pipeline:
    """L2 LR; hourly task models the time-varying baseline risk with a cubic spline of ts."""
    steps = []
    if task == Task.HOURLY:
        spline = ColumnTransformer([('ts', SplineTransformer(n_knots=TS_SPLINE_KNOTS), [TIMESTEP])],
                                   remainder='passthrough')
        steps.append(('ts_spline', spline))
    steps.append(('lr', LogisticRegression(C=C, max_iter=MAX_ITER)))
    return Pipeline(steps)


def _fit_xgb(params: dict, X_fit, y_fit, X_eval=None, y_eval=None) -> xgb.XGBClassifier:
    """Inputs as numpy: xgboost rejects feature names containing '<' (e.g. 'categorical_ivt_<90min')."""
    model = xgb.XGBClassifier(**{**XGB_FIXED, **params})
    if X_eval is None:
        return model.fit(X_fit, y_fit, verbose=False)
    return model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], early_stopping_rounds=XGB_EARLY_STOPPING,
                     verbose=False)


def _cv_lr(X: pd.DataFrame, y: np.ndarray, folds: list, task: Task, C: float) -> tuple:
    oof = np.full(len(y), np.nan)
    aurocs = []
    for train, val in folds:
        model = make_lr(C, task).fit(X.iloc[train], y[train])
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]
        aurocs.append(roc_auc_score(y[val], oof[val]))
    return np.median(aurocs), oof, {}


def _cv_xgb(X: pd.DataFrame, y: np.ndarray, folds: list, params: dict) -> tuple:
    oof = np.full(len(y), np.nan)
    aurocs, n_trees = [], []
    for train, val in folds:
        model = _fit_xgb(params, X.values[train], y[train], X.values[val], y[val])
        oof[val] = model.predict_proba(X.values[val], iteration_range=(0, model.best_iteration + 1))[:, 1]
        aurocs.append(roc_auc_score(y[val], oof[val]))
        n_trees.append(model.best_iteration + 1)
    return np.median(aurocs), oof, {'n_estimators': int(np.median(n_trees))}


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def train_model(model_type: Model, frame: pd.DataFrame, features: list, task: Task, folds: list) -> dict:
    """Grid search on the CV folds, refit on all training rows, Youden threshold from out-of-fold predictions."""
    X, y = frame[task_columns(features, task)], frame.y_true.values
    is_lr = model_type != Model.XGB_BASELINE

    grid = [{'C': C} for C in LR_C_GRID] if is_lr else XGB_GRID
    results = []
    for params in grid:
        score, oof, extra = _cv_lr(X, y, folds, task, params['C']) if is_lr else _cv_xgb(X, y, folds, params)
        results.append((score, params, oof, extra))
    score, params, oof, extra = max(results, key=lambda r: r[0])

    if is_lr:
        model = make_lr(params['C'], task).fit(X, y)
    else:
        model = _fit_xgb({**params, **extra}, X.values, y)
    print(f'{task.value}/{model_type.value}: {len(features)} features, params {params} {extra}, '
          f'median val AUROC {score:.4f}')
    return {'model': model, 'columns': list(X.columns), 'threshold': youden_threshold(y, oof),
            'params': {**params, **extra}, 'median_val_auroc': score}


def predict(fitted: dict, frame: pd.DataFrame) -> np.ndarray:
    X = frame[fitted['columns']]
    if isinstance(fitted['model'], xgb.XGBClassifier):
        X = X.values
    return fitted['model'].predict_proba(X)[:, 1]


def xgb_mean_abs_shap(fitted: dict, frame: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature (xgboost TreeSHAP, last column = bias)."""
    matrix = xgb.DMatrix(frame[fitted['columns']].values)
    contributions = fitted['model'].get_booster().predict(matrix, pred_contribs=True)[:, :-1]
    return pd.Series(np.abs(contributions).mean(axis=0), index=fitted['columns'], name='xgb_mean_abs_shap')


# ---------------------------------------------------------------------------
# XGB END reference
# ---------------------------------------------------------------------------

def xgb_end_patient_scores(hourly_preds: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    """Max hourly probability before the first END after hour 0 (all hours if no END)."""
    before_event = restrict_to_first_event(hourly_preds, predictable_events(y_df))
    scores = before_event.groupby('case_admission_id').y_prob.max()
    labels = patient_labels(hourly_preds)
    return pd.DataFrame({'y_prob': scores, 'y_true': labels}).rename_axis('case_admission_id').reset_index()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _variable_selection(train_patient: pd.DataFrame, candidates: list, folds: list, rng) -> tuple:
    X, y = train_patient[candidates], train_patient.y_true.values
    univariable = univariable_table(X, y)
    C = choose_l1_C(X, y, folds)
    frequency = stability_selection(X, y, C, rng)
    selected = select_final_variables(frequency)
    print(f'L1 C = {C}; selected {len(selected)} variables; '
          f'events per variable = {y.sum() / len(selected):.1f}')
    if y.sum() / len(selected) < MIN_EVENTS_PER_VARIABLE:
        print(f'WARNING: fewer than {MIN_EVENTS_PER_VARIABLE} events per variable')

    table = pd.DataFrame({'feature': candidates}) \
        .merge(univariable, on='feature', how='left') \
        .merge(frequency.rename_axis('feature').reset_index(), on='feature', how='left')
    table['in_literature'] = table.feature.isin(literature_features())
    table['selected'] = table.feature.isin(selected)
    return table, selected, C


def run(splits_path: str, test_data_path: str, external_data_path: str, gsu_features_path: str,
        mimic_features_path: str, model_dir: str, threshold_path: str, output_dir: str):
    ensure_dir(output_dir)
    rng = np.random.default_rng(SEED)

    # -- data: same patients and folds as the XGB END model
    X_train, y_train, fold_val_cids = load_training(splits_path)
    cohort_data = {Cohort.INTERNAL_TEST: ch.load(test_data_path, weights_only=False),
                   Cohort.EXTERNAL_TEST: ch.load(external_data_path, weights_only=False)}
    n_timepoints = X_train.shape[1]

    gsu_imaging = load_admission_imaging(gsu_features_path)
    imaging = {Cohort.TRAINING: gsu_imaging, Cohort.INTERNAL_TEST: gsu_imaging,
               Cohort.EXTERNAL_TEST: load_admission_imaging(mimic_features_path)}
    admission = {Cohort.TRAINING: extract_admission_features(X_train, gsu_imaging)}
    for cohort, (X, _) in cohort_data.items():
        admission[cohort] = extract_admission_features(X, imaging[cohort])
    for cohort, df in admission.items():
        print(f'{cohort.value}: {len(df)} patients, imaging after hour 0: {count_late_imaging(imaging[cohort], df.index)}')

    rescaling = fit_binary_rescaling(admission[Cohort.TRAINING])
    admission = {c: apply_binary_rescaling(df, rescaling) for c, df in admission.items()}
    candidates = [c for c in admission[Cohort.TRAINING].columns if admission[Cohort.TRAINING][c].nunique() > 1]

    # -- labels / XGB END predictions
    hourly = {Cohort.TRAINING: hourly_labels(admission[Cohort.TRAINING].index, y_train, n_timepoints)}
    xgb_end = {}
    for cohort, (X, y_df) in cohort_data.items():
        preds = predict_hourly(X, y_df, model_dir)
        expected = hourly_labels(admission[cohort].index, y_df, n_timepoints)
        for column in ['case_admission_id', TIMESTEP, 'y_true']:
            assert np.array_equal(preds[column].values, expected[column].values), f'{cohort.value}: {column} mismatch'
        preds['y_true'] = preds.y_true.astype(int)
        hourly[cohort] = preds
        xgb_end[cohort] = {Task.HOURLY: preds, Task.PATIENT: xgb_end_patient_scores(preds, y_df)}

    frames = {(c, t): task_frame(admission[c], hourly[c], t) for c in hourly for t in Task}
    folds = {t: fold_indices(frames[(Cohort.TRAINING, t)].case_admission_id.values, fold_val_cids) for t in Task}

    # -- variable selection (patient level, training only)
    selection, selected, _ = _variable_selection(frames[(Cohort.TRAINING, Task.PATIENT)], candidates,
                                                 folds[Task.PATIENT], rng)
    feature_sets = {Model.LR_LITERATURE: literature_features(),
                    Model.LR_SELECTED: selected,
                    Model.LR_SELECTED_NO_IMAGING: [f for f in selected if f not in IMAGING_LABELS],
                    Model.XGB_BASELINE: candidates}

    # -- train baselines
    fitted = {(t, m): train_model(m, frames[(Cohort.TRAINING, t)], feature_sets[m], t, folds[t])
              for t in Task for m in BASELINE_MODELS}

    shap = xgb_mean_abs_shap(fitted[(Task.PATIENT, Model.XGB_BASELINE)], frames[(Cohort.TRAINING, Task.PATIENT)])
    selection = selection.merge(shap.rename_axis('feature').reset_index(), on='feature', how='left')
    selection.sort_values('selection_frequency', ascending=False) \
        .to_csv(os.path.join(output_dir, 'variable_selection.csv'), index=False)

    train_patient = frames[(Cohort.TRAINING, Task.PATIENT)]
    wald_table(fit_unpenalised_lr(train_patient[selected], train_patient.y_true.values), train_patient[selected]) \
        .to_csv(os.path.join(output_dir, 'lr_coefficients.csv'), index=False)

    pd.DataFrame([{'task': t.value, 'model': m.value, 'n_features': len(feature_sets[m]),
                   'threshold_youden_j': f['threshold'], 'median_val_auroc': f['median_val_auroc'], **f['params']}
                  for (t, m), f in fitted.items()]) \
        .to_csv(os.path.join(output_dir, 'baseline_model_config.csv'), index=False)

    # -- evaluate + compare
    xgb_end_threshold = load_thresholds(threshold_path)[PRIMARY_THRESHOLD]
    performance, comparison, predictions = [], [], {t: [] for t in Task}
    for cohort in TEST_COHORTS:
        for task in Task:
            keys = ['case_admission_id', TIMESTEP] if task == Task.HOURLY else ['case_admission_id']
            reference = xgb_end[cohort][task][keys + ['y_prob']].rename(columns={'y_prob': Model.XGB_END.value})
            frame = frames[(cohort, task)].merge(reference, on=keys, how='left', validate='one_to_one')
            assert frame[Model.XGB_END.value].notna().all(), f'{cohort.value}/{task.value}: missing XGB END rows'
            thresholds = {Model.XGB_END: xgb_end_threshold}
            for model in BASELINE_MODELS:
                frame[model.value] = predict(fitted[(task, model)], frame)
                thresholds[model] = fitted[(task, model)]['threshold']

            for model, threshold in thresholds.items():
                scored = frame[['case_admission_id', 'y_true']].assign(y_prob=frame[model.value])
                metrics = _metrics_with_ci(scored, {PRIMARY_THRESHOLD: threshold}, np.random.default_rng(SEED))
                performance.append(metrics.assign(cohort=cohort.value, task=task.value, model=model.value,
                                                  **_counts(scored)))
                print(f'{cohort.value}/{task.value}/{model.value}: '
                      f'AUROC {roc_auc_score(scored.y_true, scored.y_prob):.4f}')

            for model in BASELINE_MODELS:
                tests = [paired_cluster_bootstrap(frame, Model.XGB_END.value, model.value,
                                                  np.random.default_rng(SEED))]
                if task == Task.PATIENT:
                    tests.append(delong_test(frame, Model.XGB_END.value, model.value))
                comparison.append(pd.concat(tests).assign(cohort=cohort.value, task=task.value,
                                                          model_a=Model.XGB_END.value, model_b=model.value))

            keep = keys + ['y_true', Model.XGB_END.value] + [m.value for m in BASELINE_MODELS]
            predictions[task].append(frame[keep].assign(cohort=cohort.value))

    performance = pd.concat(performance, ignore_index=True)
    leading = ['cohort', 'task', 'model', 'threshold_name', 'threshold', 'metric', 'value', 'ci_low', 'ci_high']
    performance[leading + [c for c in performance.columns if c not in leading]] \
        .to_csv(os.path.join(output_dir, 'baseline_performance.csv'), index=False)

    comparison = pd.concat(comparison, ignore_index=True)
    leading = ['cohort', 'task', 'model_a', 'model_b', 'metric', 'test']
    comparison[leading + [c for c in comparison.columns if c not in leading]] \
        .to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    for task, frames_ in predictions.items():
        pd.concat(frames_).to_csv(os.path.join(output_dir, f'{task.value}_predictions.csv'), index=False)
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--splits_path', required=True, help='train_data_splits_*.pth')
    parser.add_argument('-d', '--test_data_path', required=True, help='internal test .pth')
    parser.add_argument('-e', '--external_data_path', required=True, help='mimic_test_data.pth')
    parser.add_argument('-g', '--gsu_features_path', required=True, help='GSU preprocessed_features_*.csv')
    parser.add_argument('-x', '--mimic_features_path', required=True, help='MIMIC preprocessed_features_*.csv')
    parser.add_argument('-m', '--model_dir', required=True, help='XGB END final model directory')
    parser.add_argument('-t', '--threshold_path', required=True, help='threshold_tuning_results_*.json')
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()
    run(args.splits_path, args.test_data_path, args.external_data_path, args.gsu_features_path,
        args.mimic_features_path, args.model_dir, args.threshold_path, args.output_dir)
