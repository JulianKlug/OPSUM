"""
Out-of-fold XGB scores and SHAP contributions for every (patient, hour) row.

Each fold's model is trained like the final model (fixed config, no early
stopping) and applied to its validation patients only, so every training
patient gets a score from a model that never saw them.

    splits.pth ──> fold i: train XGB ──> validation rows: score + SHAP
                                              │
                    oof_rows.csv  <───────────┤  one line per (patient, hour)
                    fold_i_features.npy  <────┤  unscaled aggregated features
                    fold_i_shap.npy  <────────┘  log-odds contributions
"""
import argparse
import json
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch as ch
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from prediction.short_term_outcome_prediction.gridsearch_aggregate_xgb import focal_loss_objective
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.utils import ensure_dir

PREDICTION_HORIZON_HOURS = 6
PATIENT_ID_SEPARATOR = '_'

ROWS_FILE = 'oof_rows.csv'
RAW_FEATURE_NAMES_FILE = 'raw_feature_names.json'
FEATURES_FILE_TEMPLATE = 'fold_{fold}_features.npy'
SHAP_FILE_TEMPLATE = 'fold_{fold}_shap.npy'

# Indices into the last axis of the splits array: (case id, hour, feature name, value)
CASE_ID_FIELD = 0
FEATURE_NAME_FIELD = 2


def build_xgb_params(config: dict) -> dict:
    """Translate a final-model config into XGBClassifier arguments (same mapping as retrain_on_all_data)."""
    xgb_params = dict(
        learning_rate=config['learning_rate'],
        max_depth=int(config['max_depth']),
        n_estimators=int(config['n_estimators']),
        reg_lambda=config['reg_lambda'],
        reg_alpha=config.get('alpha', config.get('reg_alpha', 1)),
        min_child_weight=config['min_child_weight'],
        subsample=config['subsample'],
        colsample_bytree=config['colsample_bytree'],
        colsample_bylevel=config.get('colsample_bylevel', 1.0),
        booster=config.get('booster', 'dart'),
        grow_policy=config.get('grow_policy', 'lossguide'),
        gamma=config.get('gamma', 0.5),
        max_delta_step=config.get('max_delta_step', 0),
    )

    focal_gamma = config.get('focal_gamma', 0)
    scale_pos_weight = config.get('scale_pos_weight', 10)

    if focal_gamma <= 0:
        xgb_params['scale_pos_weight'] = scale_pos_weight
        return xgb_params

    # XGBoost ignores scale_pos_weight for custom objectives: the weight goes into the loss itself
    xgb_params['objective'] = partial(focal_loss_objective, gamma_fl=focal_gamma, pos_weight=scale_pos_weight)
    xgb_params['scale_pos_weight'] = 1

    return xgb_params


def aggregate_rows(X: np.ndarray, y_df: pd.DataFrame, config: dict):
    """
    Flatten patients x hours into rows of aggregated features with their 6h-ahead labels.

    Rows are patient-major: row r belongs to patient r // n_hours at hour r % n_hours.
    """
    data, labels = aggregate_and_label_timeseries(
        X, y_df,
        target_time_to_outcome=PREDICTION_HORIZON_HOURS,
        target_interval=config.get('target_interval', 1),
        restrict_to_first_event=config.get('restrict_to_first_event', 0),
        add_lag_features=config.get('add_lag_features', False),
        add_rolling_features=config.get('add_rolling_features', False),
    )

    return np.concatenate(data).astype('float32'), np.concatenate(labels).astype('int8')


def build_row_index(X: np.ndarray, fold: int) -> pd.DataFrame:
    """
    Identify every row of the aggregated matrix.

    e.g. 2 patients x 3 hours -> case ids [a, a, a, b, b, b], hours [0, 1, 2, 0, 1, 2]
    """
    case_admission_ids = X[:, 0, 0, CASE_ID_FIELD].astype(str)
    n_hours = X.shape[1]

    row_index = pd.DataFrame({
        'case_admission_id': np.repeat(case_admission_ids, n_hours),
        'hour': np.tile(np.arange(n_hours), len(case_admission_ids)),
        'fold': fold,
    })

    # Folds were split on patient id, the part of the case id before the separator: '1002417_9090' -> '1002417'
    row_index['patient_id'] = row_index['case_admission_id'].str.split(PATIENT_ID_SEPARATOR).str[0]

    return row_index


def compute_fold_predictions(fold_split, config: dict, fold: int):
    """Train on the fold's training patients; score and explain its validation patients."""
    X_train, X_val, y_train_df, y_val_df = fold_split

    train_features, train_labels = aggregate_rows(X_train, y_train_df, config)
    val_features, val_labels = aggregate_rows(X_val, y_val_df, config)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)

    # No eval set and no early stopping: the deployed final model was trained this way
    model = xgb.XGBClassifier(**build_xgb_params(config))
    model.fit(train_scaled, train_labels, verbose=False)

    scores = model.predict_proba(val_scaled)[:, 1].astype('float32')

    # Native SHAP in log-odds space; the last column is the base value
    shap_values = model.get_booster().predict(xgb.DMatrix(val_scaled), pred_contribs=True).astype('float32')

    rows = build_row_index(X_val, fold)
    rows['label'] = val_labels
    rows['xgb_score'] = scores

    return rows, val_features, shap_values


def compute_out_of_fold_predictions(data_splits_path: str, config: dict, output_dir: str) -> pd.DataFrame:
    """Run every fold and save the row table, the unscaled features and the SHAP values."""
    ensure_dir(output_dir)
    splits = ch.load(data_splits_path)

    raw_feature_names = [str(name) for name in splits[0][0][0, 0, :, FEATURE_NAME_FIELD]]
    with open(os.path.join(output_dir, RAW_FEATURE_NAMES_FILE), 'w') as f:
        json.dump(raw_feature_names, f)

    all_rows = []
    for fold, fold_split in enumerate(splits):
        start = time.time()
        rows, features, shap_values = compute_fold_predictions(fold_split, config, fold)

        np.save(os.path.join(output_dir, FEATURES_FILE_TEMPLATE.format(fold=fold)), features)
        np.save(os.path.join(output_dir, SHAP_FILE_TEMPLATE.format(fold=fold)), shap_values)
        all_rows.append(rows)

        print(f'Fold {fold}: {len(rows)} rows, {int(rows.label.sum())} positive, '
              f'AUROC {roc_auc_score(rows.label, rows.xgb_score):.4f}, '
              f'AUPRC {average_precision_score(rows.label, rows.xgb_score):.4f}, '
              f'{time.time() - start:.0f}s', flush=True)

    all_rows = pd.concat(all_rows, ignore_index=True)
    all_rows.to_csv(os.path.join(output_dir, ROWS_FILE), index=False)

    print(f'Pooled: AUROC {roc_auc_score(all_rows.label, all_rows.xgb_score):.4f}, '
          f'AUPRC {average_precision_score(all_rows.label, all_rows.xgb_score):.4f}')

    return all_rows


class OutOfFoldStore:
    """
    Read access to the saved out-of-fold results, addressed by row id (line number in the row table).

    Hides the per-fold file layout:

        row id ──> fold, position in fold ──> features / SHAP of that row
                                          └─> the same patient's earlier hours
    """

    def __init__(self, output_dir: str):
        self.rows = pd.read_csv(os.path.join(output_dir, ROWS_FILE), dtype={'case_admission_id': str, 'patient_id': str})

        with open(os.path.join(output_dir, RAW_FEATURE_NAMES_FILE)) as f:
            self.raw_feature_names = json.load(f)

        folds = sorted(self.rows.fold.unique())
        self._features = {fold: self._open(output_dir, FEATURES_FILE_TEMPLATE, fold) for fold in folds}
        self._shap = {fold: self._open(output_dir, SHAP_FILE_TEMPLATE, fold) for fold in folds}

        # Folds were written one after the other: the first row id of a fold is its offset
        self._fold_offset = self.rows.reset_index().groupby('fold')['index'].min().to_dict()

    @staticmethod
    def _open(output_dir: str, file_template: str, fold: int) -> np.ndarray:
        # Memory-mapped: only the requested rows are read from disk
        return np.load(os.path.join(output_dir, file_template.format(fold=fold)), mmap_mode='r')

    def _positions_by_fold(self, row_ids: np.ndarray):
        folds = self.rows.fold.values[row_ids]

        for fold in np.unique(folds):
            in_fold = np.flatnonzero(folds == fold)

            yield fold, in_fold, row_ids[in_fold] - self._fold_offset[fold]

    def _gather(self, matrices: dict, row_ids: np.ndarray) -> np.ndarray:
        row_ids = np.asarray(row_ids)
        n_columns = next(iter(matrices.values())).shape[1]
        gathered = np.empty((len(row_ids), n_columns), dtype='float32')

        for fold, in_fold, positions in self._positions_by_fold(row_ids):
            gathered[in_fold] = matrices[fold][positions]

        return gathered

    def features(self, row_ids) -> np.ndarray:
        """Unscaled aggregated features of the given rows."""
        return self._gather(self._features, row_ids)

    def shap_values(self, row_ids) -> np.ndarray:
        """SHAP contributions of the given rows; the last column is the base value."""
        return self._gather(self._shap, row_ids)

    def history_row_ids(self, row_id: int) -> np.ndarray:
        """
        Row ids of the same patient from admission up to and including this row's hour.

        e.g. row 150 is hour 6 of its patient -> [144, ..., 150]. Later hours are never returned.
        """
        hour = int(self.rows.hour.values[row_id])

        return np.arange(row_id - hour, row_id + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--data_splits_path', type=str, required=True)
    parser.add_argument('-c', '--config_path', type=str, required=True, help='final_model_config.json')
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        final_model_config = json.load(f)

    compute_out_of_fold_predictions(args.data_splits_path, final_model_config, args.output_dir)
