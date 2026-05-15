import os
import argparse
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch as ch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import StandardScaler

from prediction.utils.scoring import precision, recall, specificity
from prediction.utils.utils import ensure_dir


def raw_features_and_labels(timeseries, y_df, target_time_to_outcome=6, target_interval=True):
    """Extract raw feature values and labels for each patient.

    Unlike aggregate_and_label_timeseries(), this returns only the raw feature
    values at each timepoint (103 features) without any temporal aggregation
    (no cumulative mean/min/max/std/rate-of-change/timestep).

    Args:
        timeseries: array of shape (n_patients, n_timepoints, n_features, n_channels)
        y_df: DataFrame with columns [case_admission_id, relative_sample_date_hourly_cat]
        target_time_to_outcome: prediction window in hours (default: 6)
        target_interval: if True, label=1 when event occurs within window;
                         if False, label=1 only at exact timepoint

    Returns:
        all_subj_data: list of arrays, each (n_timepoints, n_features)
        all_subj_labels: list of arrays, each (n_timepoints,)
    """
    all_subj_labels = []
    all_subj_data = []
    n_timepoints = timeseries.shape[1]

    for idx, cid in enumerate(timeseries[:, 0, 0, 0]):
        # Raw features at last channel: shape (n_timepoints, n_features)
        x_raw = timeseries[idx, :, :, -1].astype('float32')

        if cid not in y_df.case_admission_id.values:
            labels = np.zeros(n_timepoints)
        else:
            target_events_ts = y_df[y_df.case_admission_id == cid].relative_sample_date_hourly_cat.values

            labels = []
            for ts in range(int(n_timepoints)):
                if target_interval:
                    if np.any((target_events_ts > ts) & (target_events_ts <= ts + target_time_to_outcome)):
                        labels.append(1)
                    else:
                        labels.append(0)
                else:
                    if np.any(target_events_ts == ts + target_time_to_outcome):
                        labels.append(1)
                    else:
                        labels.append(0)
            labels = np.array(labels)

        all_subj_data.append(x_raw)
        all_subj_labels.append(labels)

    return all_subj_data, all_subj_labels


def prepare_raw_features_dataset(scenario, target_time_to_outcome=6, target_interval=True):
    """Prepare raw-feature train/val arrays from a single CV fold.

    Args:
        scenario: tuple (X_train, X_val, y_train, y_val)
        target_time_to_outcome: prediction window in hours
        target_interval: interval-based labeling

    Returns:
        train_data, val_data: 2D arrays (n_samples, 103)
        train_labels, val_labels: 1D arrays
    """
    X_train, X_val, y_train, y_val = scenario

    train_data, train_labels = raw_features_and_labels(
        X_train, y_train, target_time_to_outcome, target_interval)
    val_data, val_labels = raw_features_and_labels(
        X_val, y_val, target_time_to_outcome, target_interval)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    val_data = np.concatenate(val_data)
    val_labels = np.concatenate(val_labels)

    # Scale features
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)

    # Replace NaN with 0 (LR cannot handle NaN unlike XGBoost)
    train_data = np.nan_to_num(train_data, nan=0.0)
    val_data = np.nan_to_num(val_data, nan=0.0)

    return train_data, val_data, train_labels, val_labels, scaler


def evaluate_logistic_regression(data_splits_path: str, output_dir: str):
    """Run 5-fold CV evaluation of logistic regression on raw features.

    Args:
        data_splits_path: path to .pth file with CV data splits
        output_dir: directory to save results
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    splits = ch.load(data_splits_path)
    outcome = '_'.join(os.path.basename(data_splits_path).split('_')[3:6])

    model_df = pd.DataFrame()
    val_aurocs = []
    val_auprcs = []

    for i, scenario in enumerate(splits):
        print(f"\n--- Fold {i} ---")
        train_data, val_data, train_labels, val_labels, _ = prepare_raw_features_dataset(
            scenario, target_time_to_outcome=6, target_interval=True)

        n_features = train_data.shape[1]
        pos_rate = train_labels.sum() / len(train_labels)
        print(f"  Train: {len(train_labels)} samples, {int(train_labels.sum())} positive ({pos_rate*100:.2f}%)")
        print(f"  Val:   {len(val_labels)} samples, {int(val_labels.sum())} positive ({val_labels.sum()/len(val_labels)*100:.2f}%)")
        print(f"  Features: {n_features}")

        lr = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced',
            solver='lbfgs', random_state=42)
        lr.fit(train_data, train_labels)

        # Validation metrics
        model_y_val = lr.predict_proba(val_data)[:, 1].astype('float32')
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0).astype('float32')
        model_auc_val = roc_auc_score(val_labels, model_y_val)
        model_auprc_val = average_precision_score(val_labels, model_y_val)
        model_mcc_val = matthews_corrcoef(val_labels, model_y_pred_val)
        model_acc_val = accuracy_score(val_labels, model_y_pred_val)
        model_precision_val = precision(val_labels, model_y_pred_val.astype(float)).numpy()
        model_sn_val = recall(val_labels, model_y_pred_val).numpy()
        model_sp_val = specificity(val_labels, model_y_pred_val).numpy()

        # Training metrics
        model_y_train = lr.predict_proba(train_data)[:, 1].astype('float32')
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0).astype('float32')
        model_auc_train = roc_auc_score(train_labels, model_y_train)
        model_auprc_train = average_precision_score(train_labels, model_y_train)
        model_mcc_train = matthews_corrcoef(train_labels, model_y_pred_train)
        model_acc_train = accuracy_score(train_labels, model_y_pred_train)
        model_precision_train = precision(train_labels, model_y_pred_train.astype(float)).numpy()
        model_sn_train = recall(train_labels, model_y_pred_train).numpy()
        model_sp_train = specificity(train_labels, model_y_pred_train).numpy()

        print(f"  Val AUROC: {model_auc_val:.4f}, AUPRC: {model_auprc_val:.4f}, MCC: {model_mcc_val:.4f}")

        run_df = pd.DataFrame(index=[0])
        run_df['CV'] = i
        run_df['n_features'] = n_features
        run_df['outcome'] = outcome
        run_df['auc_train'] = model_auc_train
        run_df['auc_val'] = model_auc_val
        run_df['auprc_train'] = model_auprc_train
        run_df['auprc_val'] = model_auprc_val
        run_df['mcc_train'] = model_mcc_train
        run_df['mcc_val'] = model_mcc_val
        run_df['acc_train'] = model_acc_train
        run_df['acc_val'] = model_acc_val
        run_df['precision_train'] = model_precision_train
        run_df['precision_val'] = model_precision_val
        run_df['sn_train'] = model_sn_train
        run_df['sn_val'] = model_sn_val
        run_df['sp_train'] = model_sp_train
        run_df['sp_val'] = model_sp_val
        model_df = pd.concat([model_df, run_df])

        val_aurocs.append(model_auc_val)
        val_auprcs.append(model_auprc_val)

    # Save per-fold results
    csv_path = os.path.join(output_dir, f'lr_baseline_{timestamp}.csv')
    model_df.to_csv(csv_path, index=False)
    print(f"\nPer-fold results saved to {csv_path}")

    # Save summary
    summary = {
        'model': 'LogisticRegression',
        'C': 1.0,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'n_features': int(n_features),
        'feature_type': 'raw_values_only',
        'n_folds': len(splits),
        'median_val_auroc': float(np.median(val_aurocs)),
        'median_val_auprc': float(np.median(val_auprcs)),
        'per_fold_auroc': [float(v) for v in val_aurocs],
        'per_fold_auprc': [float(v) for v in val_auprcs],
        'outcome': outcome,
        'data_splits_path': data_splits_path,
        'timestamp': timestamp,
    }
    summary_path = os.path.join(output_dir, f'lr_baseline_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    print(f"\n=== Logistic Regression Baseline Results ===")
    print(f"  Median AUROC: {np.median(val_aurocs):.4f}")
    print(f"  Median AUPRC: {np.median(val_auprcs):.4f}")
    print(f"  Features: {n_features} (raw values only, no temporal aggregation)")

    return summary


def retrain_on_all_data(data_splits_path: str, output_dir: str):
    """Retrain logistic regression on all available CV data and save the final model.

    In k-fold CV, fold 0's train + val = all data. This function combines
    them, trains a single model on the full dataset, and saves:
      - lr_final_model.pkl  (pickled LogisticRegression)
      - scaler.pkl          (fitted StandardScaler)
      - final_model_config.json (metadata)

    Args:
        data_splits_path: path to .pth file with CV data splits
        output_dir: directory to save model artifacts
    """
    splits = ch.load(data_splits_path)

    # Fold 0's train + val = all data
    X_train, X_val, y_train, y_val = splits[0]
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val])

    all_data, all_labels = raw_features_and_labels(
        X_all, y_all, target_time_to_outcome=6, target_interval=True)

    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)

    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    all_data = np.nan_to_num(all_data, nan=0.0)

    n_features = all_data.shape[1]
    n_samples = all_data.shape[0]
    n_positive = int(all_labels.sum())
    print(f"Training on {n_samples} samples ({n_positive} positive, "
          f"{n_positive/n_samples*100:.2f}%), {n_features} features")

    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=42)
    lr.fit(all_data, all_labels)

    # Save artifacts
    ensure_dir(output_dir)

    model_path = os.path.join(output_dir, 'lr_final_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(lr, f)
    print(f"Model saved to {model_path}")

    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    config = {
        'model': 'LogisticRegression',
        'C': 1.0,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'n_features': n_features,
        'feature_type': 'raw_values_only',
        'n_training_samples': n_samples,
        'n_positive_samples': n_positive,
        'positive_rate': n_positive / n_samples,
        'data_splits_path': data_splits_path,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    config_path = os.path.join(output_dir, 'final_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    return lr, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate logistic regression baseline on raw features for END prediction')
    parser.add_argument('-d', '--data_splits_path', type=str, required=True,
                        help='Path to .pth file with CV data splits')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--retrain', action='store_true',
                        help='Also retrain on all data after CV evaluation')
    args = parser.parse_args()

    summary = evaluate_logistic_regression(args.data_splits_path, args.output_dir)

    if args.retrain:
        retrain_dir = os.path.join(args.output_dir, 'final_model')
        print(f"\n--- Retraining on all data ---")
        retrain_on_all_data(args.data_splits_path, retrain_dir)
