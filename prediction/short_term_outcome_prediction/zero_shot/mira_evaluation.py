"""
Evaluation module for MIRA zero-shot prediction of early neurological deterioration.

This module evaluates MIRA's forecasting ability for predicting neurological
deterioration based on NIHSS score changes.

Classification logic:
- Forecast max_NIHSS at timestep + 6
- Compute delta_NIHSS = max_NIHSS_forecast - min_NIHSS_historical
- Predict deterioration if delta_NIHSS >= 4
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    accuracy_score
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Tuple, List, Optional

def ensure_dir(directory: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

# Local imports
from mira_data_loader import (
    load_data_splits,
    get_feature_indices,
    get_scaler,
    apply_scaler,
    load_normalisation_parameters,
    load_outcome_data,
    get_validation_patient_ids
)
from mira_inference import (
    load_mira_model,
    forecast_all_timesteps
)


def reverse_normalisation(
    data: np.ndarray,
    variable_name: str,
    normalisation_parameters_df: pd.DataFrame
) -> np.ndarray:
    """
    Reverse normalisation of the data using original mean and std.

    Args:
        data: The data to reverse normalise
        variable_name: The name of the variable to reverse normalise
        normalisation_parameters_df: DataFrame with normalisation parameters

    Returns:
        Reverse normalised data
    """
    row = normalisation_parameters_df[normalisation_parameters_df.variable == variable_name]
    if len(row) == 0:
        raise ValueError(f"Variable '{variable_name}' not found in normalisation parameters")

    std = row.original_std.iloc[0]
    mean = row.original_mean.iloc[0]
    data = (data * std) + mean
    return data


def evaluate_fold(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: pd.DataFrame,
    outcome_df: pd.DataFrame,
    normalisation_parameters_df: pd.DataFrame,
    n_time_steps: int = 72,
    eval_n_time_steps_before_event: int = 6,
    batch_size: int = 64,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """
    Evaluate MIRA model on a single cross-validation fold.

    Args:
        model: MIRA model
        X_train: Training data for scaler fitting
        X_val: Validation data
        y_val: Validation labels DataFrame
        outcome_df: Full outcome labels DataFrame
        normalisation_parameters_df: Normalisation parameters
        n_time_steps: Number of timesteps
        eval_n_time_steps_before_event: Forecast horizon (6 hours)
        batch_size: Batch size for inference
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (overall_prediction_df, metrics_per_timestep_df, per_timestep_metrics)
    """
    # Get feature indices
    min_nihss_idx, max_nihss_idx = get_feature_indices(X_train)

    # Fit scaler on training data and apply to validation
    scaler = get_scaler(X_train)
    X_val_scaled = apply_scaler(X_val, scaler)

    # Get validation patient IDs
    val_patient_cids = get_validation_patient_ids(X_val)

    # Get raw min_NIHSS values for historical comparison (before scaling)
    raw_min_nihss = X_val[:, :, min_nihss_idx, -1].astype('float32')

    # Forecast max_NIHSS for all timesteps
    print("Forecasting max_NIHSS values...")
    all_forecasts = forecast_all_timesteps(
        model,
        X_val_scaled,
        max_nihss_idx,
        n_time_steps=n_time_steps,
        n_forecast_steps=eval_n_time_steps_before_event,
        batch_size=batch_size,
        show_progress=show_progress
    )

    # Evaluate predictions
    roc_scores = []
    auprc_scores = []
    mcc_scores = []
    accuracy_scores_list = []
    n_pos_samples = []
    timesteps = []

    overall_prediction_df = pd.DataFrame(columns=['timestep', 'prediction', 'true_label'])

    timestep_range = range(n_time_steps)
    if show_progress:
        timestep_range = tqdm(timestep_range, desc="Evaluating timesteps")

    for ts in timestep_range:
        evaluated_ts = ts + eval_n_time_steps_before_event

        # Ground truth at evaluated timestep
        outcome_at_evaluated_ts_df = outcome_df[outcome_df['relative_sample_date_hourly_cat'] == evaluated_ts]
        y_true_at_evaluated_ts = np.isin(val_patient_cids, outcome_at_evaluated_ts_df['case_admission_id'].values).astype(np.int32)

        # Get forecasts at this timestep
        forecasts_at_ts = all_forecasts[ts]  # [n_cases, n_forecast_steps]

        # Get historical min_NIHSS up to current timestep (use scaled values for consistency)
        scaled_min_nihss = X_val_scaled[:, :ts + 1, min_nihss_idx]
        min_nihss_up_to_current_ts = np.min(scaled_min_nihss, axis=1)  # [n_cases]

        # Get max_NIHSS forecast at the last forecast timestep (ts + eval_n_time_steps_before_event)
        # The forecasts are scaled, so we need to reverse the scaler transform
        max_nihss_forecast_scaled = forecasts_at_ts[:, -1]  # [n_cases]

        # Reverse StandardScaler transform for max_NIHSS
        # Create dummy array with same shape as original features
        dummy_array = np.zeros((len(max_nihss_forecast_scaled), X_val_scaled.shape[2]))
        dummy_array[:, max_nihss_idx] = max_nihss_forecast_scaled
        reverse_scaled = scaler.inverse_transform(dummy_array)
        max_nihss_forecast_reverse_scaled = reverse_scaled[:, max_nihss_idx]

        # Reverse StandardScaler transform for min_NIHSS
        dummy_array[:, min_nihss_idx] = min_nihss_up_to_current_ts
        dummy_array[:, max_nihss_idx] = 0  # Reset
        reverse_scaled_min = scaler.inverse_transform(dummy_array)
        min_nihss_reverse_scaled = reverse_scaled_min[:, min_nihss_idx]

        # Reverse normalisation (original data preprocessing normalisation)
        max_nihss_actual = reverse_normalisation(
            max_nihss_forecast_reverse_scaled,
            'max_NIHSS',
            normalisation_parameters_df
        )
        min_nihss_actual = reverse_normalisation(
            min_nihss_reverse_scaled,
            'min_NIHSS',
            normalisation_parameters_df
        )

        # Compute delta NIHSS
        delta_nihss = max_nihss_actual - min_nihss_actual

        # Classification
        y_pred = delta_nihss
        y_pred_binary = (delta_nihss >= 4).astype(int)

        # Store predictions
        timestep_df = pd.DataFrame({
            'timestep': [ts] * len(y_true_at_evaluated_ts),
            'prediction': y_pred,
            'true_label': y_true_at_evaluated_ts
        })
        overall_prediction_df = pd.concat([overall_prediction_df, timestep_df])

        # Compute metrics
        timesteps.append(ts)
        n_pos_samples.append(np.sum(y_true_at_evaluated_ts))
        accuracy_scores_list.append(accuracy_score(y_true_at_evaluated_ts, y_pred_binary))

        if len(np.unique(y_true_at_evaluated_ts)) == 1:
            roc_scores.append(np.nan)
            auprc_scores.append(np.nan)
            mcc_scores.append(np.nan)
        else:
            roc_scores.append(roc_auc_score(y_true_at_evaluated_ts, y_pred))
            auprc_scores.append(average_precision_score(y_true_at_evaluated_ts, y_pred))
            mcc_scores.append(matthews_corrcoef(y_true_at_evaluated_ts, y_pred_binary))

    # Ensure types
    overall_prediction_df['true_label'] = overall_prediction_df['true_label'].astype(int)
    overall_prediction_df['prediction'] = overall_prediction_df['prediction'].astype(float)

    # Create metrics DataFrame
    metrics_per_timestep_df = pd.DataFrame({
        'timestep': timesteps,
        'roc': roc_scores,
        'auprc': auprc_scores,
        'mcc': mcc_scores,
        'accuracy': accuracy_scores_list,
        'n_pos_samples': n_pos_samples
    })

    return overall_prediction_df, metrics_per_timestep_df, [roc_scores, auprc_scores, mcc_scores, accuracy_scores_list]


def mira_validation_evaluation(
    data_path: str,
    normalisation_data_path: str,
    outcome_data_path: str,
    output_path: str = None,
    use_gpu: bool = True,
    n_time_steps: int = 72,
    eval_n_time_steps_before_event: int = 6,
    batch_size: int = 64,
    use_cross_validation: bool = True,
    model_name: str = "MIRA-Mode/MIRA"
) -> pd.DataFrame:
    """
    Evaluate MIRA model on validation set for early neurological deterioration prediction.

    Args:
        data_path: Path to data splits .pth file
        normalisation_data_path: Path to normalisation parameters CSV
        outcome_data_path: Path to outcome labels CSV
        output_path: Path to save results (optional)
        use_gpu: Whether to use GPU
        n_time_steps: Number of timesteps
        eval_n_time_steps_before_event: Forecast horizon (6 hours)
        batch_size: Batch size for inference
        use_cross_validation: Whether to evaluate all folds
        model_name: MIRA model name on HuggingFace

    Returns:
        DataFrame with overall results
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(data_path),
            f'mira_validation_evaluation_results_{eval_n_time_steps_before_event}h'
        )
    ensure_dir(output_path)

    # Load data
    print("Loading data...")
    splits = load_data_splits(data_path)
    normalisation_parameters_df = load_normalisation_parameters(normalisation_data_path)
    outcome_df = load_outcome_data(outcome_data_path)

    # Load MIRA model
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Loading MIRA model on {device}...")
    model = load_mira_model(model_name, device=device)

    # Determine which folds to evaluate
    if use_cross_validation:
        fold_range = range(len(splits))
    else:
        fold_range = [0]  # Just the first fold

    all_folds_results = pd.DataFrame()

    for cv_fold in fold_range:
        print(f"\n{'='*50}")
        print(f"Evaluating fold {cv_fold + 1}/{len(splits)}")
        print(f"{'='*50}")

        fold_result_dir = os.path.join(output_path, f'cv_fold_{cv_fold}')
        ensure_dir(fold_result_dir)

        X_train, X_val, y_train, y_val = splits[cv_fold]

        # Evaluate fold
        overall_prediction_df, metrics_per_timestep_df, _ = evaluate_fold(
            model,
            X_train,
            X_val,
            y_val,
            outcome_df,
            normalisation_parameters_df,
            n_time_steps=n_time_steps,
            eval_n_time_steps_before_event=eval_n_time_steps_before_event,
            batch_size=batch_size,
            show_progress=True
        )

        # Compute overall metrics for this fold
        overall_results_df = pd.DataFrame({
            'overall_roc': roc_auc_score(
                overall_prediction_df.true_label,
                overall_prediction_df.prediction
            ),
            'overall_auprc': average_precision_score(
                overall_prediction_df.true_label,
                overall_prediction_df.prediction
            ),
            'overall_mcc': matthews_corrcoef(
                overall_prediction_df.true_label,
                overall_prediction_df.prediction >= 4
            ),
            'overall_accuracy': accuracy_score(
                overall_prediction_df.true_label,
                overall_prediction_df.prediction >= 4
            ),
            'n_pos_samples': np.sum(overall_prediction_df.true_label),
            'n_samples': len(overall_prediction_df),
            'cv_fold': cv_fold
        }, index=[0])

        all_folds_results = pd.concat([all_folds_results, overall_results_df])

        # Compute median metrics
        median_results_df = pd.DataFrame({
            'median_roc': np.nanmedian(metrics_per_timestep_df.roc),
            'median_auprc': np.nanmedian(metrics_per_timestep_df.auprc),
            'median_mcc': np.nanmedian(metrics_per_timestep_df.mcc),
            'median_accuracy': np.nanmedian(metrics_per_timestep_df.accuracy),
            'n_pos_samples': np.nanmedian(metrics_per_timestep_df.n_pos_samples),
        }, index=[0])

        # Save fold results
        overall_prediction_df.to_csv(os.path.join(fold_result_dir, 'overall_validation_predictions.csv'), index=False)
        overall_results_df.to_csv(os.path.join(fold_result_dir, 'overall_validation_results.csv'), index=False)
        median_results_df.to_csv(os.path.join(fold_result_dir, 'median_validation_results.csv'), index=False)
        metrics_per_timestep_df.to_csv(os.path.join(fold_result_dir, 'validation_scores_per_timestep.csv'), index=False)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x=range(1, n_time_steps + 1), y=metrics_per_timestep_df.roc, label='AUROC', ax=ax)
        sns.scatterplot(x=range(1, n_time_steps + 1), y=metrics_per_timestep_df.auprc, label='AUPRC', ax=ax)
        sns.scatterplot(x=range(1, n_time_steps + 1), y=metrics_per_timestep_df.mcc, label='MCC', ax=ax)

        ax2 = ax.twinx()
        sns.scatterplot(
            x=range(1, n_time_steps + 1),
            y=metrics_per_timestep_df.n_pos_samples,
            color='red', alpha=0.3, ax=ax2,
            label='Positive Samples', zorder=0
        )

        ax.set_xlabel('Time step')
        ax.set_ylabel('Score')
        ax.set_title(f'MIRA Zero-Shot Validation Scores Over Time (Fold {cv_fold})')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.savefig(os.path.join(fold_result_dir, 'validation_scores_over_time.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFold {cv_fold} Results:")
        print(f"  Overall ROC: {overall_results_df['overall_roc'].iloc[0]:.4f}")
        print(f"  Overall AUPRC: {overall_results_df['overall_auprc'].iloc[0]:.4f}")
        print(f"  Overall MCC: {overall_results_df['overall_mcc'].iloc[0]:.4f}")
        print(f"  Overall Accuracy: {overall_results_df['overall_accuracy'].iloc[0]:.4f}")

    # Save all folds results
    all_folds_results.to_csv(os.path.join(output_path, 'all_folds_overall_validation_results.csv'), index=False)

    # Print summary
    print(f"\n{'='*50}")
    print("Overall Summary Across All Folds:")
    print(f"{'='*50}")
    print(f"Mean ROC: {all_folds_results['overall_roc'].mean():.4f} (+/- {all_folds_results['overall_roc'].std():.4f})")
    print(f"Mean AUPRC: {all_folds_results['overall_auprc'].mean():.4f} (+/- {all_folds_results['overall_auprc'].std():.4f})")
    print(f"Mean MCC: {all_folds_results['overall_mcc'].mean():.4f} (+/- {all_folds_results['overall_mcc'].std():.4f})")
    print(f"Mean Accuracy: {all_folds_results['overall_accuracy'].mean():.4f} (+/- {all_folds_results['overall_accuracy'].std():.4f})")

    return all_folds_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate MIRA for early neurological deterioration prediction')
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help='Path to data splits .pth file')
    parser.add_argument('-n', '--normalisation_data_path', type=str, required=True,
                        help='Path to normalisation parameters CSV')
    parser.add_argument('-o', '--outcome_data_path', type=str, required=True,
                        help='Path to outcome labels CSV')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save results')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference')
    parser.add_argument('--n_time_steps', type=int, default=72,
                        help='Number of timesteps')
    parser.add_argument('--eval_n_time_steps_before_event', type=int, default=6,
                        help='Forecast horizon (hours)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--use_cross_validation', action='store_true',
                        help='Evaluate all CV folds')
    parser.add_argument('--model_name', type=str, default='MIRA-Mode/MIRA',
                        help='MIRA model name on HuggingFace')

    args = parser.parse_args()

    mira_validation_evaluation(
        data_path=args.data_path,
        normalisation_data_path=args.normalisation_data_path,
        outcome_data_path=args.outcome_data_path,
        output_path=args.output_path,
        use_gpu=args.use_gpu,
        n_time_steps=args.n_time_steps,
        eval_n_time_steps_before_event=args.eval_n_time_steps_before_event,
        batch_size=args.batch_size,
        use_cross_validation=args.use_cross_validation,
        model_name=args.model_name
    )
