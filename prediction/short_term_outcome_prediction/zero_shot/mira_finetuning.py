"""
Fine-tuning pipeline for MIRA with multivariate input.

Loads pretrained MIRA weights into a model with input_size=N_features,
trains on OPSUM data using next-step forecasting, and evaluates
classification of early neurological deterioration from forecasted NIHSS deltas.
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    accuracy_score
)
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict
import math

# Add MIRA to path
MIRA_PATH = os.path.join(os.path.dirname(__file__), 'MIRA')
if MIRA_PATH not in sys.path:
    sys.path.insert(0, MIRA_PATH)

from MIRA.models.modeling_mira import MIRAForPrediction
from MIRA.models.configuration_mira import MIRAConfig

from mira_data_loader import (
    load_data_splits,
    get_feature_indices,
    get_all_feature_names_and_indices,
    get_scaler,
    apply_scaler,
    load_normalisation_parameters,
    load_outcome_data,
    get_validation_patient_ids
)
from mira_inference import (
    forecast_at_timestep_multivariate
)
from mira_evaluation import (
    reverse_normalisation,
    ensure_dir
)


class OPSUMForecastDataset(Dataset):
    """
    Dataset that creates (context, target) sliding window pairs for next-step forecasting.

    For each patient and each timestep t (0 to n_time_steps-2):
    - input_ids: X[case, :t+1, :] padded to max_seq_len
    - labels: X[case, t+1, :] (next timestep values)
    - time_values: [0, 1, ..., t] padded to max_seq_len
    - attention_mask: 1 for real tokens, 0 for padding
    """

    def __init__(self, X_scaled: np.ndarray, max_seq_len: int = 72):
        """
        Args:
            X_scaled: Scaled data [n_cases, n_time_steps, n_features]
            max_seq_len: Maximum sequence length for padding
        """
        self.X_scaled = X_scaled
        self.max_seq_len = max_seq_len
        self.n_cases, self.n_time_steps, self.n_features = X_scaled.shape

        # Build index: (case_idx, timestep) pairs
        # For each case, we create samples for t=0..n_time_steps-2
        # (need at least 1 input token and 1 target)
        self.samples = []
        for case_idx in range(self.n_cases):
            for t in range(self.n_time_steps - 1):
                self.samples.append((case_idx, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_idx, t = self.samples[idx]

        # Context: values from 0 to t (inclusive)
        context_len = t + 1
        context = self.X_scaled[case_idx, :context_len, :]  # [context_len, n_features]

        # Target: next timestep values
        target = self.X_scaled[case_idx, context_len, :]  # [n_features]

        # Pad context to max_seq_len
        padded_input = np.zeros((self.max_seq_len, self.n_features), dtype=np.float32)
        padded_input[:context_len, :] = context

        # Time values
        padded_time = np.zeros(self.max_seq_len, dtype=np.float32)
        padded_time[:context_len] = np.arange(context_len, dtype=np.float32)

        # Attention mask
        attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        attention_mask[:context_len] = 1.0

        # Labels: [1, n_features] to match model output shape
        labels = target[np.newaxis, :]  # [1, n_features]

        return {
            'input_ids': torch.tensor(padded_input, dtype=torch.float32),
            'time_values': torch.tensor(padded_time, dtype=torch.float32),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
        }


def load_mira_for_finetuning(
    model_name: str,
    n_features: int,
    device: str = "cuda"
) -> MIRAForPrediction:
    """
    Load pretrained MIRA and adapt it for multivariate input.

    Creates a new model with input_size=n_features, then copies
    all shape-compatible weights from the pretrained checkpoint.
    Embedding and output layers (which depend on input_size) are
    randomly initialized.

    Args:
        model_name: HuggingFace model name or local path
        n_features: Number of input features
        device: Device to load model on

    Returns:
        MIRAForPrediction with pretrained backbone and fresh I/O layers
    """
    # Load pretrained config and modify input_size
    config = MIRAConfig.from_pretrained(model_name)
    config.input_size = n_features
    config.use_terminal_ode = False

    # Create new model with modified config (random init)
    model = MIRAForPrediction(config)
    model.use_terminal_ode = False
    model.ode_extrapolation_block = None

    # Load pretrained state dict
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    # Try loading from safetensors first, then pytorch
    try:
        model_file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        pretrained_state = safetensors.torch.load_file(model_file)
    except Exception:
        model_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        pretrained_state = torch.load(model_file, map_location='cpu', weights_only=True)

    # Copy shape-compatible weights
    new_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, pretrained_param in pretrained_state.items():
        if key in new_state:
            if new_state[key].shape == pretrained_param.shape:
                new_state[key] = pretrained_param
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key}: pretrained {pretrained_param.shape} vs new {new_state[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in new model")

    model.load_state_dict(new_state)

    print(f"Loaded {len(loaded_keys)} pretrained weight tensors")
    print(f"Skipped {len(skipped_keys)} tensors (shape mismatch or not found):")
    for s in skipped_keys:
        print(f"  {s}")

    model = model.to(device)
    return model


def get_parameter_groups(
    model: MIRAForPrediction,
    lr_backbone: float = 1e-5,
    lr_new: float = 1e-4,
    weight_decay: float = 0.01
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.

    Backbone (transformer layers, norms, MoE): lower lr
    New layers (embed_layer, lm_heads): higher lr

    Args:
        model: The model
        lr_backbone: Learning rate for pretrained backbone
        lr_new: Learning rate for new layers
        weight_decay: Weight decay

    Returns:
        List of parameter group dicts for optimizer
    """
    new_layer_names = ['embed_layer', 'lm_heads']

    backbone_params = []
    new_params = []

    for name, param in model.named_parameters():
        if any(nl in name for nl in new_layer_names):
            new_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': new_params, 'lr': lr_new, 'weight_decay': weight_decay},
    ]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: MIRAForPrediction,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    use_amp: bool = False
) -> float:
    """
    Train for one epoch using MIRA's built-in forward + HuberLoss.

    Args:
        model: MIRA model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        use_amp: Whether to use automatic mixed precision

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        time_values = batch['time_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                time_values=time_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def compute_validation_loss(
    model: MIRAForPrediction,
    val_loader: DataLoader,
    device: str,
    use_amp: bool = False
) -> float:
    """
    Compute average validation loss.

    Args:
        model: MIRA model
        val_loader: Validation data loader
        device: Device
        use_amp: Whether to use automatic mixed precision

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation loss", leave=False):
            input_ids = batch['input_ids'].to(device)
            time_values = batch['time_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    time_values=time_values,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

            total_loss += outputs.loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_fold_forecasting(
    model: MIRAForPrediction,
    X_val: np.ndarray,
    X_train: np.ndarray,
    outcome_df: pd.DataFrame,
    scaler,
    normalisation_parameters_df: pd.DataFrame,
    n_time_steps: int = 72,
    eval_n_time_steps_before_event: int = 6,
    batch_size: int = 64,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate fine-tuned multivariate model on classification task.

    Mirrors the evaluation protocol from mira_evaluation.evaluate_fold():
    - Forecast all features autoregressively for 6 steps at each timestep
    - Extract max_NIHSS and min_NIHSS channels
    - Reverse normalization and compute delta_NIHSS
    - Classify deterioration if delta >= 4

    Args:
        model: Fine-tuned MIRA model
        X_val: Raw validation data [n_cases, n_time_steps, n_features, n_dims]
        X_train: Raw training data (for feature indices)
        outcome_df: Outcome labels DataFrame
        scaler: Fitted StandardScaler
        normalisation_parameters_df: Normalisation parameters
        n_time_steps: Number of timesteps
        eval_n_time_steps_before_event: Forecast horizon
        batch_size: Batch size for inference
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (overall_prediction_df, metrics_per_timestep_df)
    """
    model.eval()

    # Get feature indices
    min_nihss_idx, max_nihss_idx = get_feature_indices(X_train)

    # Apply scaler to validation data
    X_val_scaled = apply_scaler(X_val, scaler)

    # Get validation patient IDs
    val_patient_cids = get_validation_patient_ids(X_val)

    # Forecast at each timestep
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
        y_true_at_evaluated_ts = np.isin(
            val_patient_cids, outcome_at_evaluated_ts_df['case_admission_id'].values
        ).astype(np.int32)

        # Forecast all features at this timestep
        forecasts_at_ts = forecast_at_timestep_multivariate(
            model,
            X_val_scaled,
            timestep=ts,
            n_forecast_steps=eval_n_time_steps_before_event,
            batch_size=batch_size
        )  # [n_cases, n_forecast_steps, n_features]

        # Extract max_NIHSS from last forecast step
        max_nihss_forecast_scaled = forecasts_at_ts[:, -1, max_nihss_idx]  # [n_cases]

        # Historical min_NIHSS up to current timestep (scaled)
        scaled_min_nihss = X_val_scaled[:, :ts + 1, min_nihss_idx]
        min_nihss_up_to_current_ts = np.min(scaled_min_nihss, axis=1)  # [n_cases]

        # Reverse StandardScaler transform for max_NIHSS
        dummy_array = np.zeros((len(max_nihss_forecast_scaled), X_val_scaled.shape[2]))
        dummy_array[:, max_nihss_idx] = max_nihss_forecast_scaled
        reverse_scaled = scaler.inverse_transform(dummy_array)
        max_nihss_forecast_reverse_scaled = reverse_scaled[:, max_nihss_idx]

        # Reverse StandardScaler transform for min_NIHSS
        dummy_array_min = np.zeros((len(min_nihss_up_to_current_ts), X_val_scaled.shape[2]))
        dummy_array_min[:, min_nihss_idx] = min_nihss_up_to_current_ts
        reverse_scaled_min = scaler.inverse_transform(dummy_array_min)
        min_nihss_reverse_scaled = reverse_scaled_min[:, min_nihss_idx]

        # Reverse preprocessing normalisation
        max_nihss_actual = reverse_normalisation(
            max_nihss_forecast_reverse_scaled, 'max_NIHSS', normalisation_parameters_df
        )
        min_nihss_actual = reverse_normalisation(
            min_nihss_reverse_scaled, 'min_NIHSS', normalisation_parameters_df
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

    return overall_prediction_df, metrics_per_timestep_df


def finetune_and_evaluate(
    data_path: str,
    normalisation_data_path: str,
    outcome_data_path: str,
    output_path: str,
    model_name: str = "MIRA-Mode/MIRA",
    use_gpu: bool = True,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 0.01,
    patience: int = 5,
    n_time_steps: int = 72,
    eval_n_time_steps_before_event: int = 6,
    use_cross_validation: bool = True,
) -> pd.DataFrame:
    """
    Main cross-validation loop: fine-tune MIRA and evaluate on each fold.

    Args:
        data_path: Path to data splits .pth file
        normalisation_data_path: Path to normalisation parameters CSV
        outcome_data_path: Path to outcome labels CSV
        output_path: Path to save results
        model_name: MIRA model name on HuggingFace
        use_gpu: Whether to use GPU
        n_epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate for new layers
        lr_backbone: Learning rate for pretrained backbone
        weight_decay: Weight decay
        patience: Early stopping patience
        n_time_steps: Number of timesteps
        eval_n_time_steps_before_event: Forecast horizon
        use_cross_validation: Whether to evaluate all folds

    Returns:
        DataFrame with overall results across folds
    """
    ensure_dir(output_path)

    # Load data
    print("Loading data...")
    splits = load_data_splits(data_path)
    normalisation_parameters_df = load_normalisation_parameters(normalisation_data_path)
    outcome_df = load_outcome_data(outcome_data_path)

    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    use_amp = use_gpu and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Determine folds
    fold_range = range(len(splits)) if use_cross_validation else [0]

    all_folds_results = pd.DataFrame()

    for cv_fold in fold_range:
        print(f"\n{'='*50}")
        print(f"Fold {cv_fold + 1}/{len(splits)}")
        print(f"{'='*50}")

        fold_result_dir = os.path.join(output_path, f'cv_fold_{cv_fold}')
        ensure_dir(fold_result_dir)

        X_train, X_val, y_train, y_val = splits[cv_fold]

        # Fit scaler on training data
        scaler = get_scaler(X_train)
        X_train_scaled = apply_scaler(X_train, scaler)
        X_val_scaled = apply_scaler(X_val, scaler)

        n_features = X_train_scaled.shape[2]
        print(f"Number of features: {n_features}")

        # Create datasets
        train_dataset = OPSUMForecastDataset(X_train_scaled, max_seq_len=n_time_steps)
        val_dataset = OPSUMForecastDataset(X_val_scaled, max_seq_len=n_time_steps)

        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        # Load model with pretrained backbone
        print("Loading MIRA model for fine-tuning...")
        model = load_mira_for_finetuning(model_name, n_features, device)

        # Optimizer with differential learning rates
        param_groups = get_parameter_groups(model, lr_backbone, lr, weight_decay)
        optimizer = torch.optim.AdamW(param_groups)

        # Scheduler
        num_training_steps = n_epochs * len(train_loader)
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        training_log = []

        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, use_amp)
            val_loss = compute_validation_loss(model, val_loader, device, use_amp)

            print(f"  Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            training_log.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr_backbone': optimizer.param_groups[0]['lr'],
                'lr_new': optimizer.param_groups[1]['lr'],
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print(f"  New best validation loss: {val_loss:.6f}")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Save training log
        training_log_df = pd.DataFrame(training_log)
        training_log_df.to_csv(os.path.join(fold_result_dir, 'training_loss.csv'), index=False)

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save best model checkpoint
        best_model_dir = os.path.join(fold_result_dir, 'best_model')
        ensure_dir(best_model_dir)
        torch.save(model.state_dict(), os.path.join(best_model_dir, 'model_state_dict.pt'))
        model.config.save_pretrained(best_model_dir)

        # Evaluate classification performance
        print("\nEvaluating classification performance...")
        overall_prediction_df, metrics_per_timestep_df = evaluate_fold_forecasting(
            model,
            X_val,
            X_train,
            outcome_df,
            scaler,
            normalisation_parameters_df,
            n_time_steps=n_time_steps,
            eval_n_time_steps_before_event=eval_n_time_steps_before_event,
            batch_size=batch_size,
            show_progress=True
        )

        # Compute overall metrics
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
            'cv_fold': cv_fold,
            'best_val_loss': best_val_loss,
        }, index=[0])

        all_folds_results = pd.concat([all_folds_results, overall_results_df])

        # Save fold results
        overall_prediction_df.to_csv(
            os.path.join(fold_result_dir, 'overall_validation_predictions.csv'), index=False
        )
        overall_results_df.to_csv(
            os.path.join(fold_result_dir, 'overall_validation_results.csv'), index=False
        )
        metrics_per_timestep_df.to_csv(
            os.path.join(fold_result_dir, 'validation_scores_per_timestep.csv'), index=False
        )

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
        ax.set_title(f'MIRA Fine-tuned Validation Scores Over Time (Fold {cv_fold})')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.savefig(
            os.path.join(fold_result_dir, 'validation_scores_over_time.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

        print(f"\nFold {cv_fold} Results:")
        print(f"  Overall ROC: {overall_results_df['overall_roc'].iloc[0]:.4f}")
        print(f"  Overall AUPRC: {overall_results_df['overall_auprc'].iloc[0]:.4f}")
        print(f"  Overall MCC: {overall_results_df['overall_mcc'].iloc[0]:.4f}")
        print(f"  Overall Accuracy: {overall_results_df['overall_accuracy'].iloc[0]:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")

    # Save all folds results
    all_folds_results.to_csv(
        os.path.join(output_path, 'all_folds_overall_validation_results.csv'), index=False
    )

    # Print summary
    print(f"\n{'='*50}")
    print("Overall Summary Across All Folds:")
    print(f"{'='*50}")
    print(f"Mean ROC: {all_folds_results['overall_roc'].mean():.4f} (+/- {all_folds_results['overall_roc'].std():.4f})")
    print(f"Mean AUPRC: {all_folds_results['overall_auprc'].mean():.4f} (+/- {all_folds_results['overall_auprc'].std():.4f})")
    print(f"Mean MCC: {all_folds_results['overall_mcc'].mean():.4f} (+/- {all_folds_results['overall_mcc'].std():.4f})")
    print(f"Mean Accuracy: {all_folds_results['overall_accuracy'].mean():.4f} (+/- {all_folds_results['overall_accuracy'].std():.4f})")

    return all_folds_results
