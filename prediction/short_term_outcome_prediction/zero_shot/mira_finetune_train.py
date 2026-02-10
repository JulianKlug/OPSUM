"""
Training loop for finetuning MIRA on early neurological deterioration classification.

Supports:
- 5-fold cross-validation
- Class-imbalanced loss (pos_weight)
- Frozen backbone with trainable classification head
- Optional partial unfreezing of transformer layers
- GPU acceleration
- Early stopping
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    accuracy_score,
)
from tqdm import tqdm
from typing import Dict, Optional

# Add MIRA to path
MIRA_PATH = os.path.join(os.path.dirname(__file__), 'MIRA')
if MIRA_PATH not in sys.path:
    sys.path.insert(0, MIRA_PATH)

from mira_data_loader import (
    load_data_splits,
    load_normalisation_parameters,
    load_outcome_data,
    get_scaler,
)
from mira_finetune_dataset import MIRAFinetuneDataset, collate_fn
from mira_finetune_model import MIRAClassifier
from mira_inference import normalize_sequence


def train_epoch(
    model: MIRAClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for input_ids, time_values, attention_mask, labels in tqdm(dataloader, desc="Training", leave=False):
        input_ids = input_ids.to(device)
        time_values = time_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Per-sequence normalization (same as zero-shot inference)
        batch_size, seq_len, _ = input_ids.shape
        values = input_ids.squeeze(-1)  # [B, L]
        # Mask padded positions for normalization
        masked_values = values.clone()
        masked_values[attention_mask == 0] = float('nan')
        # Compute mean/std ignoring padding
        mean = torch.nanmean(masked_values, dim=1, keepdim=True)
        # Replace nans for std computation
        filled = values.clone()
        filled[attention_mask == 0] = 0
        counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        var = ((filled - mean) ** 2 * attention_mask).sum(dim=1, keepdim=True) / counts
        std = var.sqrt().clamp(min=1e-8)
        # Handle single-element sequences
        single_mask = (counts == 1)
        std = torch.where(single_mask, torch.ones_like(std), std)
        normalized = (values - mean) / std
        normalized = normalized * attention_mask  # zero out padding
        input_ids_norm = normalized.unsqueeze(-1)

        optimizer.zero_grad()
        logits = model(input_ids_norm, time_values, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MIRAClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate model. Returns dict of metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for input_ids, time_values, attention_mask, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = input_ids.to(device)
        time_values = time_values.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Per-sequence normalization
        batch_size, seq_len, _ = input_ids.shape
        values = input_ids.squeeze(-1)
        masked_values = values.clone()
        masked_values[attention_mask == 0] = float('nan')
        mean = torch.nanmean(masked_values, dim=1, keepdim=True)
        filled = values.clone()
        filled[attention_mask == 0] = 0
        counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        var = ((filled - mean) ** 2 * attention_mask).sum(dim=1, keepdim=True) / counts
        std = var.sqrt().clamp(min=1e-8)
        single_mask = (counts == 1)
        std = torch.where(single_mask, torch.ones_like(std), std)
        normalized = (values - mean) / std
        normalized = normalized * attention_mask
        input_ids_norm = normalized.unsqueeze(-1)

        logits = model(input_ids_norm, time_values, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze()
    all_labels = np.concatenate(all_labels, axis=0).squeeze()
    pred_binary = (all_preds >= 0.5).astype(int)

    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'accuracy': accuracy_score(all_labels, pred_binary),
    }

    if len(np.unique(all_labels)) > 1:
        metrics['auroc'] = roc_auc_score(all_labels, all_preds)
        metrics['auprc'] = average_precision_score(all_labels, all_preds)
        metrics['mcc'] = matthews_corrcoef(all_labels, pred_binary)
    else:
        metrics['auroc'] = float('nan')
        metrics['auprc'] = float('nan')
        metrics['mcc'] = float('nan')

    return metrics


def train_fold(
    fold_idx: int,
    splits: list,
    outcome_df: pd.DataFrame,
    config: dict,
    output_dir: str,
    device: str,
) -> Dict[str, float]:
    """Train and evaluate a single fold."""
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx + 1}/{len(splits)}")
    print(f"{'='*50}")

    fold_dir = os.path.join(output_dir, f'cv_fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    X_train, X_val, y_train, y_val = splits[fold_idx]

    # Fit scaler on training data
    scaler = get_scaler(X_train)

    # Create datasets
    print("Creating training dataset...")
    train_dataset = MIRAFinetuneDataset(
        X=X_train,
        X_for_scaler=X_train,
        outcome_df=outcome_df,
        n_time_steps=config['n_time_steps'],
        forecast_horizon=config['forecast_horizon'],
        scaler=scaler,
    )

    print("Creating validation dataset...")
    val_dataset = MIRAFinetuneDataset(
        X=X_val,
        X_for_scaler=X_train,
        outcome_df=outcome_df,
        n_time_steps=config['n_time_steps'],
        forecast_horizon=config['forecast_horizon'],
        scaler=scaler,
    )

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"  Positive weight: {train_dataset.pos_weight.item():.2f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    # Create model
    print("Loading model...")
    model = MIRAClassifier(
        model_name=config['model_name'],
        device=device,
        freeze_backbone=config['freeze_backbone'],
        unfreeze_last_n_layers=config.get('unfreeze_last_n_layers', 0),
        hidden_dim=config.get('classifier_hidden_dim', 256),
        dropout=config.get('dropout', 0.1),
    ).to(device)

    trainable, total = model.get_trainable_params()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Loss with class imbalance weighting
    pos_weight = train_dataset.pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer - only trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['n_epochs'],
        eta_min=config['learning_rate'] * 0.01,
    )

    # Training loop with early stopping
    best_val_auroc = -1
    patience_counter = 0
    patience = config.get('patience', 5)
    history = []

    for epoch in range(config['n_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['n_epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        print(f"  Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"  Val AUPRC: {val_metrics['auprc']:.4f}")
        print(f"  Val MCC: {val_metrics['mcc']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  LR: {lr:.6f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_auroc': val_metrics['auroc'],
            'val_auprc': val_metrics['auprc'],
            'val_mcc': val_metrics['mcc'],
            'val_accuracy': val_metrics['accuracy'],
            'lr': lr,
        })

        # Early stopping on AUROC
        val_auroc = val_metrics['auroc']
        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(fold_dir, 'best_model.pt'))
            print(f"  -> New best AUROC: {best_val_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(fold_dir, 'training_history.csv'), index=False)

    # Load best model and get final metrics
    best_ckpt = torch.load(os.path.join(fold_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.classifier.load_state_dict(best_ckpt['model_state_dict'])
    final_metrics = evaluate(model, val_loader, criterion, device)
    final_metrics['best_epoch'] = best_ckpt['epoch']

    print(f"\n  Best model (epoch {best_ckpt['epoch']}):")
    print(f"    AUROC: {final_metrics['auroc']:.4f}")
    print(f"    AUPRC: {final_metrics['auprc']:.4f}")
    print(f"    MCC: {final_metrics['mcc']:.4f}")
    print(f"    Accuracy: {final_metrics['accuracy']:.4f}")

    # Save final results
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(fold_dir, 'best_validation_results.csv'), index=False
    )

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return final_metrics


def run_finetuning(
    data_path: str,
    outcome_data_path: str,
    output_path: str,
    config: dict,
    device: str = 'cuda',
):
    """Run finetuning across all CV folds."""
    os.makedirs(output_path, exist_ok=True)

    # Save config
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load data
    print("Loading data splits...")
    splits = load_data_splits(data_path)
    outcome_df = load_outcome_data(outcome_data_path)

    all_results = []
    for fold_idx in range(len(splits)):
        fold_metrics = train_fold(
            fold_idx=fold_idx,
            splits=splits,
            outcome_df=outcome_df,
            config=config,
            output_dir=output_path,
            device=device,
        )
        fold_metrics['cv_fold'] = fold_idx
        all_results.append(fold_metrics)

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_path, 'all_folds_results.csv'), index=False)

    print(f"\n{'='*60}")
    print("Finetuning Summary Across All Folds:")
    print(f"{'='*60}")
    for metric in ['auroc', 'auprc', 'mcc', 'accuracy']:
        vals = results_df[metric].dropna()
        if len(vals) > 0:
            print(f"  {metric.upper()}: {vals.mean():.4f} (+/- {vals.std():.4f})")
    print(f"\nResults saved to: {output_path}")

    return results_df
