"""Extract encoder representations from pretrained self-supervised models.

For each CV fold, loads the best checkpoint, runs model.encode() on train and
validation data, and saves the hidden states. These representations can then be
concatenated with hand-crafted features for XGBoost.

Output structure per fold:
    {output_dir}/fold_{i}/train_representations.pth  - dict with 'representations' and 'labels'
    {output_dir}/fold_{i}/val_representations.pth    - dict with 'representations' and 'labels'

Each representations tensor has shape (n_subsequences,) where each element is
a tensor of shape (seq_len, 2*model_dim).
"""

import os
import json
import glob
import argparse
import numpy as np
import torch as ch
from torch.utils.data import DataLoader

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitSelfSupervisedModel
from prediction.short_term_outcome_prediction.timeseries_decomposition import (
    BucketBatchSampler,
    prepare_subsequence_dataset,
    aggregate_and_label_timeseries,
)
from prediction.utils.utils import ensure_dir, aggregate_features_over_time
from sklearn.preprocessing import StandardScaler


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint file in a directory."""
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    if len(ckpt_files) == 1:
        return ckpt_files[0]
    # Sort by val_loss in filename (lower is better)
    return sorted(ckpt_files)[0]


def extract_representations_for_fold(model, dataset, batch_size=1024, device='cuda'):
    """Run model.encode() on a dataset and collect per-subsequence representations.

    Returns:
        all_reprs: list of numpy arrays, each of shape (seq_len, hidden_dim)
        all_labels: numpy array of labels
    """
    bucket_sampler = BucketBatchSampler(dataset.idx_to_len_map, batch_size)
    loader = DataLoader(dataset, batch_sampler=bucket_sampler)

    all_reprs = []
    all_labels = []
    all_indices = []

    model.eval()
    with ch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            # x: (batch, seq_len, features)
            hidden = model.encode(x)  # (batch, seq_len, 2*model_dim)
            # Store each subsequence's representation
            for j in range(hidden.shape[0]):
                all_reprs.append(hidden[j].cpu().numpy())
                all_labels.append(y[j].cpu().item())

    all_labels = np.array(all_labels)
    return all_reprs, all_labels


def extract_aggregate_representations(model, dataset, batch_size=1024):
    """Extract representations and aggregate them per-timestep for XGBoost.

    For each subsequence of length T, takes the last hidden state (at position T-1)
    as the representation for that timestep. This gives one representation vector
    per (patient, timestep) pair — matching the XGBoost data structure.

    Returns:
        agg_reprs: numpy array of shape (n_subsequences, 2*model_dim)
        labels: numpy array of shape (n_subsequences,)
    """
    bucket_sampler = BucketBatchSampler(dataset.idx_to_len_map, batch_size)
    loader = DataLoader(dataset, batch_sampler=bucket_sampler)

    all_reprs = []
    all_labels = []

    model.eval()
    with ch.no_grad():
        for x, y in loader:
            hidden = model.encode(x)  # (batch, seq_len, 2*model_dim)
            # Take last timestep's representation for each subsequence
            last_hidden = hidden[:, -1, :]  # (batch, 2*model_dim)
            all_reprs.append(last_hidden.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_reprs = np.concatenate(all_reprs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_reprs, all_labels


def extract_all_folds(pretrained_dir: str, data_splits_path: str, output_dir: str,
                      use_gpu: bool = True, aggregate: bool = True):
    """Extract representations for all CV folds.

    Args:
        pretrained_dir: directory containing pretrained checkpoints (one subdir per fold)
        data_splits_path: path to data splits .pth file
        output_dir: where to save extracted representations
        use_gpu: whether to use GPU
        aggregate: if True, aggregate to per-timestep vectors for XGBoost
    """
    ensure_dir(output_dir)

    # Load config to reconstruct model
    config_path = os.path.join(pretrained_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Try to find trial params
        param_files = glob.glob(os.path.join(pretrained_dir, 'trial_params_*.json'))
        if param_files:
            with open(sorted(param_files)[-1]) as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No config found in {pretrained_dir}")

    model_dim = config.get('model_dim', 256)
    num_layers = config.get('num_layers', 4)
    num_heads = config.get('num_head', 8)
    dropout = config.get('dropout', 0.3)
    ff_dim = 2 * model_dim
    pos_encode_factor = config.get('pos_encode_factor', 0.1)

    splits = ch.load(data_splits_path)
    all_datasets = [prepare_subsequence_dataset(x, use_gpu=use_gpu,
                                                target_interval=True,
                                                restrict_to_first_event=False,
                                                ) for x in splits]

    # Find checkpoint directories
    checkpoint_dirs = sorted(glob.glob(os.path.join(pretrained_dir, 'checkpoints_ss_*')))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {pretrained_dir}")

    device = 'cuda' if use_gpu and ch.cuda.is_available() else 'cpu'

    for i, (train_dataset, val_dataset) in enumerate(all_datasets):
        print(f"\nProcessing fold {i}...")
        fold_output_dir = os.path.join(output_dir, f'fold_{i}')
        ensure_dir(fold_output_dir)

        input_dim = train_dataset[0][0].shape[-1]

        # Find checkpoint for this fold
        fold_ckpt_dir = [d for d in checkpoint_dirs if f'_cv_{i}' in d]
        if not fold_ckpt_dir:
            print(f"  WARNING: No checkpoint found for fold {i}, skipping")
            continue
        ckpt_path = find_best_checkpoint(fold_ckpt_dir[0])
        print(f"  Loading checkpoint: {ckpt_path}")

        # Reconstruct model (causal=True for proper masking)
        model = OPSUMTransformer(
            input_dim=input_dim,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=input_dim,
            max_dim=500,
            pos_encode_factor=pos_encode_factor,
            causal=True
        )

        # Load weights from lightning checkpoint
        lit_model = LitSelfSupervisedModel.load_from_checkpoint(
            ckpt_path, model=model, lr=0, wd=0, train_noise=0
        )
        model = lit_model.model
        model.to(device)
        model.eval()

        if aggregate:
            # Extract aggregated representations (one vector per subsequence)
            print("  Extracting aggregated train representations...")
            train_reprs, train_labels = extract_aggregate_representations(model, train_dataset)
            print(f"  Train: {train_reprs.shape[0]} samples, repr dim = {train_reprs.shape[1]}")

            print("  Extracting aggregated val representations...")
            val_reprs, val_labels = extract_aggregate_representations(model, val_dataset)
            print(f"  Val: {val_reprs.shape[0]} samples, repr dim = {val_reprs.shape[1]}")

            ch.save({
                'representations': train_reprs,
                'labels': train_labels,
            }, os.path.join(fold_output_dir, 'train_representations.pth'))

            ch.save({
                'representations': val_reprs,
                'labels': val_labels,
            }, os.path.join(fold_output_dir, 'val_representations.pth'))
        else:
            # Extract full sequence representations
            print("  Extracting full train representations...")
            train_reprs, train_labels = extract_representations_for_fold(model, train_dataset)
            print(f"  Train: {len(train_reprs)} subsequences")

            print("  Extracting full val representations...")
            val_reprs, val_labels = extract_representations_for_fold(model, val_dataset)
            print(f"  Val: {len(val_reprs)} subsequences")

            ch.save({
                'representations': train_reprs,
                'labels': train_labels,
            }, os.path.join(fold_output_dir, 'train_representations.pth'))

            ch.save({
                'representations': val_reprs,
                'labels': val_labels,
            }, os.path.join(fold_output_dir, 'val_representations.pth'))

    # Save extraction metadata
    meta = {
        'pretrained_dir': pretrained_dir,
        'data_splits_path': data_splits_path,
        'config': config,
        'aggregate': aggregate,
        'n_folds': len(all_datasets),
        'hidden_dim': 2 * model_dim,
    }
    with open(os.path.join(output_dir, 'extraction_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nRepresentations saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract encoder representations from pretrained self-supervised model')
    parser.add_argument('-p', '--pretrained_dir', type=str, required=True,
                        help='Directory containing pretrained checkpoints')
    parser.add_argument('-d', '--data_splits_path', type=str, required=True,
                        help='Path to data splits .pth file')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory for extracted representations')
    parser.add_argument('-g', '--use_gpu', type=int, required=False, default=1)
    parser.add_argument('--full', action='store_true',
                        help='Save full sequence representations (not aggregated)')

    args = parser.parse_args()

    extract_all_folds(
        pretrained_dir=args.pretrained_dir,
        data_splits_path=args.data_splits_path,
        output_dir=args.output_dir,
        use_gpu=args.use_gpu == 1,
        aggregate=not args.full,
    )
