#!/usr/bin/env python
"""
Main runner for MIRA finetuning on early neurological deterioration prediction.

Usage:
    python run_mira_finetuning.py --use_gpu
    python run_mira_finetuning.py --use_gpu --unfreeze_last_n_layers 2 --n_epochs 20
"""

import os
import sys
import argparse
from datetime import datetime

# Add MIRA to path
MIRA_PATH = os.path.join(os.path.dirname(__file__), 'MIRA')
if MIRA_PATH not in sys.path:
    sys.path.insert(0, MIRA_PATH)

from mira_finetune_train import run_finetuning

# Default paths
DEFAULT_DATA_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
DEFAULT_OUTCOME_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/preprocessed_outcomes_short_term_30012026_154047.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Finetune MIRA for early neurological deterioration prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument('-d', '--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='Path to data splits .pth file')
    parser.add_argument('-o', '--outcome_data_path', type=str, default=DEFAULT_OUTCOME_PATH,
                        help='Path to outcome labels CSV')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save results')

    # Model settings
    parser.add_argument('--model_name', type=str, default='MIRA-Mode/MIRA',
                        help='MIRA model name on HuggingFace')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze MIRA backbone')
    parser.add_argument('--no_freeze_backbone', action='store_false', dest='freeze_backbone',
                        help='Do not freeze MIRA backbone (full finetuning)')
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=0,
                        help='Number of last transformer layers to unfreeze')

    # Training settings
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--classifier_hidden_dim', type=int, default=256,
                        help='Hidden dim of classification head')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout in classification head')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader num_workers')

    # Evaluation settings
    parser.add_argument('--n_time_steps', type=int, default=72,
                        help='Number of timesteps')
    parser.add_argument('--forecast_horizon', type=int, default=6,
                        help='Forecast horizon (hours)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    if not os.path.exists(args.outcome_data_path):
        print(f"Error: Outcome data path does not exist: {args.outcome_data_path}")
        sys.exit(1)

    # Set output path
    if args.output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_path = os.path.join(
            os.path.dirname(__file__),
            f'mira_finetune_results_{timestamp}'
        )

    # Build config dict
    config = {
        'model_name': args.model_name,
        'freeze_backbone': args.freeze_backbone,
        'unfreeze_last_n_layers': args.unfreeze_last_n_layers,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'dropout': args.dropout,
        'num_workers': args.num_workers,
        'n_time_steps': args.n_time_steps,
        'forecast_horizon': args.forecast_horizon,
    }

    device = 'cuda' if args.use_gpu and __import__('torch').cuda.is_available() else 'cpu'

    print("=" * 60)
    print("MIRA Finetuning for Early Neurological Deterioration")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print(f"  output_path: {args.output_path}")
    print("=" * 60)

    results = run_finetuning(
        data_path=args.data_path,
        outcome_data_path=args.outcome_data_path,
        output_path=args.output_path,
        config=config,
        device=device,
    )

    print("\nDone!")
    return results


if __name__ == '__main__':
    main()
