#!/usr/bin/env python
"""
CLI entry point for fine-tuning MIRA with multivariate input.

Fine-tunes MIRA from pretrained weights using all features as input,
then evaluates classification of early neurological deterioration
from forecasted NIHSS deltas.

Usage:
    python run_mira_finetuning.py --use_gpu \
        --n_epochs 20 --batch_size 32 --lr 1e-4 --lr_backbone 1e-5 --patience 5

    python run_mira_finetuning.py --use_gpu --use_cross_validation
"""

import os
import sys
import argparse
from datetime import datetime

# Add MIRA to path
MIRA_PATH = os.path.join(os.path.dirname(__file__), 'MIRA')
if MIRA_PATH not in sys.path:
    sys.path.insert(0, MIRA_PATH)

from mira_finetuning import finetune_and_evaluate


# Default paths
DEFAULT_DATA_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
DEFAULT_NORMALISATION_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/logs_30012026_154047/normalisation_parameters.csv'
DEFAULT_OUTCOME_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/preprocessed_outcomes_short_term_30012026_154047.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune MIRA with multivariate input for early neurological deterioration prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('-d', '--data_path', type=str,
                        default=DEFAULT_DATA_PATH,
                        help='Path to data splits .pth file')
    parser.add_argument('-n', '--normalisation_data_path', type=str,
                        default=DEFAULT_NORMALISATION_PATH,
                        help='Path to normalisation parameters CSV')
    parser.add_argument('-o', '--outcome_data_path', type=str,
                        default=DEFAULT_OUTCOME_PATH,
                        help='Path to outcome labels CSV')

    # Output
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save results (default: creates timestamped dir)')

    # Model settings
    parser.add_argument('--model_name', type=str, default='MIRA-Mode/MIRA',
                        help='MIRA model name on HuggingFace')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training and inference')

    # Training hyperparameters
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for new layers (embed, output heads)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='Learning rate for pretrained backbone (transformer layers)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs)')

    # Evaluation settings
    parser.add_argument('--n_time_steps', type=int, default=72,
                        help='Number of timesteps to evaluate')
    parser.add_argument('--eval_n_time_steps_before_event', type=int, default=6,
                        help='Forecast horizon (hours)')
    parser.add_argument('--use_cross_validation', action='store_true',
                        help='Fine-tune and evaluate all CV folds (5 folds)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        sys.exit(1)
    if not os.path.exists(args.normalisation_data_path):
        print(f"Error: Normalisation data path does not exist: {args.normalisation_data_path}")
        sys.exit(1)
    if not os.path.exists(args.outcome_data_path):
        print(f"Error: Outcome data path does not exist: {args.outcome_data_path}")
        sys.exit(1)

    # Set output path if not provided
    if args.output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_path = os.path.join(
            os.path.dirname(__file__),
            f'mira_finetuning_results_{timestamp}'
        )

    print("=" * 60)
    print("MIRA Fine-tuning for Early Neurological Deterioration")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Normalisation path: {args.normalisation_data_path}")
    print(f"  Outcome path: {args.outcome_data_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {'GPU' if args.use_gpu else 'CPU'}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR (new layers): {args.lr}")
    print(f"  LR (backbone): {args.lr_backbone}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Patience: {args.patience}")
    print(f"  Timesteps: {args.n_time_steps}")
    print(f"  Forecast horizon: {args.eval_n_time_steps_before_event} hours")
    print(f"  Cross-validation: {args.use_cross_validation}")
    print("=" * 60)

    results = finetune_and_evaluate(
        data_path=args.data_path,
        normalisation_data_path=args.normalisation_data_path,
        outcome_data_path=args.outcome_data_path,
        output_path=args.output_path,
        model_name=args.model_name,
        use_gpu=args.use_gpu,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        patience=args.patience,
        n_time_steps=args.n_time_steps,
        eval_n_time_steps_before_event=args.eval_n_time_steps_before_event,
        use_cross_validation=args.use_cross_validation,
    )

    print(f"\nResults saved to: {args.output_path}")
    print("\nDone!")

    return results


if __name__ == '__main__':
    main()
