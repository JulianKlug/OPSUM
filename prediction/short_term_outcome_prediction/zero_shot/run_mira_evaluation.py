#!/usr/bin/env python
"""
Main runner script for MIRA zero-shot evaluation.

This script evaluates the pretrained MIRA model at zero-shot prediction
of early neurological deterioration in the next 6 hours.

Usage:
    python run_mira_evaluation.py \
        -d /path/to/data_splits.pth \
        -n /path/to/normalisation_parameters.csv \
        -o /path/to/outcome_labels.csv \
        --output_path /path/to/results \
        --use_gpu \
        --use_cross_validation

With defaults from CLAUDE.md:
    python run_mira_evaluation.py --use_gpu --use_cross_validation
"""

import os
import sys
import argparse
from datetime import datetime

# Add MIRA to path
MIRA_PATH = os.path.join(os.path.dirname(__file__), 'MIRA')
if MIRA_PATH not in sys.path:
    sys.path.insert(0, MIRA_PATH)

from mira_evaluation import mira_validation_evaluation


# Default paths from CLAUDE.md
DEFAULT_DATA_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
DEFAULT_NORMALISATION_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/logs_30012026_154047/normalisation_parameters.csv'
DEFAULT_OUTCOME_PATH = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/preprocessed_outcomes_short_term_30012026_154047.csv'


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MIRA for zero-shot prediction of early neurological deterioration',
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
                        help='Use GPU for inference')

    # Evaluation settings
    parser.add_argument('--n_time_steps', type=int, default=72,
                        help='Number of timesteps to evaluate')
    parser.add_argument('--eval_n_time_steps_before_event', type=int, default=6,
                        help='Forecast horizon (hours)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--use_cross_validation', action='store_true',
                        help='Evaluate all CV folds (5 folds)')

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
            f'mira_evaluation_results_{timestamp}'
        )

    print("=" * 60)
    print("MIRA Zero-Shot Evaluation for Early Neurological Deterioration")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Normalisation path: {args.normalisation_data_path}")
    print(f"  Outcome path: {args.outcome_data_path}")
    print(f"  Output path: {args.output_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {'GPU' if args.use_gpu else 'CPU'}")
    print(f"  Timesteps: {args.n_time_steps}")
    print(f"  Forecast horizon: {args.eval_n_time_steps_before_event} hours")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Cross-validation: {args.use_cross_validation}")
    print("=" * 60)

    # Run evaluation
    results = mira_validation_evaluation(
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

    print(f"\nResults saved to: {args.output_path}")
    print("\nDone!")

    return results


if __name__ == '__main__':
    main()
