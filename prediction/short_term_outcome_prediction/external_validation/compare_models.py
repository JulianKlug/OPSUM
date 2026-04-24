import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from prediction.short_term_outcome_prediction.testing.compare_models import compare_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare XGB vs LR on external MIMIC validation data')
    parser.add_argument('-a', '--pred_path_a', type=str, required=True,
                        help='Path to test_predictions.pkl for model A (XGB)')
    parser.add_argument('-b', '--pred_path_b', type=str, required=True,
                        help='Path to test_predictions.pkl for model B (LR)')
    parser.add_argument('--label_a', type=str, default='XGB',
                        help='Display name for model A (default: XGB)')
    parser.add_argument('--label_b', type=str, default='LR',
                        help='Display name for model B (default: LR)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory for comparison results')
    parser.add_argument('-n', '--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000)')
    args = parser.parse_args()

    compare_models(
        pred_path_a=args.pred_path_a,
        pred_path_b=args.pred_path_b,
        label_a=args.label_a,
        label_b=args.label_b,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )
