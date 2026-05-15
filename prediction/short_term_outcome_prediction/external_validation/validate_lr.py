import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from prediction.short_term_outcome_prediction.testing.test_logistic_regression import test_final_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='External validation of logistic regression model on MIMIC data')
    parser.add_argument('-d', '--test_data_path', type=str, required=True,
                        help='Path to MIMIC test data .pth file')
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help='Directory containing lr_final_model.pkl, scaler.pkl, final_model_config.json')
    parser.add_argument('-t', '--threshold_results_path', type=str, default=None,
                        help='Path to threshold tuning results JSON (optional)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('-n', '--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations (default: 1000)')
    args = parser.parse_args()

    test_final_model(
        test_data_path=args.test_data_path,
        model_dir=args.model_dir,
        threshold_results_path=args.threshold_results_path,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )
