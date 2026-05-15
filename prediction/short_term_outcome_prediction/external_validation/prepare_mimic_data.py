import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from prediction.outcome_prediction.data_loading.data_formatting import features_to_numpy
from prediction.utils.utils import check_data


def prepare_mimic_data(features_path: str, labels_path: str, output_path: str):
    """Convert MIMIC CSV data to .pth format for model evaluation.

    Follows the data loading pattern from data_splits.py but without
    train/test splitting — all data is used as a single external test set.

    Args:
        features_path: path to preprocessed_features CSV
        labels_path: path to preprocessed_outcomes_short_term CSV
        output_path: path to save the .pth file
    """
    # Load data
    print("Loading features...")
    X = pd.read_csv(features_path)
    print(f"Features shape: {X.shape}")

    print("Loading outcomes...")
    y = pd.read_csv(labels_path)
    print(f"Outcomes shape: {y.shape}")

    # Check data integrity
    check_data(X)

    # Filter outcomes to early_neurological_deterioration (same as data_splits.py:119)
    y_filtered = y[y.outcome_label == 'early_neurological_deterioration']
    print(f"Outcomes after filtering to END: {y_filtered.shape}")

    # Convert features to 4D numpy array
    columns_to_keep = ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value']
    print("Converting features to numpy array...")
    X_np = features_to_numpy(X, columns_to_keep)

    # Summary stats
    n_patients = X_np.shape[0]
    n_timepoints = X_np.shape[1]
    n_features = X_np.shape[2]
    n_positive = y_filtered.shape[0]
    all_cids = set(X_np[:, 0, 0, 0])
    positive_cids = set(y_filtered.case_admission_id.values)
    n_patients_positive = len(all_cids & positive_cids)

    print(f"\nSummary:")
    print(f"  Patients (admissions): {n_patients}")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Features: {n_features}")
    print(f"  Positive cases (END): {n_patients_positive}")
    print(f"  Positive rate: {n_patients_positive / n_patients * 100:.2f}%")

    # Save as (X_np, y_filtered_df) tuple
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save((X_np, y_filtered), output_path)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MIMIC CSV data to .pth format')
    parser.add_argument('-f', '--features_path', type=str, required=True,
                        help='Path to preprocessed features CSV')
    parser.add_argument('-l', '--labels_path', type=str, required=True,
                        help='Path to preprocessed short-term outcomes CSV')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='Path to save the output .pth file')
    args = parser.parse_args()

    prepare_mimic_data(
        features_path=args.features_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
    )
