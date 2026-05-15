"""Generate inference explanation plots for all test subjects.

Generates test predictions (if not already present), then plots SHAP-based
inference explanations for every subject. END-positive patients are saved
in a separate subfolder.
"""

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as ch
import xgboost as xgb
from tqdm import tqdm

from prediction.short_term_outcome_prediction.figures.inference_plotting import (
    build_inference_plot_inputs,
    load_inference_raw_inputs,
    plot_joint_subject_explanation,
)
from prediction.short_term_outcome_prediction.timeseries_decomposition import (
    aggregate_and_label_timeseries,
)


def generate_test_predictions(test_data_path, model_dir):
    """Generate predictions and save test_predictions.pkl if it doesn't exist."""
    predictions_path = os.path.join(model_dir, "test_predictions.pkl")
    if os.path.exists(predictions_path):
        print(f"Predictions already exist at {predictions_path}, skipping generation.")
        return predictions_path

    print("Generating test predictions...")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, "xgb_final_model.model"))

    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, "final_model_config.json")) as f:
        config = json.load(f)

    X_test_raw, y_test_raw = ch.load(test_data_path)
    test_data_list, test_labels_list = aggregate_and_label_timeseries(
        X_test_raw,
        y_test_raw,
        target_time_to_outcome=6,
        target_interval=config.get("target_interval", True),
        restrict_to_first_event=config.get("restrict_to_first_event", False),
        add_lag_features=config.get("add_lag_features", False),
        add_rolling_features=config.get("add_rolling_features", False),
    )

    test_data = np.concatenate(test_data_list)
    y_test = np.concatenate(test_labels_list)
    X_test_scaled = scaler.transform(test_data)
    y_prob = model.predict_proba(X_test_scaled)[:, 1].astype("float32")

    with open(predictions_path, "wb") as f:
        pickle.dump((y_test, y_prob), f)

    print(f"Saved predictions to {predictions_path}")
    print(f"  Samples: {len(y_test)}, Positive: {int(y_test.sum())}")
    return predictions_path


def plot_all_subjects(
    model_dir,
    test_data_path,
    shap_values_path,
    cat_encoding_path,
    normalisation_parameters_path,
    output_dir,
    n_time_steps=72,
    dpi=300,
    feature_to_english_name_correspondence_path=None,
):
    # Step 1: Generate predictions if needed
    predictions_path = generate_test_predictions(test_data_path, model_dir)

    # Step 2: Load inference raw inputs
    print("Loading inference raw inputs...")
    raw_inputs = load_inference_raw_inputs(
        shap_values_path=shap_values_path,
        test_data_path=test_data_path,
        cat_encoding_path=cat_encoding_path,
        normalisation_parameters_path=normalisation_parameters_path,
        predictions_path=predictions_path,
        n_time_steps=n_time_steps,
    )

    # Step 3: Build plot inputs (once for all subjects)
    print("Building inference plot inputs (this may take a while)...")
    prepared = build_inference_plot_inputs(
        raw_inputs=raw_inputs,
        n_time_steps=n_time_steps,
        reverse_categorical_encoding=True,
        pool_hourly_split_values=True,
        only_keep_current_value_shap=True,
        shap_aggregation_func="sum",
        use_simplified_shap_values=True,
        smoothing_window=15,
        feature_to_english_name_correspondence_path=feature_to_english_name_correspondence_path,
    )

    # Step 4: Determine END status per subject
    gt_over_time = prepared["gt_over_time"]
    has_end = gt_over_time.max(axis=1) > 0

    n_end = int(has_end.sum())
    n_total = len(has_end)
    print(f"Subjects: {n_total} total, {n_end} END-positive, {n_total - n_end} END-negative")

    # Step 5: Create output directories
    end_dir = os.path.join(output_dir, "end_positive")
    no_end_dir = os.path.join(output_dir, "end_negative")
    os.makedirs(end_dir, exist_ok=True)
    os.makedirs(no_end_dir, exist_ok=True)

    # Step 6: Plot all subjects
    cid_to_idx = raw_inputs["cid_to_idx"]
    all_cids = raw_inputs["case_admission_ids"]

    plot_kwargs = {
        "threshold": 0.04,
        "n_features_selection": 0,
        "n_features": 1,
        "display_legend": True,
        "display_text_labels": True,
        "display_title": False,
        "plot_ground_truth": True,
    }

    for cid in tqdm(all_cids, desc="Plotting subjects"):
        subj_idx = cid_to_idx[cid]
        fig = plot_joint_subject_explanation(
            subj=subj_idx,
            predictions_over_time=prepared["predictions_over_time"],
            gt_over_time=prepared["gt_over_time"],
            feature_values_df=prepared["feature_values_df"],
            smoothed_shap_values_over_time=prepared["smoothed_shap_values_over_time"],
            shap_values_over_time=prepared["shap_values_over_time"],
            reduced_feature_names=prepared["reduced_feature_names"],
            use_simplified_shap_values=True,
            **plot_kwargs,
        )

        save_dir = end_dir if has_end[subj_idx] else no_end_dir
        fig.savefig(
            os.path.join(save_dir, f"{cid}_inference_plot.png"),
            bbox_inches="tight",
            dpi=dpi,
        )
        plt.close(fig)

    print(f"\nDone. Plots saved to {output_dir}")
    print(f"  END-positive ({n_end}): {end_dir}")
    print(f"  END-negative ({n_total - n_end}): {no_end_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Directory with final model artifacts")
    parser.add_argument("--test-data-path", required=True, help="Path to test data .pth file")
    parser.add_argument("--shap-values-path", required=True, help="Path to SHAP values pickle")
    parser.add_argument("--cat-encoding-path", required=True, help="Path to categorical encoding CSV")
    parser.add_argument("--normalisation-parameters-path", required=True, help="Path to normalisation parameters CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--n-time-steps", type=int, default=72)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--feature-to-english-name-correspondence-path", default=None)
    args = parser.parse_args()

    plot_all_subjects(
        model_dir=args.model_dir,
        test_data_path=args.test_data_path,
        shap_values_path=args.shap_values_path,
        cat_encoding_path=args.cat_encoding_path,
        normalisation_parameters_path=args.normalisation_parameters_path,
        output_dir=args.output_dir,
        n_time_steps=args.n_time_steps,
        dpi=args.dpi,
        feature_to_english_name_correspondence_path=args.feature_to_english_name_correspondence_path,
    )
