import argparse
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch as ch
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from prediction.utils.utils import filter_consecutive_numbers, smooth
from prediction.utils.visualisation_helper_functions import (
    LegendTitle,
    reverse_normalisation_for_subj,
)


def load_inference_raw_inputs(
    shap_values_path,
    test_data_path,
    cat_encoding_path,
    normalisation_parameters_path,
    predictions_path,
    n_time_steps=72,
    only_last_timestep=False,
):
    with open(os.path.join(shap_values_path), "rb") as handle:
        original_shap_values = pickle.load(handle)

    if only_last_timestep:
        shap_values = [original_shap_values[-1]]
    else:
        shap_values = [
            np.array(
                [original_shap_values[i] for i in range(len(original_shap_values))]
            ).swapaxes(0, 1)
        ][0]

    normalisation_parameters_df = pd.read_csv(normalisation_parameters_path)

    with open(predictions_path, "rb") as handle:
        gt_over_time, predictions_over_time = pickle.load(handle)

    gt_over_time = gt_over_time.reshape(-1, n_time_steps)
    predictions_over_time = predictions_over_time.reshape(-1, n_time_steps)

    X_test, _ = ch.load(test_data_path)
    test_X_np = X_test[:, :, :, -1].astype("float32")

    features = X_test[0, 0, :, 2]
    avg_features = [f"avg_{i}" for i in features]
    min_features = [f"min_{i}" for i in features]
    max_features = [f"max_{i}" for i in features]
    aggregated_feature_names = (
        features.tolist() + avg_features + min_features + max_features + ["base_value"]
    )

    return {
        "shap_values": shap_values,
        "X_test": X_test,
        "test_X_np": test_X_np,
        "features": features,
        "aggregated_feature_names": aggregated_feature_names,
        "gt_over_time": gt_over_time,
        "predictions_over_time": predictions_over_time,
        "normalisation_parameters_df": normalisation_parameters_df,
        "cat_encoding_path": cat_encoding_path,
    }


def build_inference_plot_inputs(
    raw_inputs,
    n_time_steps=72,
    reverse_categorical_encoding=True,
    pool_hourly_split_values=True,
    only_keep_current_value_shap=True,
    shap_aggregation_func="sum",
    use_simplified_shap_values=True,
    smoothing_window=15,
    feature_to_english_name_correspondence_path=None,
):
    shap_values = raw_inputs["shap_values"]
    features = raw_inputs["features"]
    aggregated_feature_names = raw_inputs["aggregated_feature_names"]
    test_X_np = raw_inputs["test_X_np"]
    normalisation_parameters_df = raw_inputs["normalisation_parameters_df"]

    shap_values_df = pd.DataFrame()
    for ts in tqdm(range(n_time_steps), desc="Build SHAP table"):
        ts_shap_values_df = pd.DataFrame(
            data=shap_values[:, ts], columns=np.array(aggregated_feature_names)
        )
        ts_shap_values_df = ts_shap_values_df.reset_index()
        ts_shap_values_df.rename(columns={"index": "case_admission_id_idx"}, inplace=True)
        ts_shap_values_df = ts_shap_values_df.melt(
            id_vars="case_admission_id_idx", var_name="feature", value_name="shap_value"
        )
        ts_shap_values_df["time_step"] = ts
        shap_values_df = pd.concat((shap_values_df, ts_shap_values_df), ignore_index=True)

    if only_keep_current_value_shap:
        shap_values_df = shap_values_df[shap_values_df["feature"].isin(features)]

    feature_values_df = pd.DataFrame()
    for subj_idx in tqdm(range(test_X_np.shape[0]), desc="Build feature table"):
        subj_feature_values_df = pd.DataFrame(
            data=test_X_np[subj_idx, :, :], columns=np.array(features)
        )
        subj_feature_values_df = reverse_normalisation_for_subj(
            subj_feature_values_df, normalisation_parameters_df
        )
        subj_feature_values_df = subj_feature_values_df.reset_index()
        subj_feature_values_df.rename(columns={"index": "time_step"}, inplace=True)
        subj_feature_values_df["case_admission_id_idx"] = subj_idx
        subj_feature_values_df = subj_feature_values_df.melt(
            id_vars=["case_admission_id_idx", "time_step"],
            var_name="feature",
            value_name="feature_value",
        )
        feature_values_df = pd.concat(
            (feature_values_df, subj_feature_values_df), ignore_index=True
        )

    if reverse_categorical_encoding:
        cat_encoding_df = pd.read_csv(raw_inputs["cat_encoding_path"])
        for i in tqdm(range(len(cat_encoding_df)), desc="Decode categoricals"):
            cat_basename = cat_encoding_df.sample_label[i].lower().replace(" ", "_")
            cat_item_list = (
                cat_encoding_df.other_categories[i]
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(", ")
            )
            cat_item_list = [
                cat_basename + "_" + item.replace(" ", "_").lower()
                for item in cat_item_list
            ]
            for cat_item_idx, cat_item in enumerate(cat_item_list):
                feature_values_df.loc[
                    feature_values_df.feature == cat_item, "feature_value"
                ] *= cat_item_idx + 1
                feature_values_df.loc[
                    feature_values_df.feature == cat_item, "feature"
                ] = cat_encoding_df.sample_label[i]
                feature_values_df = (
                    feature_values_df.groupby(
                        ["case_admission_id_idx", "feature", "time_step"]
                    )
                    .sum()
                    .reset_index()
                )

                shap_values_df.loc[
                    shap_values_df.feature == cat_item, "feature"
                ] = cat_encoding_df.sample_label[i]
                if shap_aggregation_func == "sum":
                    shap_values_df = (
                        shap_values_df.groupby(
                            ["case_admission_id_idx", "feature", "time_step"]
                        )
                        .sum()
                        .reset_index()
                    )
                else:
                    shap_values_df = (
                        shap_values_df.groupby(
                            ["case_admission_id_idx", "feature", "time_step"]
                        )
                        .median()
                        .reset_index()
                    )

        cat_to_numerical_encoding = {
            "Prestroke disability (Rankin)": {0: 0, 1: 3, 2: 4, 3: 2, 4: 1, 5: 5},
            "categorical_onset_to_admission_time": {0: 2, 1: 1, 2: 0, 3: 3, 4: 5, 5: 4},
            "categorical_IVT": {0: 2, 1: 3, 2: 4, 3: 1, 4: 0},
            "categorical_IAT": {0: 1, 1: 2, 2: 3, 3: 0},
        }
        for cat_feature, cat_encoding in cat_to_numerical_encoding.items():
            mask = feature_values_df.feature == cat_feature
            feature_values_df.loc[mask, "feature_value"] = feature_values_df.loc[
                mask, "feature_value"
            ].map(cat_encoding)

    if pool_hourly_split_values:
        hourly_split_features = [
            "NIHSS",
            "systolic_blood_pressure",
            "diastolic_blood_pressure",
            "mean_blood_pressure",
            "heart_rate",
            "respiratory_rate",
            "temperature",
            "oxygen_saturation",
        ]
        for feature in tqdm(hourly_split_features, desc="Pool hourly feature splits"):
            shap_values_df.loc[shap_values_df.feature.str.contains(feature), "feature"] = (
                feature[0].upper() + feature[1:]
            ).replace("_", " ")
            if shap_aggregation_func == "median":
                shap_values_df = (
                    shap_values_df.groupby(
                        ["case_admission_id_idx", "feature", "time_step"]
                    )
                    .median()
                    .reset_index()
                )
            else:
                shap_values_df = (
                    shap_values_df.groupby(
                        ["case_admission_id_idx", "feature", "time_step"]
                    )
                    .sum()
                    .reset_index()
                )

            feature_values_df.loc[
                feature_values_df.feature.str.contains(feature), "feature"
            ] = (feature[0].upper() + feature[1:]).replace("_", " ")
            feature_values_df = (
                feature_values_df.groupby(["case_admission_id_idx", "feature", "time_step"])
                .median()
                .reset_index()
            )

    if feature_to_english_name_correspondence_path:
        correspondence = pd.read_excel(feature_to_english_name_correspondence_path)
        for feature in shap_values_df.feature.unique():
            if feature in correspondence.feature_name.values:
                shap_values_df.loc[shap_values_df.feature == feature, "feature"] = correspondence[
                    correspondence.feature_name == feature
                ].english_name.values[0]
        for feature in feature_values_df.feature.unique():
            if feature in correspondence.feature_name.values:
                feature_values_df.loc[
                    feature_values_df.feature == feature, "feature"
                ] = correspondence[correspondence.feature_name == feature].english_name.values[0]

    if use_simplified_shap_values:
        shap_values_over_time = []
        for ts in tqdm(range(n_time_steps), desc="Create simplified SHAP tensor"):
            subj_values_over_time = []
            for subj in range(len(test_X_np)):
                values = shap_values_df[
                    (shap_values_df.case_admission_id_idx == subj)
                    & (shap_values_df.time_step == ts)
                ].shap_value.values
                subj_values_over_time.append(values)
            shap_values_over_time.append(np.array(subj_values_over_time))
        shap_values_over_time = np.array(shap_values_over_time)
    else:
        shap_values_over_time = np.moveaxis(shap_values, 1, 0)

    reduced_feature_names = shap_values_df.feature.unique()

    smoothed_shap_values_over_time = []
    for subj_idx in range(shap_values_over_time.shape[1]):
        subj_smoothed = []
        for feature_idx in range(shap_values_over_time.shape[2]):
            subj_smoothed.append(
                smooth(shap_values_over_time[:, subj_idx, feature_idx], smoothing_window)
            )
        smoothed_shap_values_over_time.append(np.moveaxis(np.array(subj_smoothed), 0, -1))
    smoothed_shap_values_over_time = np.moveaxis(
        np.array(smoothed_shap_values_over_time), 0, 1
    )

    return {
        "predictions_over_time": raw_inputs["predictions_over_time"],
        "gt_over_time": raw_inputs["gt_over_time"],
        "feature_values_df": feature_values_df,
        "smoothed_shap_values_over_time": smoothed_shap_values_over_time,
        "shap_values_over_time": shap_values_over_time,
        "reduced_feature_names": reduced_feature_names,
        "raw_inputs": raw_inputs,
        "shap_values_df": shap_values_df,
    }


def load_preprocess_and_plot_subjects(
    subjects,
    shap_values_path,
    test_data_path,
    cat_encoding_path,
    normalisation_parameters_path,
    predictions_path,
    n_time_steps=72,
    only_last_timestep=False,
    reverse_categorical_encoding=True,
    pool_hourly_split_values=True,
    only_keep_current_value_shap=True,
    shap_aggregation_func="sum",
    use_simplified_shap_values=True,
    smoothing_window=15,
    feature_to_english_name_correspondence_path=None,
    plot_kwargs=None,
):
    if plot_kwargs is None:
        plot_kwargs = {}

    raw_inputs = load_inference_raw_inputs(
        shap_values_path=shap_values_path,
        test_data_path=test_data_path,
        cat_encoding_path=cat_encoding_path,
        normalisation_parameters_path=normalisation_parameters_path,
        predictions_path=predictions_path,
        n_time_steps=n_time_steps,
        only_last_timestep=only_last_timestep,
    )

    prepared = build_inference_plot_inputs(
        raw_inputs=raw_inputs,
        n_time_steps=n_time_steps,
        reverse_categorical_encoding=reverse_categorical_encoding,
        pool_hourly_split_values=pool_hourly_split_values,
        only_keep_current_value_shap=only_keep_current_value_shap,
        shap_aggregation_func=shap_aggregation_func,
        use_simplified_shap_values=use_simplified_shap_values,
        smoothing_window=smoothing_window,
        feature_to_english_name_correspondence_path=feature_to_english_name_correspondence_path,
    )

    figures_by_subject = {}
    for subj in subjects:
        fig = plot_joint_subject_explanation(
            subj=subj,
            predictions_over_time=prepared["predictions_over_time"],
            gt_over_time=prepared["gt_over_time"],
            feature_values_df=prepared["feature_values_df"],
            smoothed_shap_values_over_time=prepared["smoothed_shap_values_over_time"],
            shap_values_over_time=prepared["shap_values_over_time"],
            reduced_feature_names=prepared["reduced_feature_names"],
            use_simplified_shap_values=use_simplified_shap_values,
            **plot_kwargs,
        )
        figures_by_subject[subj] = fig

    return {
        "figures_by_subject": figures_by_subject,
        "prepared_inputs": prepared,
    }


def plot_joint_subject_explanation(
    subj,
    predictions_over_time,
    gt_over_time,
    feature_values_df,
    smoothed_shap_values_over_time,
    shap_values_over_time,
    reduced_feature_names,
    use_simplified_shap_values=True,
    threshold=0.04,
    n_features_selection=0,
    n_features=1,
    k=0.25,
    alpha=0.3,
    only_non_static_features=True,
    use_smoothed_shap_values=True,
    plot_ground_truth=True,
    display_significant_slopes=True,
    n_slope_steps=5,
    slope_threshold_multiplier=1.5,
    skip_label_at_zero=True,
    display_text_labels=True,
    display_legend=True,
    display_title=False,
    plot_NIHSS_continuously=True,
    ts_marker_level="shap",
    tick_label_size=13,
    label_font_size=16,
):
    subj_pred_over_ts = predictions_over_time[subj]
    subj_gt_over_ts = gt_over_time[subj]
    n_time_steps = len(subj_pred_over_ts)

    fig_joint, (ax_main, ax_features) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 12),
        gridspec_kw=dict(height_ratios=[2, 1], hspace=0.3),
    )

    if use_smoothed_shap_values:
        working_shap_values = smoothed_shap_values_over_time
    else:
        working_shap_values = shap_values_over_time

    significant_positive_timesteps = filter_consecutive_numbers(
        np.where(np.diff(subj_pred_over_ts) > threshold)[0]
    )
    significant_negative_timesteps = filter_consecutive_numbers(
        np.where(np.diff(subj_pred_over_ts) < -threshold)[0]
    )
    significant_timesteps = np.concatenate(
        (significant_positive_timesteps, significant_negative_timesteps)
    )

    non_norm_subj_df = (
        feature_values_df[feature_values_df.case_admission_id_idx == subj]
        .drop(columns=["case_admission_id_idx"])
        .pivot(index="time_step", columns="feature", values="feature_value")
    )

    if only_non_static_features:
        non_static_features = np.where(non_norm_subj_df.std() > 0.01)[0]
        if use_simplified_shap_values:
            non_static_features = np.where(
                np.isin(
                    reduced_feature_names,
                    np.array(non_norm_subj_df.std()[non_norm_subj_df.std() > 0.01].index),
                )
            )[0]
        selected_positive_features_by_impact = np.diff(
            working_shap_values[:, subj, non_static_features], axis=0
        )[significant_positive_timesteps].argmax(axis=1)
        selected_positive_features_by_impact = non_static_features[
            selected_positive_features_by_impact
        ]
        selected_negative_features_by_impact = np.diff(
            working_shap_values[:, subj, non_static_features], axis=0
        )[significant_negative_timesteps].argmin(axis=1)
        selected_negative_features_by_impact = non_static_features[
            selected_negative_features_by_impact
        ]
    else:
        non_static_features = np.arange(working_shap_values.shape[-1])
        selected_positive_features_by_impact = np.diff(
            working_shap_values[:, subj], axis=0
        )[significant_positive_timesteps].argmax(axis=1)
        selected_negative_features_by_impact = np.diff(
            working_shap_values[:, subj], axis=0
        )[significant_negative_timesteps].argmin(axis=1)

    selected_features_by_impact = np.concatenate(
        (selected_positive_features_by_impact, selected_negative_features_by_impact)
    )

    if display_significant_slopes:
        slope_threshold = slope_threshold_multiplier * threshold
        significant_positive_slope = filter_consecutive_numbers(
            set(
                np.where(
                    (
                        np.concatenate(
                            (
                                subj_pred_over_ts[n_slope_steps:],
                                np.zeros(n_slope_steps),
                            )
                        )
                        - subj_pred_over_ts
                    )[:-n_slope_steps]
                    > slope_threshold
                )[0]
            ).difference(set(significant_positive_timesteps))
        )

        significant_negative_slope = filter_consecutive_numbers(
            set(
                np.where(
                    (
                        np.concatenate(
                            (
                                subj_pred_over_ts[n_slope_steps:],
                                np.zeros(n_slope_steps),
                            )
                        )
                        - subj_pred_over_ts
                    )[:-n_slope_steps]
                    < -slope_threshold
                )[0]
            ).difference(set(significant_negative_timesteps))
        )

        delta_shap_by_features = np.concatenate(
            (
                working_shap_values[n_slope_steps:, subj, non_static_features],
                np.zeros((n_slope_steps, len(non_static_features))),
            )
        ) - working_shap_values[:, subj, non_static_features]

        selected_positive_features_by_slope = delta_shap_by_features[:-n_slope_steps][
            significant_positive_slope
        ].argmax(axis=1)
        selected_positive_features_by_slope = non_static_features[
            selected_positive_features_by_slope
        ]
        selected_negative_features_by_slope = delta_shap_by_features[:-n_slope_steps][
            significant_negative_slope
        ].argmin(axis=1)
        selected_negative_features_by_slope = non_static_features[
            selected_negative_features_by_slope
        ]

        selected_features_by_impact = np.concatenate(
            (
                selected_features_by_impact,
                selected_positive_features_by_slope,
                selected_negative_features_by_slope,
            )
        )
        significant_timesteps = np.concatenate(
            (
                significant_timesteps,
                significant_positive_slope,
                significant_negative_slope,
            )
        )
        selected_positive_features_by_impact = np.concatenate(
            (
                selected_positive_features_by_impact,
                selected_positive_features_by_slope,
            )
        )
        selected_negative_features_by_impact = np.concatenate(
            (
                selected_negative_features_by_impact,
                selected_negative_features_by_slope,
            )
        )

    if n_features_selection == 0:
        selected_positive_features = np.array([])
        selected_negative_features = np.array([])
    else:
        selected_positive_features = working_shap_values[-1, subj].argsort()[-n_features:][
            ::-1
        ]
        selected_negative_features = working_shap_values[-1, subj].argsort()[:n_features][
            ::-1
        ]

    selected_features = np.concatenate(
        (
            selected_positive_features,
            selected_positive_features_by_impact,
            selected_negative_features,
            selected_negative_features_by_impact,
        )
    ).astype(int)

    positive_color_palette = sns.color_palette(
        "mako",
        n_colors=len(
            set(np.concatenate((selected_positive_features, selected_positive_features_by_impact)))
        ),
    )
    negative_color_palette = sns.color_palette(
        "flare_r",
        n_colors=len(
            set(np.concatenate((selected_negative_features, selected_negative_features_by_impact)))
        ),
    )

    timestep_axis = np.array(range(n_time_steps))
    sns.lineplot(
        x=timestep_axis,
        y=subj_pred_over_ts,
        label="Predicted probability",
        linewidth=2,
        ax=ax_main,
    )

    if plot_ground_truth:
        changes_in_gt = np.diff(subj_gt_over_ts, prepend=0)
        change_pairs = list(zip(np.where(changes_in_gt == 1)[0], np.where(changes_in_gt == -1)[0]))
        for change_pair in change_pairs:
            ax_main.plot([change_pair[0], change_pair[1]], [0, 0], color="#7b002c", linewidth=10, alpha=0.8)
            ax_main.text(
                np.mean(change_pair),
                0 + 0.02,
                "6h to END",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=tick_label_size,
            )

    pos_baseline = subj_pred_over_ts
    neg_baseline = subj_pred_over_ts
    pos_count = 0
    neg_count = 0
    feature_color_dict = {}

    for feature in set(selected_features):
        subj_feature_shap_value_over_time = working_shap_values[:, subj, feature]
        positive_portion = subj_feature_shap_value_over_time > 0
        negative_portion = subj_feature_shap_value_over_time < 0

        pos_function = subj_feature_shap_value_over_time.copy()
        neg_function = subj_feature_shap_value_over_time.copy()
        pos_function[negative_portion] = 0
        neg_function[positive_portion] = 0

        if feature in selected_features_by_impact:
            important_ts_idx = np.where(selected_features_by_impact == feature)[0]
            if not np.logical_and(
                plot_NIHSS_continuously, reduced_feature_names[feature] == "NIHSS"
            ):
                pos_function[: significant_timesteps[important_ts_idx][0] + 1] = 0
                neg_function[: significant_timesteps[important_ts_idx][0] + 1] = 0

        if feature in selected_positive_features:
            feature_color = positive_color_palette[pos_count]
            pos_count += 1
        elif feature in selected_negative_features:
            feature_color = negative_color_palette[neg_count]
            neg_count += 1
        elif feature in selected_negative_features_by_impact:
            feature_color = negative_color_palette[neg_count]
            neg_count += 1
        elif feature in selected_positive_features_by_impact:
            feature_color = positive_color_palette[pos_count]
            pos_count += 1
        else:
            feature_color = "grey"
        feature_color_dict[feature] = feature_color

        if np.any(pos_function):
            positive_feature = pos_baseline + k * pos_function
            ax_main.fill_between(
                timestep_axis, pos_baseline, positive_feature, color=feature_color, alpha=alpha
            )
            pos_baseline = positive_feature

        if np.any(neg_function):
            negative_feature = neg_baseline + k * neg_function
            ax_main.fill_between(
                timestep_axis, negative_feature, neg_baseline, color=feature_color, alpha=alpha
            )
            neg_baseline = negative_feature

        ax_main.scatter(
            [],
            [],
            color=feature_color,
            alpha=alpha,
            label=reduced_feature_names[feature],
            marker="s",
            s=200,
        )

    for feature in set(selected_features_by_impact):
        important_ts_idx = np.where(selected_features_by_impact == feature)[0]
        for ts_idx in important_ts_idx:
            if skip_label_at_zero and significant_timesteps[ts_idx] == 0:
                continue
            if subj_pred_over_ts[significant_timesteps[ts_idx]] > subj_pred_over_ts[
                significant_timesteps[ts_idx] + 1
            ]:
                marker = "v"
                if ts_marker_level == "shap":
                    marker_y_level = pos_baseline[significant_timesteps[ts_idx]] + 0.005
                else:
                    marker_y_level = subj_pred_over_ts[significant_timesteps[ts_idx]] + 0.005
                text_y_level = marker_y_level + 0.01
            else:
                marker = "^"
                if ts_marker_level == "shap":
                    marker_y_level = neg_baseline[significant_timesteps[ts_idx]] - 0.005
                else:
                    marker_y_level = subj_pred_over_ts[significant_timesteps[ts_idx]] - 0.005
                text_y_level = marker_y_level - 0.015

            ax_main.scatter(
                significant_timesteps[ts_idx],
                marker_y_level,
                color=feature_color_dict[feature],
                s=100,
                marker=marker,
                alpha=1,
                edgecolors="white",
            )

            if display_text_labels:
                if marker == "v":
                    ax_main.text(
                        significant_timesteps[ts_idx] + 0.01,
                        text_y_level,
                        reduced_feature_names[feature],
                        fontsize=12,
                        color="black",
                        rotation=45,
                        ha="left",
                        va="bottom",
                    )
                else:
                    ax_main.text(
                        significant_timesteps[ts_idx] - 0.01,
                        text_y_level,
                        reduced_feature_names[feature],
                        fontsize=12,
                        color="black",
                    )

    if display_title:
        ax_main.set_title(f"Predictions for subject {subj} of test set along time", fontsize=20)

    ax_main.set_xlabel("Time from admission (hours)", fontsize=label_font_size)
    ax_main.set_ylabel("Probability of END", fontsize=label_font_size)
    ax_main.tick_params(axis="both", labelsize=tick_label_size)

    if display_legend:
        legend_markers, legend_labels = ax_main.get_legend_handles_labels()

        shap_shades_markers = legend_markers[1:]
        shap_shades_labels = legend_labels[1:]
        legend_markers = [legend_markers[0]]
        legend_labels = [legend_labels[0]]

        ts_marker_down = Line2D(
            [0], [0], marker="v", linestyle="", color="grey", markersize=7, alpha=0.8
        )
        ts_marker_up = Line2D(
            [0], [0], marker="^", linestyle="", color="grey", markersize=7, alpha=0.8
        )
        ts_label = "Positive / Negative impact on inflection of prediction"
        legend_markers.append((ts_marker_up, ts_marker_down))
        legend_labels.append(ts_label)

        legend_markers.append("")
        legend_labels.append("")
        legend_markers.append("Weight & direction of influence on model prediction")
        legend_labels.append("")

        legend_markers += shap_shades_markers
        legend_labels += shap_shades_labels

        ax_main.legend(
            legend_markers,
            legend_labels,
            fontsize=label_font_size,
            title="Influence on model prediction",
            title_fontsize=label_font_size,
            handler_map={
                tuple: HandlerTuple(ndivide=1),
                str: LegendTitle({"fontsize": label_font_size}),
            },
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    n_features_small = len(set(selected_features_by_impact))

    if n_features_small > 0:
        cols = min(4, n_features_small)
        rows = (n_features_small + cols - 1) // cols

        gs_nested = ax_features.figure.add_gridspec(
            rows,
            cols,
            left=ax_features.get_position().x0,
            right=ax_features.get_position().x1,
            bottom=ax_features.get_position().y0,
            top=ax_features.get_position().y1,
            hspace=0.4,
            wspace=0.3,
        )

        ax_features.remove()

        for idx, feature in enumerate(set(selected_features_by_impact)):
            row = idx // cols
            col = idx % cols
            ax_small = fig_joint.add_subplot(gs_nested[row, col])

            feature_name = reduced_feature_names[feature]
            feature_color = feature_color_dict[feature]
            feature_data = non_norm_subj_df[feature_name]

            ax_small.plot(timestep_axis, feature_data, color=feature_color, linewidth=2)
            ax_small.fill_between(timestep_axis, feature_data, alpha=0.3, color=feature_color)

            important_ts_idx = np.where(selected_features_by_impact == feature)[0]
            for ts_idx in important_ts_idx:
                timestep = significant_timesteps[ts_idx]
                ax_small.scatter(
                    timestep,
                    feature_data.iloc[timestep],
                    color=feature_color,
                    s=60,
                    zorder=5,
                    edgecolors="white",
                    linewidth=1,
                )

            ax_small.set_title(
                feature_name,
                fontsize=tick_label_size,
                color=feature_color,
                weight="bold",
            )
            ax_small.set_xlim(0, n_time_steps)
            ax_small.spines["top"].set_visible(False)
            ax_small.spines["right"].set_visible(False)

            y_min, y_max = feature_data.min(), feature_data.max()
            if y_min == y_max:
                y_ticks = [y_min]
            else:
                y_ticks = [y_min, y_max]
            ax_small.set_yticks(y_ticks)
            ax_small.tick_params(labelsize=tick_label_size - 2)
            ax_small.set_ylim(y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))

            if row == rows - 1:
                ax_small.set_xlabel("Time (h)", fontsize=tick_label_size - 1)
            else:
                ax_small.set_xticklabels([])
    else:
        ax_features.text(
            0.5,
            0.5,
            "No significant feature changes detected",
            transform=ax_features.transAxes,
            ha="center",
            va="center",
            fontsize=label_font_size,
            style="italic",
        )
        ax_features.set_xlim(0, 1)
        ax_features.set_ylim(0, 1)
        ax_features.axis("off")

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    return fig_joint


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Load inference artifacts and generate explanation plots for selected subjects."
    )
    parser.add_argument("--shap-values-path", required=True, help="Path to SHAP values pickle.")
    parser.add_argument("--test-data-path", required=True, help="Path to test data .pth file.")
    parser.add_argument(
        "--cat-encoding-path",
        required=True,
        help="Path to categorical encoding CSV.",
    )
    parser.add_argument(
        "--normalisation-parameters-path",
        required=True,
        help="Path to normalization parameters CSV.",
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="Path to predictions pickle (gt, pred).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        required=True,
        help="Subject indices to plot (e.g. --subjects 3 10 42).",
    )
    parser.add_argument("--n-time-steps", type=int, default=72)
    parser.add_argument("--only-last-timestep", action="store_true")

    parser.add_argument(
        "--reverse-categorical-encoding",
        dest="reverse_categorical_encoding",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-reverse-categorical-encoding",
        dest="reverse_categorical_encoding",
        action="store_false",
    )
    parser.add_argument(
        "--pool-hourly-split-values",
        dest="pool_hourly_split_values",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-pool-hourly-split-values",
        dest="pool_hourly_split_values",
        action="store_false",
    )
    parser.add_argument(
        "--only-keep-current-value-shap",
        dest="only_keep_current_value_shap",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--keep-all-aggregated-shap",
        dest="only_keep_current_value_shap",
        action="store_false",
    )
    parser.add_argument(
        "--shap-aggregation-func",
        choices=["sum", "median"],
        default="sum",
    )
    parser.add_argument(
        "--use-simplified-shap-values",
        dest="use_simplified_shap_values",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-simplified-shap-values",
        dest="use_simplified_shap_values",
        action="store_false",
    )
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument(
        "--feature-to-english-name-correspondence-path",
        default=None,
        help="Optional path to feature name mapping Excel file.",
    )

    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--n-features-selection", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=1)
    parser.add_argument("--display-legend", action="store_true", default=False)
    parser.add_argument("--display-text-labels", action="store_true", default=False)
    parser.add_argument("--display-title", action="store_true", default=False)
    parser.add_argument("--plot-ground-truth", action="store_true", default=True)
    parser.add_argument("--no-plot-ground-truth", dest="plot_ground_truth", action="store_false")

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory; if provided, each subject plot is saved as PNG.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display generated figures interactively.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    plot_kwargs = {
        "threshold": args.threshold,
        "n_features_selection": args.n_features_selection,
        "n_features": args.n_features,
        "display_legend": args.display_legend,
        "display_text_labels": args.display_text_labels,
        "display_title": args.display_title,
        "plot_ground_truth": args.plot_ground_truth,
    }

    result = load_preprocess_and_plot_subjects(
        subjects=args.subjects,
        shap_values_path=args.shap_values_path,
        test_data_path=args.test_data_path,
        cat_encoding_path=args.cat_encoding_path,
        normalisation_parameters_path=args.normalisation_parameters_path,
        predictions_path=args.predictions_path,
        n_time_steps=args.n_time_steps,
        only_last_timestep=args.only_last_timestep,
        reverse_categorical_encoding=args.reverse_categorical_encoding,
        pool_hourly_split_values=args.pool_hourly_split_values,
        only_keep_current_value_shap=args.only_keep_current_value_shap,
        shap_aggregation_func=args.shap_aggregation_func,
        use_simplified_shap_values=args.use_simplified_shap_values,
        smoothing_window=args.smoothing_window,
        feature_to_english_name_correspondence_path=args.feature_to_english_name_correspondence_path,
        plot_kwargs=plot_kwargs,
    )

    figures_by_subject = result["figures_by_subject"]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for subj, fig in figures_by_subject.items():
            out_path = os.path.join(args.output_dir, f"subject_{subj}_inference_plot.png")
            fig.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
            print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    else:
        for fig in figures_by_subject.values():
            plt.close(fig)


if __name__ == "__main__":
    main()
