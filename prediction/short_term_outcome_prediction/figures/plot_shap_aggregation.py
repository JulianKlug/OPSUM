"""SHAP aggregation importance plots: bar + strip and violin by aggregation type."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_config import (
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure,
)
from shap_utils import (
    AGG_DISPLAY_NAMES,
    load_shap_and_test_data, build_feature_names,
    get_prefix, peak_shap_over_time,
)


def _prepare_aggregation_data(shap_path, test_data_path,
                               add_lag_features=False, add_rolling_features=False,
                               _precomputed=None):
    """Compute per-subject, per-aggregation-type mean |SHAP|.

    Returns:
        agg_df: DataFrame with columns [subject, agg_type, mean_abs_shap]
        order: array of agg_type names sorted by descending mean |SHAP|
    """
    if _precomputed is not None:
        shap_values = _precomputed['shap_values']
        feature_names = _precomputed['feature_names']
    else:
        shap_values, X_test_raw, _ = load_shap_and_test_data(shap_path, test_data_path)
        feature_names, _ = build_feature_names(
            X_test_raw, add_lag_features=add_lag_features,
            add_rolling_features=add_rolling_features)

    # Max SHAP over time, drop base_value column
    max_shap, _ = peak_shap_over_time(shap_values)  # (n_subj, n_feat+1)
    all_names = feature_names + ['base_value']
    shap_df = pd.DataFrame(max_shap, columns=all_names).drop(columns=['base_value', 'timestep_idx'])

    # Assign each feature to its aggregation type
    prefix_map = pd.Series({name: get_prefix(name) for name in shap_df.columns})

    # Mean |SHAP| per subject per aggregation type
    agg_records = []
    for subj_i in range(shap_df.shape[0]):
        row = shap_df.iloc[subj_i]
        for agg_type in prefix_map.unique():
            cols = prefix_map[prefix_map == agg_type].index
            agg_records.append({
                'subject': subj_i,
                'agg_type': AGG_DISPLAY_NAMES.get(agg_type, agg_type),
                'mean_abs_shap': np.abs(row[cols]).mean(),
            })
    agg_df = pd.DataFrame(agg_records)

    # Order by overall mean
    order = (agg_df.groupby('agg_type')['mean_abs_shap']
             .mean()
             .sort_values(ascending=False)
             .index.values)

    return agg_df, order


def plot_shap_aggregation_bar(ax, shap_path, test_data_path,
                              add_lag_features=False, add_rolling_features=False,
                              tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL,
                              color=None, _precomputed=None):
    """Horizontal bar chart of mean |SHAP| per aggregation type with overlaid strip plot.

    Each bar is the grand mean across subjects; individual dots show per-subject values.
    """
    agg_df, order = _prepare_aggregation_data(
        shap_path, test_data_path,
        add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features,
        _precomputed=_precomputed)

    bar_color = color or '#012D98'

    # Grand mean per aggregation type
    means = agg_df.groupby('agg_type')['mean_abs_shap'].mean()
    means = means.reindex(order[::-1])  # bottom-to-top for horizontal bar

    # Bar chart
    y_pos = np.arange(len(means))
    ax.barh(y_pos, means.values, color=bar_color, alpha=0.35, height=0.6, zorder=2,
            label='Mean')

    # Strip plot (individual subjects)
    for i, agg_type in enumerate(means.index):
        vals = agg_df[agg_df.agg_type == agg_type]['mean_abs_shap'].values
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=len(vals))
        ax.scatter(vals, i + jitter, color=bar_color, alpha=0.4, s=4,
                   zorder=3, rasterized=len(vals) > 300,
                   label='Per-patient mean' if i == 0 else None)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(means.index, fontsize=label_size)
    ax.set_xlabel('Mean absolute SHAP per feature', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(-0.6, len(means) - 0.4)
    ax.legend(fontsize=tick_size, loc='lower right')

    return ax


def plot_shap_aggregation_violin(ax, shap_path, test_data_path,
                                 add_lag_features=False, add_rolling_features=False,
                                 tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL,
                                 color=None, _precomputed=None):
    """Violin plot of per-subject mean |SHAP| by aggregation type."""
    agg_df, order = _prepare_aggregation_data(
        shap_path, test_data_path,
        add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features,
        _precomputed=_precomputed)

    violin_color = color or '#012D98'

    # Order bottom-to-top (least -> most important going up)
    order_reversed = list(order[::-1])
    agg_df['agg_type'] = pd.Categorical(agg_df['agg_type'],
                                         categories=order_reversed, ordered=True)

    parts = ax.violinplot(
        [agg_df[agg_df.agg_type == t]['mean_abs_shap'].values for t in order_reversed],
        positions=np.arange(len(order_reversed)),
        vert=False, showmedians=True, showextrema=False,
    )
    for body in parts['bodies']:
        body.set_facecolor(violin_color)
        body.set_alpha(0.4)
    parts['cmedians'].set_color(violin_color)
    parts['cmedians'].set_linewidth(1.5)

    # Formatting
    ax.set_yticks(np.arange(len(order_reversed)))
    ax.set_yticklabels(order_reversed, fontsize=label_size)
    ax.set_xlabel('Mean absolute SHAP per feature', fontsize=label_size)
    ax.tick_params('x', labelsize=tick_size)
    ax.tick_params('y', labelsize=tick_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(-0.6, len(order_reversed) - 0.4)

    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot SHAP aggregation importance')
    parser.add_argument('--shap_path', required=True,
                        help='Path to tree_explainer_shap_values_over_ts.pkl')
    parser.add_argument('--test_data_path', required=True,
                        help='Path to test data .pth file')
    parser.add_argument('--add_lag_features', action='store_true',
                        help='Whether lag features were used in aggregation')
    parser.add_argument('--add_rolling_features', action='store_true',
                        help='Whether rolling features were used in aggregation')
    parser.add_argument('--type', choices=['bar', 'violin', 'both'], default='both',
                        help='Which aggregation plot(s) to generate (default: both)')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))

    common_kw = dict(
        shap_path=args.shap_path,
        test_data_path=args.test_data_path,
        add_lag_features=args.add_lag_features,
        add_rolling_features=args.add_rolling_features,
    )

    if args.type in ('bar', 'both'):
        fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
        plot_shap_aggregation_bar(ax_bar, **common_kw)
        fig_bar.tight_layout()
        save_figure(fig_bar, args.output_dir, 'shap_aggregation_bar')
        plt.close(fig_bar)

    if args.type in ('violin', 'both'):
        fig_vio, ax_vio = plt.subplots(figsize=(12, 8))
        plot_shap_aggregation_violin(ax_vio, **common_kw)
        ax_vio.set_title('Importance by aggregation type', fontsize=STANDALONE_LABEL)
        fig_vio.tight_layout()
        save_figure(fig_vio, args.output_dir, 'shap_aggregation_violin')
        plt.close(fig_vio)


if __name__ == '__main__':
    main()
