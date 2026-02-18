"""SHAP beeswarm plot for top overall features (pooled across aggregation prefixes)."""

import argparse

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colormath.color_objects import LabColor
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot_config import (
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure,
)
from prediction.utils.visualisation_helper_functions import hex_to_rgb_color, create_palette
from shap_utils import (
    load_shap_and_test_data, build_feature_names,
    select_top_predictors, prepare_individual_obs_df, format_feature_name,
)


def _plot_beeswarm(ax, selected_df, selected_features,
                   tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL,
                   reverse_outcome_direction=True, xlim=None,
                   row_height=0.4, alpha=0.8,
                   plot_colorbar=True, plot_legend=True,
                   xlabel='SHAP Value \n(impact on model output)'):
    """Core beeswarm plotting logic."""
    start_rgb = hex_to_rgb_color('#012D98')
    end_rgb = hex_to_rgb_color('#f61067')
    palette = create_palette(start_rgb, end_rgb, 50, LabColor, extrapolation_length=1)

    for pos, feature in enumerate(selected_features[::-1]):
        feature_data = selected_df[selected_df.feature == feature]
        shaps = feature_data.shap_value.values
        values = np.array(feature_data.feature_value.values, dtype=np.float64)

        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        N = len(shaps)
        if N == 0:
            continue

        # Density-based jittering (vectorised)
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        sorted_inds = np.argsort(quant + np.random.default_rng(42).standard_normal(N) * 1e-6)
        sorted_quant = quant[sorted_inds]

        # Layer index within each bin (vectorised cumcount)
        bin_boundaries = np.concatenate([[True], sorted_quant[1:] != sorted_quant[:-1]])
        bin_start_positions = np.cumsum(bin_boundaries) - 1
        bin_start_idx = np.zeros(bin_start_positions[-1] + 1, dtype=np.int64)
        bin_start_idx[bin_start_positions[bin_boundaries]] = np.where(bin_boundaries)[0]
        layers = np.arange(N) - bin_start_idx[bin_start_positions]

        # Alternating y-offsets
        ys_sorted = np.ceil(layers / 2).astype(float) * ((layers % 2) * 2 - 1)
        ys = np.empty(N)
        ys[sorted_inds] = ys_sorted
        ys *= 0.9 * (row_height / (np.max(np.abs(ys)) + 1))

        # Trim color range
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(values, 1)
            vmax = np.nanpercentile(values, 99)
            if vmin == vmax:
                vmin = np.min(values)
                vmax = np.max(values)
        if vmin > vmax:
            vmin = vmax

        cvals = values.astype(np.float64).copy()
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
        cvals[cvals_imp > vmax] = vmax
        cvals[cvals_imp < vmin] = vmin

        point_size = 4 if N > 10000 else (8 if N > 1000 else 16)
        point_alpha = 0.3 if N > 10000 else (0.5 if N > 1000 else alpha)
        ax.scatter(shaps, pos + ys,
                   cmap=ListedColormap(palette), vmin=vmin, vmax=vmax, s=point_size,
                   c=cvals, alpha=point_alpha, linewidth=0,
                   zorder=3, rasterized=len(shaps) > 500)

    # Colorbar
    if plot_colorbar:
        m = cm.ScalarMappable(cmap=ListedColormap(palette))
        m.set_array([0, 1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.5)
        fig = ax.get_figure()
        cb = fig.colorbar(m, ticks=[0, 1], aspect=10, shrink=0.2, ax=cax)
        cb.set_ticklabels(['Low', 'High'])
        cb.ax.tick_params(labelsize=tick_size, length=0)
        cb.set_label('Feature value', size=label_size, backgroundcolor="white")
        cb.ax.yaxis.set_label_position('left')
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        cax.grid(False)
        for spine in cax.spines.values():
            spine.set_visible(False)
        cax.set_xticks([])
        cax.set_yticks([])

    # Legend
    if plot_legend:
        single_dot = mlines.Line2D([], [], color=palette[len(palette) // 2], marker='.',
                                    linestyle='None', markersize=10)
        ax.legend([single_dot], ['Single observation\n(patient \u00d7 timepoint)'],
                  title='SHAP/Feature values', fontsize=tick_size, title_fontsize=label_size,
                  handler_map={tuple: HandlerTuple(ndivide=None)},
                  loc='lower right', frameon=True)

    # Axes formatting
    axis_color = "#333333"
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)

    yticklabels = list(selected_features[::-1])
    ax.set_yticks(range(len(selected_features)))
    ax.set_yticklabels(yticklabels, fontsize=label_size)
    ax.tick_params('y', length=20, width=0.5, which='major')
    ax.tick_params('x', labelsize=tick_size)
    ax.set_ylim(-1, len(selected_features))
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.grid(color='white', axis='y')

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])

    # Direction labels on x-axis
    x_ticks_coordinates = ax.get_xticks()
    x_ticks_labels = [f'{x:.1f}' for x in x_ticks_coordinates]
    if reverse_outcome_direction:
        x_ticks_labels[0] = 'Toward better\noutcome'
        x_ticks_labels[-1] = 'Toward worse\noutcome'
    else:
        x_ticks_labels[0] = 'Toward worse\noutcome'
        x_ticks_labels[-1] = 'Toward better\noutcome'
    ax.set_xticks(x_ticks_coordinates)
    ax.set_xticklabels(x_ticks_labels)

    return ax


def plot_shap_beeswarm(ax, shap_path, test_data_path,
                       feature_names_path=None, n_top_features=10,
                       add_lag_features=False, add_rolling_features=False,
                       peak_per_subject=False,
                       tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL,
                       reverse_outcome_direction=True, xlim=None,
                       plot_colorbar=True, plot_legend=True,
                       _precomputed=None):
    """Plot SHAP beeswarm for top features.

    By default each dot = one patient x timepoint x aggregation variant.
    With peak_per_subject=True, each dot = one patient (observation with
    the highest |SHAP| across all timepoints and aggregation variants).

    Features are pooled by base name (aggregation, hourly, and BP prefixes stripped).
    Top features are selected by total |SHAP| across all observations.
    """
    if _precomputed is not None:
        df = _precomputed['df']
        top_features = _precomputed['top_features']
    else:
        shap_values, X_test_raw, _ = load_shap_and_test_data(shap_path, test_data_path)
        feature_names, _ = build_feature_names(
            X_test_raw, add_lag_features=add_lag_features,
            add_rolling_features=add_rolling_features)

        top_features = select_top_predictors(shap_values, feature_names, n_top_features)
        df = prepare_individual_obs_df(shap_values, X_test_raw, feature_names,
                                       top_features, peak_per_subject=peak_per_subject)

        # Map base names to readable display names
        if feature_names_path is not None:
            correspondence = pd.read_excel(feature_names_path)
        else:
            correspondence = None
        name_map = {f: format_feature_name(f, _correspondence=correspondence)
                    for f in top_features}
        df['feature'] = df['feature'].map(name_map)
        top_features = np.array([name_map[f] for f in top_features])

    _plot_beeswarm(ax, df, top_features,
                   tick_size=tick_size, label_size=label_size,
                   reverse_outcome_direction=reverse_outcome_direction,
                   xlim=xlim, plot_colorbar=plot_colorbar, plot_legend=plot_legend)
    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot SHAP beeswarm for top features')
    parser.add_argument('--shap_path', required=True,
                        help='Path to tree_explainer_shap_values_over_ts.pkl')
    parser.add_argument('--test_data_path', required=True,
                        help='Path to test data .pth file')
    parser.add_argument('--feature_names_path', default=None,
                        help='Path to feature_name_to_english_name_correspondence.xlsx')
    parser.add_argument('--n_top_features', type=int, default=10,
                        help='Number of top features to display')
    parser.add_argument('--add_lag_features', action='store_true',
                        help='Whether lag features were used in aggregation')
    parser.add_argument('--add_rolling_features', action='store_true',
                        help='Whether rolling features were used in aggregation')
    parser.add_argument('--peak', action='store_true',
                        help='One dot per subject: observation with highest |SHAP|')
    parser.add_argument('--reverse_outcome_direction', action='store_true', default=True,
                        help='If set, positive SHAP = worse outcome (default: True)')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots()
    plot_shap_beeswarm(
        ax,
        shap_path=args.shap_path,
        test_data_path=args.test_data_path,
        feature_names_path=args.feature_names_path,
        n_top_features=args.n_top_features,
        add_lag_features=args.add_lag_features,
        add_rolling_features=args.add_rolling_features,
        peak_per_subject=args.peak,
        reverse_outcome_direction=args.reverse_outcome_direction,
    )
    suffix = 'shap_beeswarm_peak' if args.peak else 'shap_beeswarm'
    save_figure(fig, args.output_dir, suffix)
    plt.close(fig)


if __name__ == '__main__':
    main()
