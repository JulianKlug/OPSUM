"""SHAP beeswarm plot for imaging features."""

import argparse

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colormath.color_objects import LabColor
from matplotlib.colors import ListedColormap

from plot_config import (
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure,
)
from plot_shap_beeswarm import _plot_beeswarm
from prediction.utils.visualisation_helper_functions import hex_to_rgb_color, create_palette
from shap_utils import (
    load_shap_and_test_data, build_feature_names,
    prepare_individual_obs_df, strip_to_base_name,
)

# Fixed order: grouped by type, decreasing threshold within each group
IMAGING_FEATURE_ORDER = [
    'cbf_lt_38', 'cbf_lt_34', 'cbf_lt_30', 'cbf_lt_20',
    'cbv_lt_42', 'cbv_lt_38', 'cbv_lt_34',
    'tmax_gt_10', 'tmax_gt_8', 'tmax_gt_6', 'tmax_gt_4',
]

IMAGING_DISPLAY_NAMES = {
    'cbf_lt_20': 'CBF < 20%', 'cbf_lt_30': 'CBF < 30%',
    'cbf_lt_34': 'CBF < 34%', 'cbf_lt_38': 'CBF < 38%',
    'cbv_lt_34': 'CBV < 34%', 'cbv_lt_38': 'CBV < 38%',
    'cbv_lt_42': 'CBV < 42%',
    'tmax_gt_4': 'Tmax > 4s', 'tmax_gt_6': 'Tmax > 6s',
    'tmax_gt_8': 'Tmax > 8s', 'tmax_gt_10': 'Tmax > 10s',
}


def plot_imaging_beeswarm(ax, shap_path, test_data_path,
                          add_lag_features=False, add_rolling_features=False,
                          tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL,
                          reverse_outcome_direction=True,
                          _precomputed=None):
    """Beeswarm plot for imaging features, ordered by decreasing threshold."""
    if _precomputed is not None:
        df = _precomputed['df']
        ordered_display = _precomputed['ordered']
    else:
        shap_values, X_test_raw, _ = load_shap_and_test_data(shap_path, test_data_path)
        feature_names, _ = build_feature_names(
            X_test_raw, add_lag_features=add_lag_features,
            add_rolling_features=add_rolling_features)

        ordered = np.array(IMAGING_FEATURE_ORDER)
        df = prepare_individual_obs_df(
            shap_values, X_test_raw, feature_names,
            ordered, peak_per_subject=True)

        name_map = {f: IMAGING_DISPLAY_NAMES[f] for f in ordered}
        df['feature'] = df['feature'].map(name_map)
        ordered_display = np.array([name_map[f] for f in ordered])

    _plot_beeswarm(ax, df, ordered_display,
                   tick_size=tick_size, label_size=label_size,
                   reverse_outcome_direction=reverse_outcome_direction,
                   plot_colorbar=False, plot_legend=False)

    # Colorbar outside the plot on the right
    fig = ax.get_figure()
    start_rgb = hex_to_rgb_color('#012D98')
    end_rgb = hex_to_rgb_color('#f61067')
    palette = create_palette(start_rgb, end_rgb, 50, LabColor, extrapolation_length=1)
    m = cm.ScalarMappable(cmap=ListedColormap(palette))
    m.set_array([0, 1])
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cb = fig.colorbar(m, cax=cax, ticks=[0, 1])
    cb.set_ticklabels(['Low', 'High'])
    cb.ax.tick_params(labelsize=tick_size, length=0)
    cb.set_label('Feature value', size=label_size)
    cb.outline.set_visible(False)

    return ax


def main():
    parser = argparse.ArgumentParser(
        description='SHAP beeswarm plot for imaging features')
    parser.add_argument('--shap_path', required=True,
                        help='Path to tree_explainer_shap_values_over_ts.pkl')
    parser.add_argument('--test_data_path', required=True,
                        help='Path to test data .pth file')
    parser.add_argument('--add_lag_features', action='store_true')
    parser.add_argument('--add_rolling_features', action='store_true')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_imaging_beeswarm(
        ax,
        shap_path=args.shap_path,
        test_data_path=args.test_data_path,
        add_lag_features=args.add_lag_features,
        add_rolling_features=args.add_rolling_features,
    )
    fig.subplots_adjust(right=0.88)
    save_figure(fig, args.output_dir, 'shap_imaging_beeswarm')
    plt.close(fig)


if __name__ == '__main__':
    main()
