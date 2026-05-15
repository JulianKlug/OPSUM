"""Combined SHAP figure: beeswarm peak + aggregation bar.

Layouts:
  --layout side   -> side-by-side (A left, B right)  [default]
  --layout stacked -> stacked (A top, B bottom)
"""

import argparse

import matplotlib.pyplot as plt

from plot_config import (
    PUB_TICK, PUB_LABEL, PUB_SUBPLOT_NUM,
    setup_theme, save_figure, CM,
)
from plot_shap_beeswarm import plot_shap_beeswarm
from plot_shap_aggregation import plot_shap_aggregation_bar


def _build_figure(layout):
    """Create figure and axes for the requested layout."""
    if layout == 'stacked':
        fig, (ax_a, ax_b) = plt.subplots(
            2, 1, figsize=(18 * CM, 20 * CM),
            gridspec_kw={'height_ratios': [1.2, 1]},
            constrained_layout=True,
        )
    else:  # side
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(18 * CM, 14 * CM),
            gridspec_kw={'width_ratios': [1.2, 1]},
            constrained_layout=True,
        )
    return fig, ax_a, ax_b


def main():
    parser = argparse.ArgumentParser(
        description='Combined SHAP figure: beeswarm peak + aggregation bar')
    parser.add_argument('--shap_path', required=True,
                        help='Path to tree_explainer_shap_values_over_ts.pkl')
    parser.add_argument('--test_data_path', required=True,
                        help='Path to test data .pth file')
    parser.add_argument('--feature_names_path', default=None,
                        help='Path to feature_name_to_english_name_correspondence.xlsx')
    parser.add_argument('--n_top_features', type=int, default=10,
                        help='Number of top features for beeswarm')
    parser.add_argument('--add_lag_features', action='store_true')
    parser.add_argument('--add_rolling_features', action='store_true')
    parser.add_argument('--layout', choices=['side', 'stacked'], default='side',
                        help='Panel layout (default: side)')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_theme()

    common_kw = dict(
        shap_path=args.shap_path,
        test_data_path=args.test_data_path,
        add_lag_features=args.add_lag_features,
        add_rolling_features=args.add_rolling_features,
    )

    for layout in ([args.layout] if args.layout != 'both' else ['side', 'stacked']):
        fig, ax_a, ax_b = _build_figure(layout)

        # A. Beeswarm peak
        plot_shap_beeswarm(
            ax_a, **common_kw,
            feature_names_path=args.feature_names_path,
            n_top_features=args.n_top_features,
            peak_per_subject=True,
            tick_size=PUB_TICK, label_size=PUB_LABEL,
            reverse_outcome_direction=True,
            plot_colorbar=True, plot_legend=False,
        )

        # B. Aggregation bar
        plot_shap_aggregation_bar(
            ax_b, **common_kw,
            tick_size=PUB_TICK, label_size=PUB_LABEL,
        )

        # Panel labels
        for label, ax in zip(['A.', 'B.'], [ax_a, ax_b]):
            ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                    fontsize=PUB_SUBPLOT_NUM, fontweight='bold', va='top')

        suffix = f'shap_combined_{layout}'
        save_figure(fig, args.output_dir, suffix)
        plt.close(fig)


if __name__ == '__main__':
    main()
