"""Poster-friendly SHAP beeswarm: top 4 predictors only, big fonts, no decorations.

Designed as a small subfigure on a poster. Uses the existing peak-per-subject
plotting logic from plot_shap_beeswarm but strips the colorbar, the legend, and
trims to top-4 features. Tick / label sizes are bumped so it stays legible at
poster scale even when the rendered figure is small.
"""
import argparse
import os

import matplotlib.pyplot as plt

from plot_config import setup_theme, save_figure
from plot_shap_beeswarm import plot_shap_beeswarm


def main():
    parser = argparse.ArgumentParser(description='Poster SHAP beeswarm — top 4 only')
    parser.add_argument('--shap_path', required=True)
    parser.add_argument('--test_data_path', required=True)
    parser.add_argument('--feature_names_path', default=None)
    parser.add_argument('--n_top_features', type=int, default=4,
                        help='Number of top features to show (default: 4)')
    parser.add_argument('--add_lag_features', action='store_true')
    parser.add_argument('--add_rolling_features', action='store_true')
    parser.add_argument('--tick_size', type=int, default=22,
                        help='Tick label font size (default: 22)')
    parser.add_argument('--label_size', type=int, default=26,
                        help='Axis label font size (default: 26)')
    parser.add_argument('--figsize', type=float, nargs=2, default=(9, 5),
                        help='Figure size in inches (default: 9 5)')
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()

    setup_theme(figsize=tuple(args.figsize))
    fig, ax = plt.subplots()

    plot_shap_beeswarm(
        ax,
        shap_path=args.shap_path,
        test_data_path=args.test_data_path,
        feature_names_path=args.feature_names_path,
        n_top_features=args.n_top_features,
        add_lag_features=args.add_lag_features,
        add_rolling_features=args.add_rolling_features,
        peak_per_subject=True,
        tick_size=args.tick_size,
        label_size=args.label_size,
        reverse_outcome_direction=True,
        plot_colorbar=False,
        plot_legend=False,
    )

    # Trim chrome. A small poster subfigure should be just the swarm + labels.
    ax.set_xlabel('')
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)

    # Replace the numeric x-axis ticks with just the two directional anchors
    # at the actual data extremes — no numbers, no "details".
    xmin, xmax = ax.get_xlim()
    ax.set_xticks([xmin, xmax])
    ax.set_xticklabels(['Toward better\noutcome', 'Toward worse\noutcome'],
                       fontsize=args.label_size)
    # Keep numeric x-grid off — user asked for no details
    ax.grid(False, axis='x')

    # A little extra top padding so the densest swarm (NIHSS) doesn't touch
    # the figure edge.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.3)

    # Y-tick labels (feature names) are the message — heavier weight.
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')

    fig.tight_layout()
    save_figure(fig, args.output_dir, f'shap_beeswarm_poster_top{args.n_top_features}')
    plt.close(fig)


if __name__ == '__main__':
    main()
