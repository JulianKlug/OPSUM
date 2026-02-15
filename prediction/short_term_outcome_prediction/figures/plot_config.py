"""Shared aesthetics, colors, theme, fonts, and save helpers for publication figures."""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

# --- Colors ---
COLOR_XGB = '#012D98'
COLOR_LR = '#049b9a'

# --- Font sizes ---
STANDALONE_TICK = 11
STANDALONE_LABEL = 13

PUB_TICK = 6
PUB_LABEL = 7
PUB_SUBPLOT_NUM = 9
PUB_SUPTITLE = 10

# --- Unit conversion ---
CM = 1 / 2.54


def setup_theme(figsize=(10, 10)):
    """Apply the standard OPSUM publication theme."""
    custom_params = {
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.figsize': figsize,
    }
    sns.set_theme(style='whitegrid', rc=custom_params, context='paper', font_scale=1)


def save_figure(fig, output_dir, stem):
    """Save figure as SVG (1200 dpi) and TIFF (900 dpi)."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, f'{stem}.svg'),
        bbox_inches='tight', format='svg', dpi=1200,
    )
    fig.savefig(
        os.path.join(output_dir, f'{stem}.tiff'),
        bbox_inches='tight', format='tiff', dpi=900,
    )
    print(f'Saved {stem}.svg and {stem}.tiff to {output_dir}')


def make_ci_legend(ax, colors, labels):
    """Add grouped CI-band patch to an existing legend using HandlerTuple."""
    legend_markers, legend_labels = ax.get_legend_handles_labels()

    patches = tuple(mpatches.Patch(color=c, alpha=0.2) for c in colors)
    legend_markers.append(patches)
    legend_labels.append('95% CI')

    ax.legend(
        legend_markers, legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    return ax
