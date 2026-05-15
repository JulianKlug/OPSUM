"""Generate waffle-chart figure visualising END event rates across derivation and test sets."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Add project root so we can import plot_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from prediction.short_term_outcome_prediction.figures.plot_config import (
    setup_theme, COLOR_XGB, STANDALONE_TICK, STANDALONE_LABEL,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'end_event_rates.csv')
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

COLOR_POS = COLOR_XGB          # positive / END class
COLOR_NEG = '#B0B0B0'          # negative / no-END class
DATASET_LABELS = {'Derivation': 'Derivation cohort', 'Test': 'Test cohort'}


def load_data():
    return pd.read_csv(CSV_PATH, index_col='Dataset')


def _draw_waffle(ax, pct_value, cmap, scale_pct):
    """Draw a 10x10 waffle grid. scale_pct is the value each cell represents."""
    n_colored = max(1, round(pct_value / scale_pct))
    n_colored = min(n_colored, 100)
    grid = np.zeros((10, 10))
    count = 0
    for r in range(9, -1, -1):
        for c in range(10):
            if count < n_colored:
                grid[r, c] = 1
                count += 1
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    for edge in range(11):
        ax.axhline(edge - 0.5, color='white', linewidth=2)
        ax.axvline(edge - 0.5, color='white', linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])


def make_figure(df):
    setup_theme(figsize=(10, 8))
    fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    cmap = ListedColormap([COLOR_NEG, COLOR_POS])

    datasets = ['Derivation', 'Test']

    # --- Row 1 (A): Positive timestep rate (each cell = 1%) ---
    for ax, ds in zip(axes[0], datasets):
        pct = df.loc[ds, 'Proportion positive'] * 100
        _draw_waffle(ax, pct, cmap, scale_pct=1.0)
        ax.set_title(f'{DATASET_LABELS[ds]}\n{pct:.2f}% positive timesteps',
                     fontsize=STANDALONE_LABEL, pad=8)

    axes[0, 0].set_ylabel('1 square = 1%', fontsize=STANDALONE_TICK, labelpad=10)

    # Row label A
    axes[0, 0].text(-0.15, 1.12, 'A', transform=axes[0, 0].transAxes,
                    fontsize=STANDALONE_LABEL + 3, fontweight='bold', va='top')

    # --- Row 2 (B): Event rate (each cell = 0.1%) ---
    for ax, ds in zip(axes[1], datasets):
        rate = df.loc[ds, 'Event rate (events/patient-hour)'] * 100
        _draw_waffle(ax, rate, cmap, scale_pct=0.1)
        n_events = int(df.loc[ds, 'Distinct END events'])
        hours = int(df.loc[ds, 'Total patient-hours'])
        ax.set_title(f'{DATASET_LABELS[ds]}\n{rate:.2f}% event rate ({n_events} events / {hours:,} pt-hrs)',
                     fontsize=STANDALONE_TICK + 1, pad=8)

    axes[1, 0].set_ylabel('1 square = 0.1%', fontsize=STANDALONE_TICK, labelpad=10)

    # Row label B
    axes[1, 0].text(-0.15, 1.18, 'B', transform=axes[1, 0].transAxes,
                    fontsize=STANDALONE_LABEL + 3, fontweight='bold', va='top')

    # Legend
    legend_elements = [Patch(facecolor=COLOR_POS, label='END'),
                       Patch(facecolor=COLOR_NEG, label='No END')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=STANDALONE_TICK, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    fig = make_figure(df)
    path = os.path.join(OUT_DIR, 'end_event_rates_waffle.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
