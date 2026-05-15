"""SHAP heatmap for time-invariant categorical features."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_config import (
    STANDALONE_TICK, STANDALONE_LABEL,
    setup_theme, save_figure,
)
from shap_utils import (
    load_shap_and_test_data, build_feature_names,
    prepare_ti_category_shap,
)


def plot_shap_ti_heatmap(ax, shap_path, test_data_path, cat_encoding_path,
                          feature_names_path,
                          add_lag_features=False, add_rolling_features=False,
                          tick_size=STANDALONE_TICK, label_size=STANDALONE_LABEL,
                          _precomputed=None):
    """Heatmap of mean SHAP per category level for TI categorical features.

    Rows = parent features, columns = category levels, cell color = mean SHAP.
    """
    import seaborn as sns

    if _precomputed is not None and 'cat_df' in _precomputed:
        cat_df = _precomputed['cat_df']
    else:
        if _precomputed is not None:
            shap_values = _precomputed['shap_values']
            X_test_raw = _precomputed['X_test_raw']
            feature_names = _precomputed['feature_names']
        else:
            shap_values, X_test_raw, _ = load_shap_and_test_data(shap_path, test_data_path)
            feature_names, _ = build_feature_names(
                X_test_raw, add_lag_features=add_lag_features,
                add_rolling_features=add_rolling_features)
        cat_df = prepare_ti_category_shap(
            shap_values, X_test_raw, feature_names,
            cat_encoding_path, feature_names_path)

    if cat_df.empty:
        return ax

    # Sort parents by total |SHAP|
    parent_importance = (cat_df.groupby('parent')['mean_shap']
                         .apply(lambda x: np.abs(x).sum())
                         .sort_values(ascending=False))
    parents = list(parent_importance.index)

    max_levels = cat_df.groupby('parent').size().max()

    # Build matrix and annotations
    matrix = np.full((len(parents), max_levels), np.nan)
    annot = np.empty((len(parents), max_levels), dtype=object)
    annot[:] = ''

    for i, parent in enumerate(parents):
        levels = cat_df[cat_df.parent == parent].sort_values('mean_shap', ascending=False)
        for j, (_, row) in enumerate(levels.iterrows()):
            val = row['mean_shap']
            matrix[i, j] = val
            level_label = row['level'].replace('_', ' ')
            if val != 0 and abs(val) < 0.001:
                if val < 0:
                    annot[i, j] = f"{level_label}\n(> -0.001)"
                else:
                    annot[i, j] = f"{level_label}\n(< 0.001)"
            else:
                annot[i, j] = f"{level_label}\n({val:.3f})"

    mask = np.isnan(matrix)

    # Symmetric color limits
    vmax = np.nanmax(np.abs(matrix))

    sns.heatmap(matrix, ax=ax, mask=mask, cmap='RdBu_r', center=0,
                vmin=-vmax, vmax=vmax,
                annot=annot, fmt='', annot_kws={'fontsize': tick_size - 1},
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Mean SHAP value', 'shrink': 0.8})

    ax.set_yticks(np.arange(len(parents)) + 0.5)
    ax.set_yticklabels(parents, fontsize=label_size, rotation=0)
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.cbar_ax = ax.collections[0].colorbar.ax
    ax.cbar_ax.tick_params(labelsize=tick_size)
    ax.cbar_ax.set_ylabel('Mean SHAP value', fontsize=label_size)

    return ax


def main():
    parser = argparse.ArgumentParser(description='Plot SHAP heatmap for TI categorical features')
    parser.add_argument('--shap_path', required=True,
                        help='Path to tree_explainer_shap_values_over_ts.pkl')
    parser.add_argument('--test_data_path', required=True,
                        help='Path to test data .pth file')
    parser.add_argument('--cat_encoding_path', required=True,
                        help='Path to categorical_variable_encoding.csv')
    parser.add_argument('--feature_names_path', required=True,
                        help='Path to feature_name_to_english_name_correspondence.xlsx')
    parser.add_argument('--add_lag_features', action='store_true',
                        help='Whether lag features were used in aggregation')
    parser.add_argument('--add_rolling_features', action='store_true',
                        help='Whether rolling features were used in aggregation')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for figures')
    args = parser.parse_args()

    setup_theme(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_shap_ti_heatmap(
        ax,
        shap_path=args.shap_path,
        test_data_path=args.test_data_path,
        cat_encoding_path=args.cat_encoding_path,
        feature_names_path=args.feature_names_path,
        add_lag_features=args.add_lag_features,
        add_rolling_features=args.add_rolling_features,
    )
    fig.tight_layout()
    save_figure(fig, args.output_dir, 'shap_ti_heatmap')
    plt.close(fig)


if __name__ == '__main__':
    main()
