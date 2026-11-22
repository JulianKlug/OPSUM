"""
ROC and PR curves: time-series XGB END model vs admission-only baseline models (reviewer response).

    rows    task    (hourly: END in next 6 h at each hour; patient: any END)
    columns cohort x curve (internal ROC, internal PR, external ROC, external PR)
    legend  model names (shared); per panel AUROC / AUPRC with 95% CI from baseline_performance.csv
    lr_literature is omitted: identical to lr_selected (no extra variable passed stability selection)
    xgb_end is omitted at patient level: its max-before-END aggregation is not a fair patient-level score
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from prediction.short_term_outcome_prediction.figures.plot_config import \
    COLOR_XGB, COLOR_LR, PUB_TICK, PUB_LABEL, PUB_SUBPLOT_NUM, CM, setup_theme, save_figure

MODEL_STYLES = {
    'xgb_end': ('XGB END (time series)', COLOR_XGB, '-'),
    'lr_selected': ('LR, admission variables', COLOR_LR, '-'),
    'xgb_baseline': ('XGB, all admission variables', '#d1495b', '--'),
}
TASK_TITLES = {'hourly': 'Hourly (END within 6 h)', 'patient': 'Patient level (any END)'}
COHORT_TITLES = {'internal_test': 'Internal test', 'external_test': 'External test (MIMIC)'}
FIGURE_SIZE = (18 * CM, 10 * CM)
LINE_WIDTH = 0.9
ANNOTATION_LINE_SPACING = 0.075
MODELS_BY_TASK = {'hourly': list(MODEL_STYLES), 'patient': ['lr_selected', 'xgb_baseline']}


def _ci_text(performance: pd.DataFrame, cohort: str, task: str, model: str, metric: str) -> str:
    row = performance[(performance.cohort == cohort) & (performance.task == task)
                      & (performance.model == model) & (performance.metric == metric)].iloc[0]
    return f'{row.value:.2f} ({row.ci_low:.2f}–{row.ci_high:.2f})'


def _plot_curves(ax_roc, ax_pr, preds: pd.DataFrame, performance: pd.DataFrame, cohort: str, task: str):
    styles = {m: MODEL_STYLES[m] for m in MODELS_BY_TASK[task]}
    for model, (label, color, style) in styles.items():
        fpr, tpr, _ = roc_curve(preds.y_true, preds[model])
        precision, recall, _ = precision_recall_curve(preds.y_true, preds[model])
        ax_roc.plot(fpr, tpr, color=color, linestyle=style, linewidth=LINE_WIDTH, label=label)
        ax_pr.plot(recall, precision, color=color, linestyle=style, linewidth=LINE_WIDTH, label=label)

    # metric values stacked in model colours: ROC bottom right, PR top right
    for i, (model, (_, color, _)) in enumerate(styles.items()):
        ax_roc.text(0.98, 0.04 + ANNOTATION_LINE_SPACING * (len(styles) - 1 - i),
                    _ci_text(performance, cohort, task, model, 'auroc'), color=color, ha='right',
                    fontsize=PUB_TICK - 1, transform=ax_roc.transAxes)
        ax_pr.text(0.98, 0.94 - ANNOTATION_LINE_SPACING * i, _ci_text(performance, cohort, task, model, 'auprc'),
                   color=color, ha='right', va='top', fontsize=PUB_TICK - 1, transform=ax_pr.transAxes)

    ax_roc.plot([0, 1], [0, 1], color='black', linestyle=':', linewidth=0.5)
    ax_pr.axhline(preds.y_true.mean(), color='black', linestyle=':', linewidth=0.5)
    ax_roc.set(xlabel='1 - specificity', ylabel='Sensitivity', xlim=(0, 1), ylim=(0, 1))
    ax_pr.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    for ax in (ax_roc, ax_pr):
        ax.tick_params(labelsize=PUB_TICK)
        ax.xaxis.label.set_size(PUB_LABEL)
        ax.yaxis.label.set_size(PUB_LABEL)


def plot(results_dir: str, output_dir: str):
    performance = pd.read_csv(os.path.join(results_dir, 'baseline_performance.csv'))
    setup_theme(FIGURE_SIZE)
    fig, axes = plt.subplots(len(TASK_TITLES), 2 * len(COHORT_TITLES), figsize=FIGURE_SIZE)

    for row, task in enumerate(TASK_TITLES):
        preds = pd.read_csv(os.path.join(results_dir, f'{task}_predictions.csv'))
        for i, cohort in enumerate(COHORT_TITLES):
            ax_roc, ax_pr = axes[row, 2 * i], axes[row, 2 * i + 1]
            _plot_curves(ax_roc, ax_pr, preds[preds.cohort == cohort], performance, cohort, task)
            ax_roc.set_title(f'{COHORT_TITLES[cohort]}: ROC', fontsize=PUB_LABEL)
            ax_pr.set_title(f'{COHORT_TITLES[cohort]}: PR', fontsize=PUB_LABEL)
        # row label: task
        axes[row, 0].annotate(TASK_TITLES[task], xy=(-0.45, 0.5), xycoords='axes fraction', rotation=90,
                              ha='center', va='center', fontsize=PUB_LABEL, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.text(-0.3, 1.08, chr(ord('a') + i), transform=ax.transAxes, fontsize=PUB_SUBPLOT_NUM, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(MODEL_STYLES), fontsize=PUB_TICK, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, output_dir, 'baseline_vs_xgb_roc_pr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results_dir', required=True, help='baseline_model_evaluation output directory')
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()
    plot(args.results_dir, args.output_dir)
