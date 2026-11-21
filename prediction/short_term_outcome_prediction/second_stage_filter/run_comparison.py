"""
Runs the comparison of second-stage arms on out-of-fold alarm rows.

    stacker ──> alarm_rows.csv, step1_comparison.csv   arms A (XGB score) and B (stacker), all alarm rows
    sample  ──> llm_sample_dev.csv, llm_sample_eval.csv rows the LLM reviews
    (llm_filter.py reviews the samples)
    final   ──> final_comparison.csv, report.md        arms A, B, C (LLM), D (blend), on the reviewed eval rows

Fold 0 patients are the development set (prompt design). Reported numbers
come from the patients of the other folds only.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prediction.short_term_outcome_prediction.second_stage_filter.alarm_set import (
    LABEL_COLUMN, SCORE_COLUMN, WEIGHT_COLUMN, sample_alarm_rows, select_alarm_rows, threshold_at_recall)
from prediction.short_term_outcome_prediction.second_stage_filter.evaluation import PPV_METRIC_TEMPLATE, compare_arms
from prediction.short_term_outcome_prediction.second_stage_filter.llm_filter import REVIEWS_FILE
from prediction.short_term_outcome_prediction.second_stage_filter.out_of_fold_xgb import OutOfFoldStore
from prediction.short_term_outcome_prediction.second_stage_filter.stacker import (
    build_stacker_inputs, cross_fit_scores, logit, new_linear_blender, new_tree_stacker)

STAGE1_TARGET_RECALL = 0.80
OVERALL_RECALL_LEVELS = (0.6, 0.4, 0.2)
DEV_FOLD = 0
LLM_SAMPLE_POSITIVE_SHARE = 0.5
SEED = 42

PATIENT_COLUMN = 'patient_id'
STACKER_COLUMN = 'stacker_score'
LLM_COLUMN = 'llm_probability'
BLEND_WITHOUT_LLM_COLUMN = 'blend_xgb_stacker'
BLEND_WITH_LLM_COLUMN = 'blend_xgb_stacker_llm'

ALARM_ROWS_FILE = 'alarm_rows.csv'
STEP1_COMPARISON_FILE = 'step1_comparison.csv'
SAMPLE_FILE_TEMPLATE = 'llm_sample_{subset}.csv'
FINAL_COMPARISON_FILE = 'final_comparison.csv'
LLM_INCREMENT_FILE = 'llm_increment_comparison.csv'
STEP1_OVERALL_FILE = 'step1_overall_auprc.csv'
FINAL_OVERALL_FILE = 'final_overall_auprc.csv'
REPORT_FILE = 'report.md'


def within_alarm_recall_levels(all_rows: pd.DataFrame, alarm_rows: pd.DataFrame):
    """
    Recall the second stage must keep among alarm rows to reach each overall recall level.

    e.g. the first stage catches 80% of positives; an overall recall of 0.6 needs 0.6 / 0.8 = 75% of those kept
    """
    stage1_recall = alarm_rows[LABEL_COLUMN].sum() / all_rows[LABEL_COLUMN].sum()

    return [level / stage1_recall for level in OVERALL_RECALL_LEVELS], stage1_recall


def name_metrics_by_overall_recall(comparison: pd.DataFrame, recall_levels) -> pd.DataFrame:
    """
    Metrics are computed at a recall among alarm rows; readers care about recall among all positives.

    e.g. 'ppv_at_recall_0.75' (of alarm rows) -> 'ppv_at_recall_0.60' (overall)
    """
    overall_names = {PPV_METRIC_TEMPLATE.format(recall=within_alarms): PPV_METRIC_TEMPLATE.format(recall=overall)
                     for within_alarms, overall in zip(recall_levels, OVERALL_RECALL_LEVELS)}

    return comparison.replace({'metric': overall_names})


def compare_overall_auprc(all_rows: pd.DataFrame, scored_alarm_rows: pd.DataFrame, arm_columns, reference_arm: str,
                          weight_column: str = None) -> pd.DataFrame:
    """
    AUPRC of each two-stage system over all patient-hours, comparable with the first-stage AUPRC.

    Rows below the alarm threshold keep their XGB score; alarm rows are ranked above
    them, in the order given by the arm:

        non-alarm row: xgb score (< threshold)      alarm row: 1 + arm score

    With the XGB score as arm this is the XGB ranking itself.
    """
    non_alarm_rows = all_rows[all_rows[SCORE_COLUMN] < scored_alarm_rows[SCORE_COLUMN].min()]
    overall = pd.DataFrame({
        LABEL_COLUMN: np.concatenate([non_alarm_rows[LABEL_COLUMN].values, scored_alarm_rows[LABEL_COLUMN].values]),
        PATIENT_COLUMN: np.concatenate([non_alarm_rows[PATIENT_COLUMN].values, scored_alarm_rows[PATIENT_COLUMN].values]),
    })

    # Every non-alarm row is present once; sampled alarm rows stand for several
    alarm_weights = np.ones(len(scored_alarm_rows)) if weight_column is None else scored_alarm_rows[weight_column].values
    overall[WEIGHT_COLUMN] = np.concatenate([np.ones(len(non_alarm_rows)), alarm_weights])

    for arm in arm_columns:
        overall[arm] = np.concatenate([non_alarm_rows[SCORE_COLUMN].values, 1 + scored_alarm_rows[arm].values])

    return compare_arms(overall, arm_columns, reference_arm=reference_arm, recall_levels=[], label_column=LABEL_COLUMN,
                        patient_column=PATIENT_COLUMN, weight_column=WEIGHT_COLUMN, seed=SEED)


def is_eval(rows: pd.DataFrame) -> pd.Series:
    return rows.fold != DEV_FOLD


def to_markdown_table(table: pd.DataFrame) -> str:
    """Plain markdown table; pandas' own needs the tabulate package, absent from the project environment."""
    header = '| ' + ' | '.join(table.columns) + ' |'
    separator = '|' + '---|' * len(table.columns)
    body = ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in table.values]

    return '\n'.join([header, separator] + body)


def run_stacker(output_dir: str):
    """Step 1: can a model trained on alarm rows, given features and SHAP, out-rank the XGB score?"""
    store = OutOfFoldStore(output_dir)
    rows = store.rows

    print(f'Pooled out-of-fold: AUROC {roc_auc_score(rows[LABEL_COLUMN], rows[SCORE_COLUMN]):.4f}, '
          f'AUPRC {average_precision_score(rows[LABEL_COLUMN], rows[SCORE_COLUMN]):.4f}, '
          f'prevalence {rows[LABEL_COLUMN].mean():.4f}')

    threshold = threshold_at_recall(rows[LABEL_COLUMN].values, rows[SCORE_COLUMN].values, STAGE1_TARGET_RECALL)
    alarm_rows = select_alarm_rows(rows, threshold).copy()
    print(f'Alarm threshold {threshold:.4f}: {len(alarm_rows)} alarm rows ({len(alarm_rows) / len(rows):.1%} of rows), '
          f'PPV {alarm_rows[LABEL_COLUMN].mean():.4f}')

    inputs = build_stacker_inputs(store.features(alarm_rows.index), store.shap_values(alarm_rows.index),
                                  alarm_rows[SCORE_COLUMN].values)
    labels = alarm_rows[LABEL_COLUMN].values
    patient_ids = alarm_rows[PATIENT_COLUMN].values

    alarm_rows[STACKER_COLUMN] = cross_fit_scores(new_tree_stacker, inputs, labels, patient_ids, seed=SEED)

    # Leak check: trained on shuffled labels, an honest cross-fit cannot beat the prevalence
    shuffled_labels = np.random.RandomState(SEED).permutation(labels)
    shuffled_scores = cross_fit_scores(new_tree_stacker, inputs, shuffled_labels, patient_ids, seed=SEED)
    print(f'Shuffled-label control: AUPRC {average_precision_score(shuffled_labels, shuffled_scores):.4f} '
          f'vs prevalence {shuffled_labels.mean():.4f}')

    alarm_rows.rename_axis('row_id').to_csv(os.path.join(output_dir, ALARM_ROWS_FILE))

    eval_alarm_rows = alarm_rows[is_eval(alarm_rows)]
    recall_levels, stage1_recall = within_alarm_recall_levels(rows[is_eval(rows)], eval_alarm_rows)
    print(f'Eval patients: {len(eval_alarm_rows)} alarm rows, first-stage recall {stage1_recall:.3f}')

    comparison = compare_arms(eval_alarm_rows, [SCORE_COLUMN, STACKER_COLUMN], reference_arm=SCORE_COLUMN,
                              recall_levels=recall_levels, label_column=LABEL_COLUMN, patient_column=PATIENT_COLUMN,
                              seed=SEED)
    comparison = name_metrics_by_overall_recall(comparison, recall_levels)
    comparison.to_csv(os.path.join(output_dir, STEP1_COMPARISON_FILE), index=False)
    print(comparison.round(4).to_string(index=False))

    overall = compare_overall_auprc(rows[is_eval(rows)], eval_alarm_rows, [SCORE_COLUMN, STACKER_COLUMN], SCORE_COLUMN)
    overall.to_csv(os.path.join(output_dir, STEP1_OVERALL_FILE), index=False)
    print('Overall AUPRC, all eval patient-hours:')
    print(overall.round(4).to_string(index=False))


def load_alarm_rows(output_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(output_dir, ALARM_ROWS_FILE), index_col='row_id',
                       dtype={'case_admission_id': str, PATIENT_COLUMN: str})


def run_sample(output_dir: str, max_dev_rows: int, max_eval_rows: int):
    """Choose the alarm rows the LLM reviews, separately for development and evaluation patients."""
    alarm_rows = load_alarm_rows(output_dir)

    for subset, subset_rows, max_rows in (('dev', alarm_rows[~is_eval(alarm_rows)], max_dev_rows),
                                          ('eval', alarm_rows[is_eval(alarm_rows)], max_eval_rows)):
        sample = sample_alarm_rows(subset_rows, max_rows, LLM_SAMPLE_POSITIVE_SHARE, seed=SEED)

        # Shuffled: an interrupted LLM run still leaves a usable random subset
        sample = sample.sample(frac=1, random_state=SEED)
        sample.to_csv(os.path.join(output_dir, SAMPLE_FILE_TEMPLATE.format(subset=subset)))

        print(f'{subset}: {len(sample)} of {len(subset_rows)} alarm rows sampled, {int(sample[LABEL_COLUMN].sum())} positive; '
              f'true-alarm rate of all {subset} alarm rows {subset_rows[LABEL_COLUMN].mean():.4f}')


def load_llm_probabilities(output_dir: str, model: str, prompt_version: str) -> pd.DataFrame:
    """Reviews of one model and prompt version, one line per (case, hour)."""
    with open(os.path.join(output_dir, REVIEWS_FILE)) as f:
        reviews = pd.DataFrame([json.loads(line) for line in f])

    reviews = reviews[reviews.key.str.startswith(f'{prompt_version}|{model}|')]

    return reviews[['case_admission_id', 'hour', 'probability']].rename(columns={'probability': LLM_COLUMN})


def run_final(output_dir: str, model: str, prompt_version: str):
    """Steps 2 and 3: the LLM alone, and what it adds once blended with the other scores."""
    store = OutOfFoldStore(output_dir)
    sample = pd.read_csv(os.path.join(output_dir, SAMPLE_FILE_TEMPLATE.format(subset='eval')),
                         dtype={'case_admission_id': str, PATIENT_COLUMN: str})

    reviewed = sample.merge(load_llm_probabilities(output_dir, model, prompt_version), on=['case_admission_id', 'hour'])
    print(f'{len(reviewed)} of {len(sample)} sampled eval rows reviewed by {model} ({prompt_version}), '
          f'{int(reviewed[LABEL_COLUMN].sum())} positive, {reviewed[LLM_COLUMN].nunique()} distinct LLM scores')

    # A partial LLM run reviewed a random part of each stratum: rescale the weights to the full alarm set
    alarm_rows = load_alarm_rows(output_dir)
    eval_alarm_rows = alarm_rows[is_eval(alarm_rows)]
    for label in (0, 1):
        in_stratum = reviewed[LABEL_COLUMN] == label
        reviewed.loc[in_stratum, WEIGHT_COLUMN] = (eval_alarm_rows[LABEL_COLUMN] == label).sum() / in_stratum.sum()

    labels = reviewed[LABEL_COLUMN].values
    patient_ids = reviewed[PATIENT_COLUMN].values
    weights = reviewed[WEIGHT_COLUMN].values
    first_and_second_stage = np.column_stack([logit(reviewed[SCORE_COLUMN].values), logit(reviewed[STACKER_COLUMN].values)])

    reviewed[BLEND_WITHOUT_LLM_COLUMN] = cross_fit_scores(new_linear_blender, first_and_second_stage, labels,
                                                          patient_ids, weights, seed=SEED)
    reviewed[BLEND_WITH_LLM_COLUMN] = cross_fit_scores(
        new_linear_blender, np.column_stack([first_and_second_stage, reviewed[LLM_COLUMN].values]), labels,
        patient_ids, weights, seed=SEED)

    recall_levels, stage1_recall = within_alarm_recall_levels(store.rows[is_eval(store.rows)], eval_alarm_rows)
    shared_arguments = dict(recall_levels=recall_levels, label_column=LABEL_COLUMN, patient_column=PATIENT_COLUMN,
                            weight_column=WEIGHT_COLUMN, seed=SEED)

    comparison = compare_arms(reviewed, [SCORE_COLUMN, STACKER_COLUMN, LLM_COLUMN, BLEND_WITHOUT_LLM_COLUMN,
                                         BLEND_WITH_LLM_COLUMN], reference_arm=SCORE_COLUMN, **shared_arguments)
    comparison = name_metrics_by_overall_recall(comparison, recall_levels)
    comparison.to_csv(os.path.join(output_dir, FINAL_COMPARISON_FILE), index=False)

    # The cleanest test of the LLM: the same blend with and without its score
    llm_increment = compare_arms(reviewed, [BLEND_WITHOUT_LLM_COLUMN, BLEND_WITH_LLM_COLUMN],
                                 reference_arm=BLEND_WITHOUT_LLM_COLUMN, **shared_arguments)
    llm_increment = name_metrics_by_overall_recall(llm_increment, recall_levels)
    llm_increment.to_csv(os.path.join(output_dir, LLM_INCREMENT_FILE), index=False)

    all_arms = [SCORE_COLUMN, STACKER_COLUMN, LLM_COLUMN, BLEND_WITHOUT_LLM_COLUMN, BLEND_WITH_LLM_COLUMN]
    overall = compare_overall_auprc(store.rows[is_eval(store.rows)], reviewed, all_arms, SCORE_COLUMN, WEIGHT_COLUMN)
    overall.to_csv(os.path.join(output_dir, FINAL_OVERALL_FILE), index=False)

    report = '\n\n'.join([
        '# Second-stage alarm filter: comparison on out-of-fold alarm rows',
        f'Model: {model}, prompt {prompt_version}. {len(reviewed)} reviewed alarm rows of {len(eval_alarm_rows)} '
        f'(eval patients, weighted to the full alarm set), {int(labels.sum())} true alarms. '
        f'First-stage recall {stage1_recall:.3f}. PPV columns are at overall recall {OVERALL_RECALL_LEVELS}.',
        '## All arms vs XGB score (raised threshold)',
        to_markdown_table(comparison.round(4)),
        '## Overall AUPRC of each two-stage system, all eval patient-hours',
        to_markdown_table(overall.round(4)),
        '## LLM increment: blend with vs without the LLM score',
        to_markdown_table(llm_increment.round(4)),
    ])
    with open(os.path.join(output_dir, REPORT_FILE), 'w') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['stacker', 'sample', 'final'])
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--max_dev_rows', type=int, default=300)
    parser.add_argument('--max_eval_rows', type=int, default=3000)
    parser.add_argument('-m', '--model', type=str, default='qwen3.8:27b')
    parser.add_argument('-p', '--prompt_version', type=str, default='v1')
    args = parser.parse_args()

    if args.stage == 'stacker':
        run_stacker(args.output_dir)
    elif args.stage == 'sample':
        run_sample(args.output_dir, args.max_dev_rows, args.max_eval_rows)
    else:
        run_final(args.output_dir, args.model, args.prompt_version)
