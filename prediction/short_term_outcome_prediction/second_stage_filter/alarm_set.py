"""
Alarm rows: the (patient, hour) rows the first-stage XGB flags.

The second stage only ever sees these rows. The threshold is set for a high
recall so that the second stage has room to remove false alarms.
"""
import numpy as np
import pandas as pd

LABEL_COLUMN = 'label'
SCORE_COLUMN = 'xgb_score'
WEIGHT_COLUMN = 'sampling_weight'


def threshold_at_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """
    Highest threshold whose alarms (score >= threshold) catch at least target_recall of the positives.

    e.g. positive scores [0.9, 0.7, 0.4, 0.2], target 0.75 -> 0.4 (3 of 4 caught)
    """
    positive_scores = np.sort(scores[labels == 1])[::-1]
    if len(positive_scores) == 0:
        raise ValueError('No positive rows: recall is undefined.')

    n_positives_to_catch = int(np.ceil(target_recall * len(positive_scores)))

    return float(positive_scores[max(n_positives_to_catch, 1) - 1])


def select_alarm_rows(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Keep the rows XGB flags. The original index is kept: it addresses the feature and SHAP matrices."""
    return rows[rows[SCORE_COLUMN] >= threshold]


def sample_alarm_rows(alarm_rows: pd.DataFrame, max_rows: int, positive_share: float, seed: int) -> pd.DataFrame:
    """
    Draw the alarm rows the LLM will score, enriched in true alarms.

    True alarms are rare, so a uniform sample would hold too few to compare
    models. Positives and negatives are sampled separately; each sampled row
    carries the number of alarm rows it stands for, which makes weighted
    metrics unbiased for the full alarm set.

    e.g. 100 positives all kept -> weight 1; 5000 negatives, 500 kept -> weight 10
    """
    rng = np.random.RandomState(seed)
    positive_budget = int(round(max_rows * positive_share))

    sampled_strata = []
    for label, budget in ((1, positive_budget), (0, max_rows - positive_budget)):
        stratum = alarm_rows[alarm_rows[LABEL_COLUMN] == label]
        n_sampled = min(len(stratum), budget)
        if n_sampled == 0:
            continue

        sampled = stratum.loc[rng.choice(stratum.index, size=n_sampled, replace=False)].copy()
        sampled[WEIGHT_COLUMN] = len(stratum) / n_sampled
        sampled_strata.append(sampled)

    return pd.concat(sampled_strata).sort_index()
