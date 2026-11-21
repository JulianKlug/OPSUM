"""
Compare second-stage arms on the same alarm rows.

Every arm is a score per alarm row. Arms are compared on ranking quality
(AUPRC) and on PPV at matched recall, with confidence intervals from
resampling patients (hours of one patient are not independent).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

AUPRC_METRIC = 'auprc'
PPV_METRIC_TEMPLATE = 'ppv_at_recall_{recall:.2f}'

CONFIDENCE_LEVEL = 0.95
RECALL_TOLERANCE = 1e-9


def ppv_at_recall(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, target_recall: float) -> float:
    """
    PPV of the strictest score threshold that still reaches target_recall.

    Rows sharing a score cannot be separated by a threshold, so they are kept
    or dropped together. e.g. coarse LLM scores [80, 80, 80, 20]: the first
    possible cut is after the three 80s.
    """
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    weighted_positives = (weights * labels)[order]
    weighted_negatives = (weights * (1 - labels))[order]

    true_positives = np.cumsum(weighted_positives)
    false_positives = np.cumsum(weighted_negatives)
    recall = true_positives / true_positives[-1]

    is_last_of_tie_group = np.append(sorted_scores[1:] != sorted_scores[:-1], True)
    is_valid_cut = is_last_of_tie_group & (recall >= target_recall - RECALL_TOLERANCE)
    cut = np.argmax(is_valid_cut)

    return float(true_positives[cut] / (true_positives[cut] + false_positives[cut]))


def compute_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, recall_levels) -> dict:
    """AUPRC and PPV at each recall level, for one arm."""
    metrics = {AUPRC_METRIC: average_precision_score(labels, scores, sample_weight=weights)}

    for recall_level in recall_levels:
        metrics[PPV_METRIC_TEMPLATE.format(recall=recall_level)] = ppv_at_recall(labels, scores, weights, recall_level)

    return metrics


def bootstrap_patients(patient_ids: np.ndarray, n_bootstrap: int, seed: int):
    """
    Yield row positions of resampled patients, all hours of a drawn patient together.

    e.g. patients [a, a, b] -> one draw (b, a) yields rows [2, 0, 1]
    """
    rng = np.random.RandomState(seed)
    unique_patients, patient_of_row = np.unique(patient_ids, return_inverse=True)
    rows_of_patient = [np.flatnonzero(patient_of_row == i) for i in range(len(unique_patients))]

    for _ in range(n_bootstrap):
        drawn = rng.randint(0, len(unique_patients), size=len(unique_patients))

        yield np.concatenate([rows_of_patient[i] for i in drawn])


def compare_arms(rows: pd.DataFrame, arm_columns, reference_arm: str, recall_levels,
                 label_column: str, patient_column: str, weight_column: str = None,
                 n_bootstrap: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Metrics per arm with patient-bootstrap intervals, and paired differences to the reference arm.

    Returns one line per (arm, metric): estimate, CI, delta to reference, CI of the delta.
    """
    labels = rows[label_column].values.astype(float)
    weights = np.ones(len(rows)) if weight_column is None else rows[weight_column].values.astype(float)
    scores = {arm: rows[arm].values.astype(float) for arm in arm_columns}

    estimates = {arm: compute_metrics(labels, scores[arm], weights, recall_levels) for arm in arm_columns}

    # Same resampled patients for every arm, so that differences are paired
    resampled = {arm: [] for arm in arm_columns}
    for positions in bootstrap_patients(rows[patient_column].values, n_bootstrap, seed):
        if labels[positions].sum() == 0:
            continue

        for arm in arm_columns:
            resampled[arm].append(compute_metrics(labels[positions], scores[arm][positions],
                                                  weights[positions], recall_levels))

    resampled = {arm: pd.DataFrame(draws) for arm, draws in resampled.items()}
    lower_quantile = (1 - CONFIDENCE_LEVEL) / 2
    upper_quantile = 1 - lower_quantile

    comparison = []
    for arm in arm_columns:
        deltas = resampled[arm] - resampled[reference_arm]

        for metric, estimate in estimates[arm].items():
            comparison.append({
                'arm': arm,
                'metric': metric,
                'estimate': estimate,
                'ci_low': resampled[arm][metric].quantile(lower_quantile),
                'ci_high': resampled[arm][metric].quantile(upper_quantile),
                'delta_vs_reference': estimate - estimates[reference_arm][metric],
                'delta_ci_low': deltas[metric].quantile(lower_quantile),
                'delta_ci_high': deltas[metric].quantile(upper_quantile),
            })

    return pd.DataFrame(comparison)
