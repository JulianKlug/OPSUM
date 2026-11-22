"""
Paired comparison of two models scored on the same rows (reviewer response).

    rows (patient-hours or patients), columns y_true, prob_a, prob_b
        paired patient-cluster bootstrap: same patient draw for both models
            -> delta = metric(a) - metric(b), 95% percentile CI, two-sided p = 2 * min(P(delta <= 0), P(delta >= 0))
        DeLong (patient-level only: rows must be independent)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from prediction.outcome_prediction.model_comparison.de_long_comparison import delong_roc_test
from prediction.short_term_outcome_prediction.evaluation.first_event.first_event_evaluation import \
    N_BOOTSTRAP, CI_LOW_Q, CI_HIGH_Q

METRICS = {'auroc': roc_auc_score, 'auprc': average_precision_score}


def _deltas(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    return {name: fn(y, a) - fn(y, b) for name, fn in METRICS.items()}


def paired_cluster_bootstrap(df: pd.DataFrame, prob_a: str, prob_b: str, rng: np.random.Generator,
                             n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Delta (a - b) per metric with patient-cluster bootstrap CI and two-sided p."""
    y, a, b = df.y_true.values, df[prob_a].values, df[prob_b].values
    rows_by_patient = list(df.groupby('case_admission_id').indices.values())
    n_patients = len(rows_by_patient)

    point = _deltas(y, a, b)
    samples = {name: [] for name in METRICS}
    for _ in range(n_bootstrap):
        idx = np.concatenate([rows_by_patient[i] for i in rng.integers(0, n_patients, n_patients)])
        if len(np.unique(y[idx])) < 2:
            continue
        for name, delta in _deltas(y[idx], a[idx], b[idx]).items():
            samples[name].append(delta)

    rows = []
    for name, deltas in samples.items():
        deltas = np.asarray(deltas)
        p = min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
        rows.append({'metric': name, 'test': 'paired_cluster_bootstrap',
                     'value_a': METRICS[name](y, a), 'value_b': METRICS[name](y, b),
                     'delta': point[name], 'ci_low': np.percentile(deltas, CI_LOW_Q),
                     'ci_high': np.percentile(deltas, CI_HIGH_Q), 'p_value': p})
    return pd.DataFrame(rows)


def delong_test(df: pd.DataFrame, prob_a: str, prob_b: str) -> pd.DataFrame:
    """DeLong AUROC test; valid only with one row per patient."""
    assert df.case_admission_id.is_unique, 'DeLong requires independent rows (one per patient)'
    y, a, b = df.y_true.values, df[prob_a].values, df[prob_b].values
    p = float(10 ** np.ravel(delong_roc_test(y, a, b))[0])  # delong_roc_test returns log10(p)
    auroc_a, auroc_b = roc_auc_score(y, a), roc_auc_score(y, b)
    return pd.DataFrame([{'metric': 'auroc', 'test': 'delong', 'value_a': auroc_a, 'value_b': auroc_b,
                          'delta': auroc_a - auroc_b, 'ci_low': np.nan, 'ci_high': np.nan, 'p_value': p}])
