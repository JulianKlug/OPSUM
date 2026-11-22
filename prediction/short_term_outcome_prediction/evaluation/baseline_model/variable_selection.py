"""
Data-driven identification of admission predictors of END, training patients only (reviewer response).

    admission table + patient label (any END)
        |-- univariable LR          -> OR per unit (per SD for z-scored), Wald 95% CI, BH-FDR p
        |-- L1-LR stability selection (C from the XGB CV folds, STABILITY_N_BOOTSTRAP patient bootstraps)
        |                           -> selection frequency
        '-- XGB on all variables    -> mean |SHAP| (computed by the caller)

    final LR set = literature predictors U {frequency >= STABILITY_THRESHOLD}, one-hot groups kept whole
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from prediction.short_term_outcome_prediction.evaluation.baseline_model.admission_features import \
    feature_group, literature_features

STABILITY_N_BOOTSTRAP = 200
STABILITY_THRESHOLD = 0.6
L1_C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
MAX_ITER = 5000
Z_95 = stats.norm.ppf(0.975)
UNPENALISED_C = 1e6  # sklearn 1.1: penalty='none' + lbfgs can fail on separable columns; tiny ridge instead


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (scipy 1.8 has no false_discovery_control)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def fit_unpenalised_lr(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=UNPENALISED_C, max_iter=MAX_ITER).fit(X.values, y)


def wald_table(model: LogisticRegression, X: pd.DataFrame) -> pd.DataFrame:
    """Odds ratios with Wald 95% CI and p from the inverse Fisher information."""
    design = np.column_stack([np.ones(len(X)), X.values])
    p = model.predict_proba(X.values)[:, 1]
    fisher = design.T @ (design * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.pinv(fisher)))[1:]

    coef = model.coef_[0]
    return pd.DataFrame({'feature': X.columns, 'coef': coef, 'se': se,
                         'odds_ratio': np.exp(coef),
                         'or_ci_low': np.exp(coef - Z_95 * se), 'or_ci_high': np.exp(coef + Z_95 * se),
                         'p_value': 2 * stats.norm.sf(np.abs(coef / se))})


def univariable_table(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """One LR per variable; constant columns (e.g. imaging fully imputed) are skipped."""
    rows = []
    for column in X.columns:
        if X[column].nunique() < 2:
            continue
        model = fit_unpenalised_lr(X[[column]], y)
        rows.append(wald_table(model, X[[column]]).assign(auroc=roc_auc_score(y, X[column])))
    table = pd.concat(rows, ignore_index=True)
    table['p_fdr'] = benjamini_hochberg(table.p_value.values)
    return table.add_prefix('univariable_').rename(columns={'univariable_feature': 'feature'})


# ---------------------------------------------------------------------------
# Stability selection
# ---------------------------------------------------------------------------

def _l1_lr(C: float) -> LogisticRegression:
    return LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=MAX_ITER)


def choose_l1_C(X: pd.DataFrame, y: np.ndarray, folds: list) -> float:
    """C maximising median validation AUROC over the given (train_idx, val_idx) folds."""
    scores = {}
    for C in L1_C_GRID:
        aurocs = [roc_auc_score(y[val], _l1_lr(C).fit(X.values[train], y[train]).predict_proba(X.values[val])[:, 1])
                  for train, val in folds]
        scores[C] = np.median(aurocs)
    return max(scores, key=scores.get)


def stability_selection(X: pd.DataFrame, y: np.ndarray, C: float, rng: np.random.Generator,
                        n_bootstrap: int = STABILITY_N_BOOTSTRAP) -> pd.Series:
    """feature -> fraction of patient bootstraps with a non-zero L1 coefficient."""
    selected = np.zeros(X.shape[1])
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        selected += _l1_lr(C).fit(X.values[idx], y[idx]).coef_[0] != 0
    return pd.Series(selected / n_bootstrap, index=X.columns, name='selection_frequency')


# ---------------------------------------------------------------------------
# Final set
# ---------------------------------------------------------------------------

def expand_groups(features, candidates) -> list:
    """Add the other levels of any one-hot group with a selected level, keeping candidate order."""
    groups = {feature_group(f) for f in features}
    return [c for c in candidates if feature_group(c) in groups]


def select_final_variables(frequency: pd.Series, threshold: float = STABILITY_THRESHOLD) -> list:
    """Literature predictors U stably selected variables."""
    stable = frequency[frequency >= threshold].index.tolist()
    return expand_groups(literature_features() + stable, frequency.index.tolist())
