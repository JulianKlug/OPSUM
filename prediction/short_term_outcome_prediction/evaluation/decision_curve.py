"""
Decision curve analysis: net benefit of acting on hourly risk predictions, across threshold probabilities.

Why: PPV alone does not say whether a model is worth using. Net benefit weighs true positives against false positives
at the harm/benefit trade-off implied by the threshold probability pt (Vickers & Elkin, 2006).

    NB(pt) = TP/N − FP/N · pt / (1 − pt)

    pt = 5%  ->  acting on 19 false positives is accepted to act on 1 true positive (odds 1:19)

Net benefit is only meaningful on calibrated probabilities: scores are first recalibrated on held-out data.

    raw score ──logit──> a + b · logit(p) ──sigmoid──> recalibrated probability
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BOOTSTRAP_SEED = 42

LOWER_CI_PERCENTILE = 2.5
UPPER_CI_PERCENTILE = 97.5
LOWER_CI_SUFFIX = '_lower_ci'
UPPER_CI_SUFFIX = '_upper_ci'

# Keeps logit finite for scores of exactly 0 or 1
PROBABILITY_CLIP = 1e-6

# Large inverse regularisation: unpenalised logistic regression (sklearn 1.1 has no penalty=None)
UNPENALISED_C = 1e10

PER_100 = 100

# Metrics with a bootstrap interval
BOOTSTRAPPED_METRICS = ['net_benefit', 'net_benefit_vs_treat_all']


class ScoreScale(Enum):
    RAW = 'raw'
    RECALIBRATED = 'recalibrated'


def _logit(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.log(clipped / (1 - clipped))


def _sigmoid(log_odds):
    return 1 / (1 + np.exp(-log_odds))


@dataclass(frozen=True)
class Recalibration:
    """Logistic recalibration: P(y = 1) = sigmoid(intercept + slope · logit(p))"""
    intercept: float
    slope: float

    def apply(self, probabilities):
        return _sigmoid(self.intercept + self.slope * _logit(probabilities))


def fit_recalibration(y_true, y_prob):
    model = LogisticRegression(C=UNPENALISED_C, max_iter=1000)
    model.fit(_logit(y_prob).reshape(-1, 1), np.asarray(y_true))

    return Recalibration(intercept=float(model.intercept_[0]), slope=float(model.coef_[0, 0]))


def calibration_summary(y_true, y_prob):
    """Mean predicted vs observed risk, and calibration slope (1 = ideal; < 1 = predictions too extreme)"""
    return {
        'observed_risk': float(np.mean(y_true)),
        'mean_predicted_risk': float(np.mean(y_prob)),
        'calibration_slope': fit_recalibration(y_true, y_prob).slope,
    }


def _harm_weight(threshold_probabilities):
    """Odds of the threshold probability: weight of one false positive relative to one true positive"""
    threshold_probabilities = np.asarray(threshold_probabilities, dtype=float)
    return threshold_probabilities / (1 - threshold_probabilities)


def treat_all_net_benefit(prevalence, threshold_probabilities):
    return prevalence - (1 - prevalence) * _harm_weight(threshold_probabilities)


def _positive_counts(y_true, y_prob, threshold_probabilities):
    """TP and FP counts at each threshold, as arrays aligned with threshold_probabilities"""
    y_true = np.asarray(y_true).astype(bool)
    is_positive = np.asarray(y_prob)[np.newaxis, :] >= np.asarray(threshold_probabilities)[:, np.newaxis]

    n_true_positives = (is_positive & y_true).sum(axis=1)
    n_false_positives = (is_positive & ~y_true).sum(axis=1)

    return n_true_positives, n_false_positives


def net_benefit(y_true, y_prob, threshold_probabilities):
    n_true_positives, n_false_positives = _positive_counts(y_true, y_prob, threshold_probabilities)
    n_samples = len(y_true)

    return n_true_positives / n_samples - n_false_positives / n_samples * _harm_weight(threshold_probabilities)


def decision_curve(y_true, y_prob, threshold_probabilities):
    """One row per threshold probability: model vs treat all; treat none is 0 by definition"""
    threshold_probabilities = np.asarray(threshold_probabilities, dtype=float)
    n_samples = len(y_true)
    prevalence = float(np.mean(y_true))

    n_true_positives, n_false_positives = _positive_counts(y_true, y_prob, threshold_probabilities)
    model_net_benefit = net_benefit(y_true, y_prob, threshold_probabilities)
    all_net_benefit = treat_all_net_benefit(prevalence, threshold_probabilities)

    return pd.DataFrame({
        'threshold_probability': threshold_probabilities,
        'net_benefit': model_net_benefit,
        'treat_all_net_benefit': all_net_benefit,
        'net_benefit_vs_treat_all': model_net_benefit - all_net_benefit,
        # Fraction of the maximal achievable net benefit (= prevalence, a perfect model)
        'standardised_net_benefit': model_net_benefit / prevalence,
        # e.g. 30 -> per 100 predictions, 30 fewer false positives than treating all, at equal net benefit
        'net_false_positives_avoided_per_100': (model_net_benefit - all_net_benefit)
                                               / _harm_weight(threshold_probabilities) * PER_100,
        'true_positives_per_100': n_true_positives / n_samples * PER_100,
        'false_positives_per_100': n_false_positives / n_samples * PER_100,
        'ppv': n_true_positives / np.maximum(n_true_positives + n_false_positives, 1),
        'prevalence': prevalence,
    })


def bootstrap_decision_curve(y_true_per_patient, y_prob_per_patient, threshold_probabilities,
                             n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_BOOTSTRAP_SEED):
    """Percentile CIs from patient-level resampling: hourly predictions of one patient are correlated.

    Inputs are (n_patients, n_hours) arrays. Returns one row per threshold probability.
    """
    y_true_per_patient = np.asarray(y_true_per_patient)
    y_prob_per_patient = np.asarray(y_prob_per_patient)
    n_patients = len(y_true_per_patient)
    rng = np.random.default_rng(seed)

    draws = {metric: [] for metric in BOOTSTRAPPED_METRICS}
    for _ in range(n_bootstrap):
        patient_idx = rng.integers(0, n_patients, n_patients)
        curve_df = decision_curve(y_true_per_patient[patient_idx].ravel(), y_prob_per_patient[patient_idx].ravel(),
                                  threshold_probabilities)
        for metric in BOOTSTRAPPED_METRICS:
            draws[metric].append(curve_df[metric].values)

    ci_df = pd.DataFrame({'threshold_probability': np.asarray(threshold_probabilities, dtype=float)})
    for metric in BOOTSTRAPPED_METRICS:
        ci_df[metric + LOWER_CI_SUFFIX] = np.percentile(draws[metric], LOWER_CI_PERCENTILE, axis=0)
        ci_df[metric + UPPER_CI_SUFFIX] = np.percentile(draws[metric], UPPER_CI_PERCENTILE, axis=0)

    return ci_df
