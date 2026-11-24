import numpy as np
import pytest

from prediction.short_term_outcome_prediction.evaluation.decision_curve import (
    Recalibration, fit_recalibration, net_benefit, treat_all_net_benefit, decision_curve, bootstrap_decision_curve)


def test_net_benefit_matches_hand_computation():
    # 10 samples, at pt = 0.2 (harm weight 0.25): 2 TP, 4 FP -> 0.2 - 0.4 * 0.25 = 0.1
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.5, 0.1, 0.6, 0.6, 0.3, 0.3, 0.1, 0.1, 0.1])

    assert net_benefit(y_true, y_prob, [0.2])[0] == pytest.approx(0.1)


def test_perfect_model_reaches_prevalence():
    y_true = np.array([1, 0, 0, 0])

    assert net_benefit(y_true, y_true.astype(float), [0.1, 0.5])  == pytest.approx([0.25, 0.25])


def test_treat_all_equals_prevalence_at_threshold_zero_and_zero_at_prevalence():
    assert treat_all_net_benefit(0.02, [0.0, 0.02]) == pytest.approx([0.02, 0.0])


def test_model_flagging_everyone_equals_treat_all():
    y_true = np.array([1, 0, 0, 0, 0])

    curve_df = decision_curve(y_true, np.ones(5), [0.05, 0.1])

    assert curve_df.net_benefit_vs_treat_all.values == pytest.approx([0, 0])


def test_recalibration_recovers_known_parameters():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.01, 0.99, 200_000)
    truth = Recalibration(intercept=-2.0, slope=0.7)
    y_true = rng.uniform(size=raw.size) < truth.apply(raw)

    fitted = fit_recalibration(y_true, raw)

    assert fitted.intercept == pytest.approx(truth.intercept, abs=0.05)
    assert fitted.slope == pytest.approx(truth.slope, abs=0.05)


def test_recalibration_preserves_ranking():
    raw = np.array([0.1, 0.3, 0.2, 0.9])

    recalibrated = Recalibration(intercept=-3, slope=0.5).apply(raw)

    assert np.argsort(recalibrated).tolist() == np.argsort(raw).tolist()


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(size=(50, 24))
    y_true = (rng.uniform(size=(50, 24)) < y_prob * 0.2).astype(int)
    thresholds = [0.05, 0.1]

    point = decision_curve(y_true.ravel(), y_prob.ravel(), thresholds).net_benefit.values
    ci_df = bootstrap_decision_curve(y_true, y_prob, thresholds, n_bootstrap=200)

    assert np.all(ci_df.net_benefit_lower_ci <= point)
    assert np.all(point <= ci_df.net_benefit_upper_ci)
    assert ci_df.equals(bootstrap_decision_curve(y_true, y_prob, thresholds, n_bootstrap=200))
