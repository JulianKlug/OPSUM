import numpy as np
import pandas as pd
import pytest

from prediction.short_term_outcome_prediction.second_stage_filter.alarm_set import (
    WEIGHT_COLUMN, sample_alarm_rows, select_alarm_rows, threshold_at_recall)
from prediction.short_term_outcome_prediction.second_stage_filter.evaluation import (
    AUPRC_METRIC, bootstrap_patients, compare_arms, compute_metrics, ppv_at_recall)
from prediction.short_term_outcome_prediction.second_stage_filter.out_of_fold_xgb import build_row_index
from prediction.short_term_outcome_prediction.second_stage_filter.stacker import (
    build_stacker_inputs, cross_fit_scores, logit, new_linear_blender)

N_HOURS = 3
N_FEATURES = 2


def make_splits_array(case_admission_ids):
    """Mimic the splits layout: (patients, hours, features, [case id, hour, feature name, value])."""
    X = np.empty((len(case_admission_ids), N_HOURS, N_FEATURES, 4), dtype=object)
    for patient_idx, case_admission_id in enumerate(case_admission_ids):
        X[patient_idx, :, :, 0] = case_admission_id
        X[patient_idx, :, :, 3] = 0.0

    return X


def make_alarm_rows(n_positive, n_negative):
    return pd.DataFrame({
        'label': [1] * n_positive + [0] * n_negative,
        'xgb_score': np.linspace(0.9, 0.1, n_positive + n_negative),
    })


def test_row_index_is_patient_major():
    rows = build_row_index(make_splits_array(['11_1', '22_7']), fold=3)

    assert rows.case_admission_id.tolist() == ['11_1'] * N_HOURS + ['22_7'] * N_HOURS
    assert rows.hour.tolist() == [0, 1, 2, 0, 1, 2]
    assert rows.patient_id.tolist() == ['11'] * N_HOURS + ['22'] * N_HOURS
    assert (rows.fold == 3).all()


def test_threshold_at_recall_catches_the_requested_share_of_positives():
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.7, 0.4, 0.2, 0.8, 0.1])

    threshold = threshold_at_recall(labels, scores, target_recall=0.75)

    assert threshold == pytest.approx(0.4)
    assert (scores[labels == 1] >= threshold).mean() == 0.75


def test_threshold_at_recall_without_positives_raises():
    with pytest.raises(ValueError):
        threshold_at_recall(np.zeros(3), np.array([0.1, 0.2, 0.3]), target_recall=0.8)


def test_select_alarm_rows_keeps_original_index():
    rows = pd.DataFrame({'xgb_score': [0.1, 0.6, 0.3, 0.9]})

    assert select_alarm_rows(rows, threshold=0.5).index.tolist() == [1, 3]


def test_sampling_weights_restore_stratum_sizes():
    alarm_rows = make_alarm_rows(n_positive=10, n_negative=200)

    sample = sample_alarm_rows(alarm_rows, max_rows=40, positive_share=0.5, seed=0)

    assert len(sample) == 30
    assert sample[sample.label == 1][WEIGHT_COLUMN].sum() == pytest.approx(10)
    assert sample[sample.label == 0][WEIGHT_COLUMN].sum() == pytest.approx(200)


def test_ppv_at_recall_on_a_perfect_ranking():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert ppv_at_recall(labels, scores, np.ones(4), target_recall=1.0) == 1.0


def test_ppv_at_recall_keeps_tied_scores_together():
    labels = np.array([1, 0, 0, 1])
    scores = np.array([80, 80, 80, 20])

    # One positive is enough for recall 0.5, but it cannot be separated from the two tied negatives
    assert ppv_at_recall(labels, scores, np.ones(4), target_recall=0.5) == pytest.approx(1 / 3)


def test_unit_weights_equal_unweighted_metrics():
    rng = np.random.RandomState(0)
    labels = rng.binomial(1, 0.2, 500)
    scores = rng.rand(500) + 0.3 * labels

    weighted = compute_metrics(labels, scores, np.ones(500), recall_levels=[0.5])
    doubled = compute_metrics(labels, scores, np.full(500, 2.0), recall_levels=[0.5])

    assert weighted == pytest.approx(doubled)


def test_weighted_metrics_match_expanded_rows():
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    weights = np.array([1, 3, 1, 2, 1])

    weighted = compute_metrics(labels, scores, weights.astype(float), recall_levels=[1.0])
    expanded = compute_metrics(np.repeat(labels, weights), np.repeat(scores, weights),
                               np.ones(weights.sum()), recall_levels=[1.0])

    assert weighted == pytest.approx(expanded)


def test_bootstrap_draws_whole_patients():
    patient_ids = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
    rows_per_patient = {'a': 2, 'b': 1, 'c': 3}

    for positions in bootstrap_patients(patient_ids, n_bootstrap=20, seed=0):
        drawn, counts = np.unique(patient_ids[positions], return_counts=True)

        for patient, count in zip(drawn, counts):
            assert count % rows_per_patient[patient] == 0


def test_compare_arms_reference_has_zero_delta_and_better_arm_is_positive():
    rng = np.random.RandomState(1)
    n_rows = 2000
    labels = rng.binomial(1, 0.1, n_rows)
    rows = pd.DataFrame({
        'label': labels,
        'patient_id': rng.randint(0, 200, n_rows),
        'noise': rng.rand(n_rows),
        'informative': rng.rand(n_rows) + labels,
    })

    comparison = compare_arms(rows, ['noise', 'informative'], reference_arm='noise', recall_levels=[0.5],
                              label_column='label', patient_column='patient_id', n_bootstrap=100)
    auprc = comparison[comparison.metric == AUPRC_METRIC].set_index('arm')

    assert auprc.loc['noise', 'delta_vs_reference'] == 0
    assert auprc.loc['informative', 'delta_ci_low'] > 0


def test_logit_is_finite_at_the_extremes():
    assert np.isfinite(logit(np.array([0.0, 0.5, 1.0]))).all()
    assert logit(np.array([0.5]))[0] == pytest.approx(0)


def test_stacker_inputs_drop_the_shap_base_value():
    features = np.zeros((4, 3))
    shap_values = np.ones((4, 3 + 1))

    inputs = build_stacker_inputs(features, shap_values, np.full(4, 0.5))

    assert inputs.shape == (4, 3 + 3 + 1)


def test_cross_fit_never_scores_a_patient_with_its_own_model():
    rng = np.random.RandomState(2)
    n_patients, rows_per_patient = 60, 5
    patient_ids = np.repeat(np.arange(n_patients), rows_per_patient)
    labels = np.repeat(rng.binomial(1, 0.5, n_patients), rows_per_patient)

    # The only input is a patient fingerprint: it predicts the label only if the patient was seen in training
    inputs = np.eye(n_patients)[patient_ids]
    scores = cross_fit_scores(new_linear_blender, inputs, labels, patient_ids)

    assert not np.isnan(scores).any()
    assert abs(np.corrcoef(scores, labels)[0, 1]) < 0.3


# ---------------------------------------------------------------------------
# Feature layout, out-of-fold store, patient narrative, LLM answer parsing
# ---------------------------------------------------------------------------
import json

from prediction.short_term_outcome_prediction.second_stage_filter import out_of_fold_xgb
from prediction.short_term_outcome_prediction.second_stage_filter.feature_layout import (
    Block, ClinicalUnits, build_layout)
from prediction.short_term_outcome_prediction.second_stage_filter.llm_filter import (
    build_review_prompt, build_summary_prompt, parse_review)
from prediction.short_term_outcome_prediction.second_stage_filter.out_of_fold_xgb import OutOfFoldStore
from prediction.short_term_outcome_prediction.second_stage_filter.patient_narrative import (
    NarrativeContext, PatientHistory, describe_feature, render_trajectory, trajectory_hours)
from prediction.short_term_outcome_prediction.testing.compute_shap_explanations_over_time import (
    _build_aggregated_feature_names)

VITAL_NAMES = [f'{statistic}_{signal}' for signal in
               ('NIHSS', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'mean_blood_pressure',
                'heart_rate', 'oxygen_saturation', 'respiratory_rate')
               for statistic in ('min', 'median', 'max')]
RAW_NAMES = VITAL_NAMES + ['Glasgow Coma Scale', 'temperature', 'FIO2', 'glucose', 'age', 'weight',
                           'sodium', 'tmax_gt_6', 'sex_male', 'medhist_diabetes_yes']
BINARY_NAMES = ['sex_male', 'medhist_diabetes_yes']
N_ALL_BLOCKS = 11


def make_context():
    normalised_names = [name for name in RAW_NAMES if name not in BINARY_NAMES]
    normalisation_parameters = pd.DataFrame({'variable': normalised_names,
                                             'original_mean': 100.0, 'original_std': 10.0})
    n_columns = N_ALL_BLOCKS * len(RAW_NAMES) + 1

    return NarrativeContext(layout=build_layout(RAW_NAMES, n_columns), raw_feature_names=RAW_NAMES,
                            clinical_units=ClinicalUnits(normalisation_parameters), english_names={})


def make_history(n_hours, seed=0):
    rng = np.random.RandomState(seed)
    n_columns = N_ALL_BLOCKS * len(RAW_NAMES) + 1

    return PatientHistory(features=rng.randn(n_hours, n_columns).astype('float32'),
                          shap_values=rng.randn(n_hours, n_columns + 1).astype('float32'),
                          xgb_scores=rng.rand(n_hours))


@pytest.mark.parametrize('with_lag, with_rolling', [(False, False), (True, False), (False, True), (True, True)])
def test_layout_matches_the_aggregation_column_order(with_lag, with_rolling):
    raw_names = ['max_NIHSS', 'glucose', 'age']
    expected_names = _build_aggregated_feature_names(raw_names, add_lag_features=with_lag,
                                                     add_rolling_features=with_rolling)

    layout = build_layout(raw_names, n_columns=len(expected_names))

    assert [feature.raw_feature for feature in layout] == [
        next((raw for raw in raw_names if name.endswith(raw)), None) for name in expected_names]
    assert layout[expected_names.index('timestep_idx')].block == Block.HOUR
    assert layout[expected_names.index('max_max_NIHSS')].block == Block.CUMULATIVE_MAX
    assert layout[expected_names.index('max_NIHSS')].block == Block.RAW


def test_layout_rejects_an_unknown_column_count():
    with pytest.raises(ValueError):
        build_layout(['a', 'b'], n_columns=10)


def test_clinical_units_add_the_mean_to_levels_but_not_to_changes():
    context = make_context()
    level = next(f for f in context.layout if f.block == Block.RAW and f.raw_feature == 'glucose')
    change = next(f for f in context.layout if f.block == Block.DIFF and f.raw_feature == 'glucose')
    binary = next(f for f in context.layout if f.block == Block.RAW and f.raw_feature == 'sex_male')
    hour = next(f for f in context.layout if f.block == Block.HOUR)

    assert context.clinical_units.convert(1.0, level) == pytest.approx(110)
    assert context.clinical_units.convert(1.0, change) == pytest.approx(10)
    assert context.clinical_units.convert(1.0, binary) == 1.0
    assert context.clinical_units.convert(1.0, hour) == pytest.approx(71)


def test_describe_feature_names_block_and_variable():
    context = make_context()
    trend = next(f for f in context.layout if f.block == Block.ROLLING_TREND and f.raw_feature == 'max_NIHSS')

    assert describe_feature(trend, context) == 'max_NIHSS, trend per hour over last 6h'


def test_trajectory_hours_end_at_the_alarm_hour():
    assert trajectory_hours(3).tolist() == [0, 1, 2, 3]
    assert trajectory_hours(30).tolist() == [0, 4, 8, 12, 16] + list(range(19, 31))


def test_trajectory_has_one_line_per_shown_hour():
    table = render_trajectory(make_history(n_hours=31), make_context())

    assert len(table.splitlines()) == 1 + len(trajectory_hours(30))


def test_prompts_render_for_first_and_late_hours():
    context = make_context()

    for n_hours in (1, 40):
        history = make_history(n_hours)
        review_prompt = build_review_prompt('stable course', history, context)

        assert f'Current hour since admission: {n_hours - 1}' in build_summary_prompt(history, context)
        assert 'stable course' in review_prompt
        assert 'Top drivers now' in review_prompt


def make_store(tmp_path, hours_per_patient, patients_per_fold, n_columns=4):
    """Write out-of-fold files the way compute_out_of_fold_predictions does; feature value = global row id."""
    all_rows, next_row_id = [], 0
    for fold, n_patients in enumerate(patients_per_fold):
        case_ids = [f'{fold}{patient}_1' for patient in range(n_patients)]
        rows = build_row_index(make_splits_array(case_ids)[:, :hours_per_patient], fold)
        rows['label'] = 0
        rows['xgb_score'] = 0.5

        row_ids = np.arange(next_row_id, next_row_id + len(rows), dtype='float32')
        next_row_id += len(rows)
        np.save(tmp_path / out_of_fold_xgb.FEATURES_FILE_TEMPLATE.format(fold=fold),
                np.repeat(row_ids[:, None], n_columns, axis=1))
        np.save(tmp_path / out_of_fold_xgb.SHAP_FILE_TEMPLATE.format(fold=fold),
                np.repeat(-row_ids[:, None], n_columns + 1, axis=1))
        all_rows.append(rows)

    pd.concat(all_rows, ignore_index=True).to_csv(tmp_path / out_of_fold_xgb.ROWS_FILE, index=False)
    (tmp_path / out_of_fold_xgb.RAW_FEATURE_NAMES_FILE).write_text(json.dumps(['a']))

    return OutOfFoldStore(str(tmp_path))


def test_store_returns_rows_across_folds_in_the_requested_order(tmp_path):
    store = make_store(tmp_path, hours_per_patient=3, patients_per_fold=[2, 3])
    row_ids = [13, 0, 7, 5]

    assert store.features(row_ids)[:, 0].tolist() == row_ids
    assert store.shap_values(row_ids)[:, 0].tolist() == [-row_id for row_id in row_ids]
    assert store.rows.case_admission_id.tolist()[7] == '10_1'


def test_store_history_stops_at_the_alarm_hour(tmp_path):
    store = make_store(tmp_path, hours_per_patient=3, patients_per_fold=[2, 3])

    # Row 10 is hour 1 of the second patient of fold 1 (rows 9, 10, 11)
    assert store.history_row_ids(10).tolist() == [9, 10]
    assert store.history_row_ids(9).tolist() == [9]


def test_parse_review_converts_percent_to_probability():
    assert parse_review({'rationale': 'stable', 'probability': 3}) == (0.03, 'stable')


@pytest.mark.parametrize('response', [{}, {'probability': 'high'}, {'probability': 140},
                                      {'probability': -1}, {'probability': True}])
def test_parse_review_rejects_invalid_answers(response):
    with pytest.raises(ValueError):
        parse_review(response)


def test_overall_auprc_with_the_xgb_arm_equals_the_first_stage_auprc():
    from sklearn.metrics import average_precision_score
    from prediction.short_term_outcome_prediction.second_stage_filter.run_comparison import compare_overall_auprc

    rng = np.random.RandomState(3)
    n_rows = 3000
    labels = rng.binomial(1, 0.05, n_rows)
    all_rows = pd.DataFrame({'label': labels, 'patient_id': rng.randint(0, 100, n_rows).astype(str),
                             'xgb_score': np.clip(rng.rand(n_rows) * 0.8 + 0.15 * labels, 0, 1)})
    alarm_rows = select_alarm_rows(all_rows, threshold=0.5).copy()
    alarm_rows['oracle'] = alarm_rows.label.astype(float)

    overall = compare_overall_auprc(all_rows, alarm_rows, ['xgb_score', 'oracle'], 'xgb_score').set_index('arm')

    assert overall.loc['xgb_score', 'estimate'] == pytest.approx(
        average_precision_score(all_rows.label, all_rows.xgb_score))
    assert overall.loc['oracle', 'delta_ci_low'] > 0


def test_written_reasoning_is_generated_before_the_score():
    from prediction.short_term_outcome_prediction.second_stage_filter.llm_filter import (
        ReasoningMode, build_review_schema, prompt_version)

    assert list(build_review_schema(ReasoningMode.WRITTEN)['properties']) == ['reasoning', 'rationale', 'probability']
    assert list(build_review_schema(ReasoningMode.NONE)['properties']) == ['rationale', 'probability']
    assert len({prompt_version(mode) for mode in ReasoningMode}) == len(ReasoningMode)
