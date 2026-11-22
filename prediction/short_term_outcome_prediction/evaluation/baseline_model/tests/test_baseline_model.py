import numpy as np
import pandas as pd

from prediction.short_term_outcome_prediction.evaluation.baseline_model.admission_features import \
    extract_admission_features, first_measured_imaging, hourly_labels, patient_labels, fit_binary_rescaling, \
    apply_binary_rescaling, count_late_imaging
from prediction.short_term_outcome_prediction.evaluation.baseline_model.baseline_model_evaluation import \
    xgb_end_patient_scores, fold_indices, train_model, predict, Model, Task
from prediction.short_term_outcome_prediction.evaluation.baseline_model.model_comparison import \
    paired_cluster_bootstrap, delong_test
from prediction.short_term_outcome_prediction.evaluation.baseline_model.variable_selection import \
    benjamini_hochberg, expand_groups
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries

LABELS = ['age', 'max_NIHSS', 'median_NIHSS', 'cbf_lt_30']
N_TS = 10


def _X(cids, values_by_ts):
    """values_by_ts: (n_patients, n_ts, n_labels) -> X (n, ts, labels, [cid, ts, label, value])."""
    n, n_ts, n_labels = values_by_ts.shape
    X = np.empty((n, n_ts, n_labels, 4), dtype=object)
    for i, cid in enumerate(cids):
        for t in range(n_ts):
            for j, label in enumerate(LABELS):
                X[i, t, j] = [cid, t, label, values_by_ts[i, t, j]]
    return X


def _events(cid_bins):
    return pd.DataFrame([(c, b) for c, bins in cid_bins.items() for b in bins],
                        columns=['case_admission_id', 'relative_sample_date_hourly_cat'])


def test_admission_features_hour0_with_later_imaging():
    values = np.arange(2 * N_TS * len(LABELS), dtype=float).reshape(2, N_TS, len(LABELS))
    X = _X(['p1', 'p2'], values)
    # p1: CT perfusion measured at hour 3 -> overrides imputed hour-0 value
    imaging = pd.DataFrame({'case_admission_id': ['p1'], 'sample_label': ['cbf_lt_30'],
                            'relative_sample_date_hourly_cat': [3], 'value': [99.0]})
    admission = extract_admission_features(X, imaging)

    assert list(admission.columns) == ['age', 'median_NIHSS', 'cbf_lt_30']  # max_ dropped
    assert admission.loc['p1', 'cbf_lt_30'] == 99.0
    assert admission.loc['p2', 'cbf_lt_30'] == values[1, 0, 3]
    assert admission.loc['p2', 'age'] == values[1, 0, 0]


def test_first_measured_imaging_skips_imputed_and_late():
    long = pd.DataFrame({
        'case_admission_id': ['p1'] * 4,
        'sample_label': ['cbf_lt_30'] * 4,
        'relative_sample_date_hourly_cat': [0, 2, 5, 30],
        'source': ['EHR_pop_imputed', 'EHR', 'EHR', 'EHR'],
        'value': [0.0, 1.0, 2.0, 3.0]})
    first = first_measured_imaging(long, window_h=24)
    assert first.value.tolist() == [1.0]

    late_only = long.assign(source=['EHR_pop_imputed'] * 3 + ['EHR'])
    assert first_measured_imaging(late_only, window_h=24).empty


def test_hourly_labels_match_aggregate_and_label_timeseries():
    rng = np.random.default_rng(0)
    X = _X(['p1', 'p2', 'p3'], rng.normal(size=(3, N_TS, len(LABELS))))
    y_df = _events({'p1': [0, 4], 'p2': [9]})

    _, expected = aggregate_and_label_timeseries(X, y_df, target_time_to_outcome=6)
    labels = hourly_labels(['p1', 'p2', 'p3'], y_df, N_TS)
    assert np.array_equal(labels.y_true.values, np.concatenate(expected))


def test_patient_labels_ignore_event_at_hour0():
    y_df = _events({'p1': [0], 'p2': [5]})
    labels = patient_labels(hourly_labels(['p1', 'p2', 'p3'], y_df, N_TS))
    assert labels.to_dict() == {'p1': 0, 'p2': 1, 'p3': 0}


def test_xgb_end_patient_score_is_max_before_first_predictable_event():
    # p1: END at bins 0 and 4 -> score = max over ts 0..3; p2: no END -> max over all ts
    preds = pd.DataFrame({'case_admission_id': ['p1'] * 6 + ['p2'] * 6, 'ts': list(range(6)) * 2,
                          'y_prob': [.1, .2, .3, .4, .9, .9, .1, .1, .1, .1, .1, .7]})
    y_df = _events({'p1': [0, 4]})
    preds['y_true'] = hourly_labels(['p1', 'p2'], y_df, 6).y_true.values

    scores = xgb_end_patient_scores(preds, y_df).set_index('case_admission_id')
    assert scores.loc['p1', 'y_prob'] == .4 and scores.loc['p1', 'y_true'] == 1
    assert scores.loc['p2', 'y_prob'] == .7 and scores.loc['p2', 'y_true'] == 0


def test_binary_rescaling_maps_zscored_binary_to_01():
    train = pd.DataFrame({'b': [-0.5, 2.0, -0.5], 'x': [0.1, 0.2, 0.3], 'ok': [0, 1, 1]})
    params = fit_binary_rescaling(train)
    assert list(params) == ['b']
    assert apply_binary_rescaling(train, params).b.tolist() == [0, 1, 0]


def test_identical_models_have_zero_delta():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({'case_admission_id': np.repeat(np.arange(40), 3), 'y_true': rng.integers(0, 2, 120),
                       'a': rng.random(120)})
    df['b'] = df.a
    result = paired_cluster_bootstrap(df, 'a', 'b', np.random.default_rng(0), n_bootstrap=50)
    assert (result.delta == 0).all() and (result.p_value == 1).all()


def test_delong_detects_better_model():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame({'case_admission_id': range(300), 'y_true': y,
                       'good': y + rng.normal(0, .5, 300), 'noise': rng.random(300)})
    result = delong_test(df, 'good', 'noise').iloc[0]
    assert result.delta > 0 and result.p_value < 0.001


def test_benjamini_hochberg():
    assert np.allclose(benjamini_hochberg([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


def test_expand_groups_keeps_whole_onehot_group():
    candidates = ['age', 'categorical_ivt_<90min', 'categorical_ivt_no_ivt', 'glucose']
    assert expand_groups(['categorical_ivt_no_ivt'], candidates) == ['categorical_ivt_<90min', 'categorical_ivt_no_ivt']


def test_fold_indices_partition_rows_by_patient():
    cids = np.array(['a', 'a', 'b', 'c', 'c'])
    folds = fold_indices(cids, [{'a'}, {'b', 'c'}])
    assert folds[0][1].tolist() == [0, 1] and folds[0][0].tolist() == [2, 3, 4]
    assert folds[1][1].tolist() == [2, 3, 4]


def test_late_imaging_counts_perfusion_after_registry_imaging_at_hour0():
    # p1: registry occlusion at hour 0, perfusion at hour 2 -> late; p2: all at hour 0
    imaging = pd.DataFrame({'case_admission_id': ['p1', 'p1', 'p2'],
                            'sample_label': ['vascular_occlusion', 'cbf_lt_30', 'cbf_lt_30'],
                            'relative_sample_date_hourly_cat': [0, 2, 0], 'value': [1.0, 5.0, 3.0]})
    assert count_late_imaging(imaging, ['p1', 'p2']) == 1


def test_xgb_baseline_accepts_feature_names_with_brackets():
    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame({'case_admission_id': np.arange(n), 'categorical_ivt_<90min': rng.integers(0, 2, n),
                          'age': rng.normal(size=n)})
    frame['y_true'] = (frame.age + rng.normal(0, 1, n) > 1).astype(int)
    folds = [(np.arange(100, n), np.arange(100)), (np.arange(100), np.arange(100, n))]
    fitted = train_model(Model.XGB_BASELINE, frame, ['categorical_ivt_<90min', 'age'], Task.PATIENT, folds)
    assert predict(fitted, frame).shape == (n,)
