import pandas as pd

from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.concern_markers import \
    detect_nihss_ascertainment_surge, detect_fio2_rise, Marker


def _nihss(times, cid='p1'):
    return pd.DataFrame({'case_admission_id': cid, 'relative_hour': times})


def _fio2(times, values, cid='p1'):
    return pd.DataFrame({'case_admission_id': cid, 'relative_hour': times, 'fio2': values})


def test_regular_cadence_has_no_surge():
    # q4h routine
    assert detect_nihss_ascertainment_surge(_nihss([0, 4, 8, 12, 16, 20])).empty


def test_surge_after_routine_cadence():
    # q4h routine, then check 1h after previous -> surge at 13
    markers = detect_nihss_ascertainment_surge(_nihss([0, 4, 8, 12, 13]))
    assert markers.relative_hour.tolist() == [13]
    assert markers.marker.iloc[0] == Marker.NIHSS_SURGE.value


def test_protocol_dense_start_is_not_surge():
    # post-thrombolysis q1h, then q4h: never < half the running median
    assert detect_nihss_ascertainment_surge(_nihss([0, 1, 2, 3, 4, 8, 12])).empty


def test_too_few_prior_assessments():
    # short interval on the 3rd assessment: < MIN_PRIOR_ASSESSMENTS prior -> no marker
    assert detect_nihss_ascertainment_surge(_nihss([0, 4, 4.5])).empty


def test_duplicate_timestamps_collapsed():
    assert detect_nihss_ascertainment_surge(_nihss([0, 4, 4, 8, 12, 16])).empty


def test_fio2_rise_detected():
    markers = detect_fio2_rise(_fio2([1, 5, 9], [21, 21, 25]))
    assert markers.relative_hour.tolist() == [9]
    assert markers.marker.iloc[0] == Marker.FIO2_RISE.value


def test_fio2_small_rise_or_fall_ignored():
    assert detect_fio2_rise(_fio2([1, 5, 9], [28, 30, 24])).empty


def test_fio2_first_value_never_marker():
    # high first value (O2 already on at admission) is not a rise
    assert detect_fio2_rise(_fio2([1], [40])).empty


def test_markers_per_patient_independent():
    df = pd.concat([_fio2([1, 2], [21, 21], 'a'), _fio2([1, 2], [21, 30], 'b')])
    assert detect_fio2_rise(df).case_admission_id.tolist() == ['b']


def test_fio2_simultaneous_records_not_a_rise():
    # same time documented in two tables (21 and 28) -> one value, no rise
    assert detect_fio2_rise(_fio2([1, 1, 5], [21, 28, 28])).empty
