import pandas as pd

from prediction.short_term_outcome_prediction.evaluation.first_event.first_event_evaluation import \
    restrict_to_first_event


def _preds(cid, n_ts):
    return pd.DataFrame({'case_admission_id': cid, 'ts': range(n_ts), 'y_true': 0, 'y_prob': 0.0})


def test_truncates_before_first_event():
    # END at bins 5 and 9 -> keep ts 0..4
    preds = _preds('p1', 12)
    y_df = pd.DataFrame({'case_admission_id': ['p1', 'p1'], 'relative_sample_date_hourly_cat': [9, 5]})
    assert restrict_to_first_event(preds, y_df).ts.tolist() == [0, 1, 2, 3, 4]


def test_patient_without_event_keeps_all():
    preds = pd.concat([_preds('p1', 4), _preds('p2', 4)])
    y_df = pd.DataFrame({'case_admission_id': ['p1'], 'relative_sample_date_hourly_cat': [2]})
    kept = restrict_to_first_event(preds, y_df)
    assert kept[kept.case_admission_id == 'p2'].ts.tolist() == [0, 1, 2, 3]
    assert kept[kept.case_admission_id == 'p1'].ts.tolist() == [0, 1]
