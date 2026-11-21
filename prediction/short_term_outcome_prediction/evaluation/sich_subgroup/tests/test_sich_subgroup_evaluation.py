import pandas as pd

from prediction.short_term_outcome_prediction.evaluation.sich_subgroup.sich_subgroup_evaluation import \
    split_by_sich, Subgroup


def test_split_by_sich_status():
    # p3 has no registry status -> only in ALL
    preds = pd.DataFrame({'case_admission_id': ['p1', 'p1', 'p2', 'p3'], 'ts': [0, 1, 0, 0]})
    status = pd.Series({'p1': 'yes', 'p2': 'no'})
    groups = split_by_sich(preds, status)

    assert groups[Subgroup.SICH].case_admission_id.tolist() == ['p1', 'p1']
    assert groups[Subgroup.NO_SICH].case_admission_id.tolist() == ['p2']
    assert len(groups[Subgroup.ALL]) == 4
