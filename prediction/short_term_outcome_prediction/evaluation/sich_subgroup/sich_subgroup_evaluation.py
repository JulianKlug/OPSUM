"""
END model performance by symptomatic ICH (hemorrhagic transformation) status, internal test set (reviewer response).

    registry 'Symptomatic ICH' (yes / no)  --case_admission_id-->  hourly test predictions
        -> all metrics per subgroup, patient-level bootstrap 95% CI (as in first_event_evaluation)

Patients with missing sICH status are excluded from both subgroups.
"""
import argparse
import os
from enum import Enum

import numpy as np
import pandas as pd
import torch as ch

from preprocessing.geneva_stroke_unit_preprocessing.utils import create_registry_case_identification_column
from prediction.short_term_outcome_prediction.evaluation.clinical_concern_timing.concern_timing_analysis import \
    predict_hourly, load_thresholds
from prediction.short_term_outcome_prediction.evaluation.first_event.first_event_evaluation import \
    _metrics_with_ci, _counts, SEED
from prediction.utils.utils import ensure_dir

SICH_COLUMN = 'Symptomatic ICH'
SICH_YES = 'yes'
SICH_NO = 'no'


class Subgroup(str, Enum):
    ALL = 'all'
    SICH = 'sich'
    NO_SICH = 'no_sich'


def load_sich_status(registry_path: str) -> pd.Series:
    """case_admission_id -> 'yes' / 'no' (NaN dropped)."""
    registry = pd.read_excel(registry_path)
    registry['case_admission_id'] = create_registry_case_identification_column(registry)
    registry = registry.drop_duplicates(subset='case_admission_id')
    return registry.set_index('case_admission_id')[SICH_COLUMN].dropna()


def split_by_sich(preds: pd.DataFrame, sich_status: pd.Series) -> dict:
    """Subgroup -> hourly predictions. Patients without sICH status only appear in Subgroup.ALL."""
    status = preds.case_admission_id.astype(str).map(sich_status)
    return {Subgroup.ALL: preds,
            Subgroup.SICH: preds[status == SICH_YES],
            Subgroup.NO_SICH: preds[status == SICH_NO]}


def run(test_data_path: str, model_dir: str, threshold_path: str, registry_path: str, output_dir: str):
    ensure_dir(output_dir)
    thresholds = load_thresholds(threshold_path)

    X, y_df = ch.load(test_data_path, weights_only=False)
    preds = predict_hourly(X, y_df, model_dir)
    subgroups = split_by_sich(preds, load_sich_status(registry_path))

    n_missing = preds.case_admission_id.nunique() - sum(subgroups[s].case_admission_id.nunique()
                                                        for s in (Subgroup.SICH, Subgroup.NO_SICH))
    print(f'Patients without sICH status (excluded from subgroups): {n_missing}')

    # same seed per subgroup -> each subgroup reproducible on its own
    frames = []
    for subgroup, df in subgroups.items():
        metrics = _metrics_with_ci(df, thresholds, np.random.default_rng(SEED))
        frames.append(metrics.assign(subgroup=subgroup.value, **_counts(df)))
        print(f'{subgroup.value}: {_counts(df)}')

    results = pd.concat(frames, ignore_index=True)
    leading = ['subgroup', 'threshold_name', 'threshold', 'metric', 'value', 'ci_low', 'ci_high']
    results = results[leading + [c for c in results.columns if c not in leading]]
    results.to_csv(os.path.join(output_dir, 'sich_subgroup_performance.csv'), index=False)
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_data_path', required=True, help='internal test .pth')
    parser.add_argument('-m', '--model_dir', required=True)
    parser.add_argument('-t', '--threshold_path', required=True, help='threshold_tuning_results_*.json')
    parser.add_argument('-r', '--registry_path', required=True, help='stroke_registry_post_hoc_modified.xlsx')
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()
    run(args.test_data_path, args.model_dir, args.threshold_path, args.registry_path, args.output_dir)
