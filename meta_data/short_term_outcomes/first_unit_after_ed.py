"""
First hospital unit after the emergency department (ED), per case of the END dataset (derivation + test).

    passages (chronological): ATT > ROUGE-A > SUOB-US > BO-OPE > OPESSPI-RE > 2EL+-US > 2DL-US
                              '-------- ED --------'  '---- transit ----'   ^ first unit after ED

Leading ED and transit (OR, recovery, exams, leave) passages are skipped.
Direct admissions (no ED passage): first ward / ICU unit.

Outputs:
    first_unit_after_ed.csv          case_admission_id, first_unit_after_ed, first_unit_begin, went_through_ed
    first_unit_after_ed_summary.csv  unit, n_cases, fraction (of whole dataset; unlinked cases as NO_UNIT_LABEL)
"""
import argparse
import glob
import os
import re

import pandas as pd

from preprocessing.geneva_stroke_unit_preprocessing.utils import create_ehr_case_identification_column
from prediction.utils.utils import ensure_dir

PASSAGES_FILE_PATTERN = 'passages_j*.csv'
DATE_FORMAT = '%d.%m.%Y %H:%M'
PASSAGE_TYPE = 'passage'
NO_UNIT_LABEL = 'no_unit_identified'

# ED: triage colour zones, waiting area, emergency services (adult, ophtalmo, ORL, gyneco, 3-Chene, ambulatory),
# ED observation unit (SUOB*)
ED_UNIT_PATTERN = re.compile(r'^(?:ATT|ROUGE-A|ROUGE-B|JAUNE|MAUVE|ORANGE|VERT|BLEU?-US|GYOBURG-US|PRE0UUA-US)$'
                             r'|^URG|^U3C|^UUA|^SUOB')
# transit: operating rooms, recovery rooms, exams, endoscopy, leave, inter-site transfer, administrative
TRANSIT_UNIT_PATTERN = re.compile(r'^BO-|SSPI|^PM-|^ENDO|^(?:CONGE|INTERSITE|SORTIE_24H|U388)$')


def load_passages(extraction_dir: str, cohort_ids) -> pd.DataFrame:
    """Chronological, deduplicated unit passages of cohort cases."""
    files = sorted(glob.glob(os.path.join(extraction_dir, PASSAGES_FILE_PATTERN)))
    passages = pd.concat([pd.read_csv(f, sep=';', dtype=str, encoding='utf-8-sig') for f in files])
    passages = passages[passages.type == PASSAGE_TYPE].dropna(subset=['care_unit'])

    passages['case_admission_id'] = create_ehr_case_identification_column(passages)
    passages = passages[passages.case_admission_id.isin(cohort_ids)]
    passages = passages.drop_duplicates(subset=['case_admission_id', 'begin_date', 'end_date', 'care_unit'])

    passages['begin'] = pd.to_datetime(passages.begin_date, format=DATE_FORMAT)
    return passages.sort_values(['case_admission_id', 'begin'])


def first_unit_after_ed(passages: pd.DataFrame) -> pd.DataFrame:
    """Per case: first non-ED, non-transit unit + whether any ED passage occurred."""
    is_ed = passages.care_unit.str.contains(ED_UNIT_PATTERN)
    is_transit = passages.care_unit.str.contains(TRANSIT_UNIT_PATTERN)

    went_through_ed = is_ed.groupby(passages.case_admission_id).any().rename('went_through_ed')
    wards = passages[~is_ed & ~is_transit]
    first_unit = (wards.groupby('case_admission_id').first()[['care_unit', 'begin_date']]
                  .rename(columns={'care_unit': 'first_unit_after_ed', 'begin_date': 'first_unit_begin'}))
    return first_unit.join(went_through_ed).reset_index()


def summarize(first_unit: pd.DataFrame, n_cohort: int) -> pd.DataFrame:
    """Cases per unit, fraction of whole dataset; cases without a unit as NO_UNIT_LABEL."""
    summary = first_unit.first_unit_after_ed.value_counts().rename_axis('unit').rename('n_cases').reset_index()
    n_no_unit = n_cohort - len(first_unit)
    summary = pd.concat([summary, pd.DataFrame({'unit': [NO_UNIT_LABEL], 'n_cases': [n_no_unit]})],
                        ignore_index=True)
    summary['fraction'] = summary.n_cases / n_cohort
    return summary


def run(features_path: str, extraction_dir: str, output_dir: str):
    ensure_dir(output_dir)

    # cohort: train + test = every case in preprocessed features (see data_splits.py)
    cohort_ids = pd.read_csv(features_path, usecols=['case_admission_id'], dtype=str).case_admission_id.unique()

    first_unit = first_unit_after_ed(load_passages(extraction_dir, cohort_ids))
    summary = summarize(first_unit, len(cohort_ids))

    first_unit.to_csv(os.path.join(output_dir, 'first_unit_after_ed.csv'), index=False)
    summary.to_csv(os.path.join(output_dir, 'first_unit_after_ed_summary.csv'), index=False)

    print(f'cohort cases: {len(cohort_ids)}, with first unit: {len(first_unit)}, '
          f'without ED passage: {(~first_unit.went_through_ed).sum()}')
    print(summary.head(10).to_string(index=False))
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_path', required=True, help='preprocessed features .csv (derivation + test)')
    parser.add_argument('-e', '--extraction_dir', required=True, help='raw extraction dir with passages_j*.csv')
    parser.add_argument('-o', '--output_dir', required=True)
    args = parser.parse_args()
    run(args.features_path, args.extraction_dir, args.output_dir)
