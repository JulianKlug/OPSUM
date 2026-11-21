"""
LLM review of first-stage alarms.

    alarm row ──> call 1: summarise the hospital course so far
              └─> call 2: summary + current status + XGB explanation ──> P(alarm is true)

Reviews are appended to a JSONL file as they come, which doubles as a cache:
an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import time
from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd

from prediction.short_term_outcome_prediction.second_stage_filter.feature_layout import ClinicalUnits, build_layout
from prediction.short_term_outcome_prediction.second_stage_filter.ollama_client import LLMResponseError, OllamaClient, Thinking
from prediction.short_term_outcome_prediction.second_stage_filter.out_of_fold_xgb import OutOfFoldStore
from prediction.short_term_outcome_prediction.second_stage_filter.patient_narrative import (
    NarrativeContext, PatientHistory, render_baseline, render_current_status, render_trajectory,
    render_xgb_explanation)

# Part of the cache key: bump on any prompt change, or old answers would be reused
PROMPT_VERSION = 'v3'

DEFAULT_MODEL = 'qwen3.8:27b'
CONTEXT_TOKENS = 8192
SUMMARY_MAX_TOKENS = 400


class ReasoningMode(Enum):
    """How much the model reasons before scoring an alarm."""
    NONE = 'none'        # short rationale only
    WRITTEN = 'written'  # step-by-step assessment written into the answer; works with any model
    NATIVE = 'native'    # the model's own thinking phase; reasoning models only


# Native thinking tokens count against the limit and their number cannot be capped otherwise
REVIEW_MAX_TOKENS = {ReasoningMode.NONE: 200, ReasoningMode.WRITTEN: 800, ReasoningMode.NATIVE: 4000}

MAX_PERCENT = 100
REVIEWS_FILE = 'llm_reviews.jsonl'
ENGLISH_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'preprocessing', 'preprocessing_tools',
                                  'feature_name_to_english_name_correspondence.xlsx')

SUMMARY_SYSTEM_PROMPT = """You are a stroke unit physician. Summarise the hospital course of an acute ischaemic stroke patient, from admission to the current hour, for a colleague who must judge the risk of early neurological deterioration.

Rules:
- At most 100 words.
- Cover the neurological course (NIHSS level, trend, fluctuations), haemodynamics, respiration, temperature, glucose and the relevant baseline factors.
- Facts and trends only. No prediction, no recommendation.
- Data are hourly. Values are carried forward from the last measurement and may be imputed when never measured, so an unchanged value does not prove a repeated measurement."""

REVIEW_SYSTEM_PROMPT_TEMPLATE = """You are a stroke neurologist reviewing an alarm raised by an automated model (XGBoost) on a stroke unit.

The model predicts early neurological deterioration (END) within the next 6 hours: an increase of at least 4 NIHSS points above the patient's lowest previous NIHSS. After an END event the reference is reset to the NIHSS of that event, so a patient can deteriorate again. The model raises many false alarms: only about {base_rate_percent:.0f}% of its alarms are followed by END. Its risk score is not a calibrated probability: an alarm is raised whenever the score exceeds {alarm_threshold:.2f}.

You receive a summary of the hospital course, the current status, and the model's explanation: its risk score over the recent hours and the SHAP contributions of its main drivers.

Rate how likely this alarm is to be true compared with the model's other alarms.

Most patients look stable in the hours before END: a stable NIHSS and normal vitals are not evidence against the alarm. Weigh instead the substrate for deterioration: persistent vessel occlusion or stenosis, hypoperfusion with mismatch (large Tmax volume, small core), severe or fluctuating deficit, earlier deterioration, the first 24 hours and the period after thrombolysis or thrombectomy, blood pressure drops or wide swings, hyperglycaemia, fever, atrial fibrillation, low oxygen saturation. Also judge whether the model's drivers are clinically plausible or artefacts: an isolated outlier, an event long past, or carried-forward values.

Your rating is recalibrated afterwards, so only the ordering of alarms matters. Do not anchor on the base rate: 50 is a typical alarm of this model, lower is less convincing than typical, higher is more convincing. Use the whole range from 0 to 100 and avoid round defaults.

Answer in JSON: {answer_format}."""

ANSWER_FORMAT = '"rationale" (at most 40 words), then "probability" (integer, 0 to 100)'
WRITTEN_REASONING_FORMAT = '"reasoning" (your step-by-step clinical assessment, at most 250 words), then '


ANSWER_PROPERTIES = {
    'rationale': {'type': 'string'},
    'probability': {'type': 'integer', 'minimum': 0, 'maximum': MAX_PERCENT},
}


def build_review_schema(reasoning: ReasoningMode) -> dict:
    """Properties are generated in order: written reasoning comes first so that the score can build on it."""
    properties = dict(ANSWER_PROPERTIES)
    if reasoning == ReasoningMode.WRITTEN:
        properties = {'reasoning': {'type': 'string'}, **ANSWER_PROPERTIES}

    return {'type': 'object', 'properties': properties, 'required': list(properties)}


def prompt_version(reasoning: ReasoningMode) -> str:
    """Cache key part. e.g. NONE -> 'v3', WRITTEN -> 'v3-written'"""
    if reasoning == ReasoningMode.NONE:
        return PROMPT_VERSION

    return f'{PROMPT_VERSION}-{reasoning.value}'


class AlarmReview(NamedTuple):
    summary: str
    rationale: str
    probability: float


def parse_review(response: dict):
    """
    Validate the model's answer and return (probability in [0, 1], rationale).

    e.g. {'rationale': 'stable NIHSS', 'probability': 3} -> (0.03, 'stable NIHSS')
    """
    probability_percent = response.get('probability')

    # bool is a subclass of int in Python: reject it explicitly
    if isinstance(probability_percent, bool) or not isinstance(probability_percent, (int, float)):
        raise ValueError(f'Probability missing or not a number: {response}')

    if not 0 <= probability_percent <= MAX_PERCENT:
        raise ValueError(f'Probability outside 0-{MAX_PERCENT}: {probability_percent}')

    return probability_percent / MAX_PERCENT, str(response.get('rationale', ''))


def build_summary_prompt(history: PatientHistory, context: NarrativeContext) -> str:
    return (f'Current hour since admission: {history.alarm_hour}\n\n'
            f'BASELINE\n{render_baseline(history, context)}\n\n'
            f'COURSE (hourly, older hours every 4h)\n{render_trajectory(history, context)}')


def build_review_prompt(summary: str, history: PatientHistory, context: NarrativeContext) -> str:
    return (f'Alarm at hour {history.alarm_hour} since admission.\n\n'
            f'SUMMARY OF THE HOSPITAL COURSE\n{summary}\n\n'
            f'BASELINE\n{render_baseline(history, context)}\n\n'
            f'CURRENT STATUS\n{render_current_status(history, context)}\n\n'
            f'MODEL EXPLANATION\n{render_xgb_explanation(history, context)}')


class LLMAlarmFilter:
    def __init__(self, client: OllamaClient, context: NarrativeContext, alarm_base_rate: float,
                 alarm_threshold: float, reviews_path: str, reasoning: ReasoningMode = ReasoningMode.NONE):
        self._client = client
        self._context = context
        self._review_system_prompt = REVIEW_SYSTEM_PROMPT_TEMPLATE.format(
            base_rate_percent=alarm_base_rate * MAX_PERCENT, alarm_threshold=alarm_threshold,
            answer_format=(WRITTEN_REASONING_FORMAT if reasoning == ReasoningMode.WRITTEN else '') + ANSWER_FORMAT)
        self._reasoning = reasoning
        self._reviews_path = reviews_path
        self._cache = self._load_cache(reviews_path)

    @staticmethod
    def _load_cache(reviews_path: str) -> dict:
        if not os.path.exists(reviews_path):
            return {}

        with open(reviews_path) as f:
            return {record['key']: record for record in map(json.loads, f)}

    def _cache_key(self, case_admission_id: str, hour: int) -> str:
        return f'{prompt_version(self._reasoning)}|{self._client.model}|{case_admission_id}|{hour}'

    def review(self, case_admission_id: str, history: PatientHistory) -> AlarmReview:
        """Summarise the course, then score the alarm. Answers are cached on disk."""
        key = self._cache_key(case_admission_id, history.alarm_hour)
        if key in self._cache:
            cached = self._cache[key]
            return AlarmReview(cached['summary'], cached['rationale'], cached['probability'])

        summary = self._client.chat_text(SUMMARY_SYSTEM_PROMPT, build_summary_prompt(history, self._context),
                                         max_tokens=SUMMARY_MAX_TOKENS)
        response = self._client.chat_json(self._review_system_prompt,
                                          build_review_prompt(summary, history, self._context),
                                          json_schema=build_review_schema(self._reasoning),
                                          max_tokens=REVIEW_MAX_TOKENS[self._reasoning],
                                          thinking=Thinking.ON if self._reasoning == ReasoningMode.NATIVE else Thinking.OFF)
        probability, rationale = parse_review(response)

        record = {'key': key, 'case_admission_id': case_admission_id, 'hour': history.alarm_hour,
                  'summary': summary, 'rationale': rationale, 'probability': probability}
        with open(self._reviews_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        self._cache[key] = record

        return AlarmReview(summary, rationale, probability)


def load_narrative_context(store: OutOfFoldStore, normalisation_parameters_path: str) -> NarrativeContext:
    n_columns = store.features([0]).shape[1]
    english_names = pd.read_excel(ENGLISH_NAMES_PATH).set_index('feature_name')['english_name'].to_dict()

    # 'Sex' alone does not say which one the flag stands for
    english_names['sex_male'] = 'Male sex'

    return NarrativeContext(
        layout=build_layout(store.raw_feature_names, n_columns),
        raw_feature_names=store.raw_feature_names,
        clinical_units=ClinicalUnits(pd.read_csv(normalisation_parameters_path)),
        english_names=english_names,
    )


def load_history(store: OutOfFoldStore, row_id: int) -> PatientHistory:
    """Everything known about the row's patient up to the row's hour, and nothing later."""
    history_row_ids = store.history_row_ids(row_id)

    return PatientHistory(
        features=store.features(history_row_ids),
        shap_values=store.shap_values(history_row_ids),
        xgb_scores=store.rows.xgb_score.values[history_row_ids],
    )


def review_alarm_rows(row_ids, store: OutOfFoldStore, alarm_filter: LLMAlarmFilter) -> pd.Series:
    """LLM probability per row id. Rows the model fails on are left out and reported."""
    probabilities = {}
    start = time.time()

    for n_done, row_id in enumerate(row_ids, start=1):
        case_admission_id = store.rows.case_admission_id.values[row_id]

        try:
            probabilities[row_id] = alarm_filter.review(case_admission_id, load_history(store, row_id)).probability
        except (LLMResponseError, ValueError) as error:
            print(f'Row {row_id} skipped: {error}', flush=True)

        if n_done % 50 == 0:
            print(f'{n_done}/{len(row_ids)} rows, {(time.time() - start) / n_done:.1f}s per row', flush=True)

    return pd.Series(probabilities, name='llm_probability')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='directory of the out-of-fold results')
    parser.add_argument('-s', '--sample_path', type=str, required=True, help='csv of alarm rows to review, with row_id')
    parser.add_argument('-n', '--normalisation_parameters_path', type=str, required=True)
    parser.add_argument('-b', '--alarm_base_rate', type=float, required=True, help='share of true alarms, from dev rows')
    parser.add_argument('-t', '--alarm_threshold', type=float, required=True, help='first-stage score cut-off')
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('-r', '--reasoning', type=str, default=ReasoningMode.NONE.value,
                        choices=[mode.value for mode in ReasoningMode])
    parser.add_argument('-l', '--limit', type=int, default=None, help='review only the first rows')
    args = parser.parse_args()

    oof_store = OutOfFoldStore(args.output_dir)
    llm_filter = LLMAlarmFilter(
        client=OllamaClient(args.model, context_tokens=CONTEXT_TOKENS),
        context=load_narrative_context(oof_store, args.normalisation_parameters_path),
        alarm_base_rate=args.alarm_base_rate,
        alarm_threshold=args.alarm_threshold,
        reviews_path=os.path.join(args.output_dir, REVIEWS_FILE),
        reasoning=ReasoningMode(args.reasoning),
    )

    sample_row_ids = pd.read_csv(args.sample_path).row_id.values[:args.limit]
    run_start = time.time()
    reviewed = review_alarm_rows(sample_row_ids, oof_store, llm_filter)
    print(f'{(time.time() - run_start) / max(len(sample_row_ids), 1):.1f}s per row overall')
    print(f'{len(reviewed)}/{len(sample_row_ids)} rows reviewed, mean probability {np.mean(reviewed):.3f}')
