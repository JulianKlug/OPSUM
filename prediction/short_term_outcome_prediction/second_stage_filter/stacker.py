"""
Second-stage models trained on alarm rows only, to tell true from false alarms.

Scores are cross-fitted: each row is scored by a model that saw no hour of
the same patient.

    alarm rows ──> split patients in k groups ──> train on k-1, score the k-th
"""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_CROSS_FIT_SPLITS = 5
PROBABILITY_FLOOR = 1e-6

# Fixed, deliberately small and regularised: tuning on the evaluation rows would bias the comparison
STACKER_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_lambda=10,
    min_child_weight=5,
    tree_method='hist',
)


def logit(probabilities: np.ndarray) -> np.ndarray:
    """Probability to log-odds, e.g. 0.5 -> 0."""
    clipped = np.clip(probabilities, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)

    return np.log(clipped / (1 - clipped))


def build_stacker_inputs(features: np.ndarray, shap_values: np.ndarray, xgb_scores: np.ndarray) -> np.ndarray:
    """
    Everything the first stage knew about a row: its inputs, its explanation, its output.

    The last SHAP column is the base value, constant within a fold, and is dropped.
    """
    feature_contributions = shap_values[:, :-1]

    return np.hstack([features, feature_contributions, logit(xgb_scores)[:, None]]).astype('float32')


def new_tree_stacker():
    return xgb.XGBClassifier(**STACKER_PARAMS)


def new_linear_blender():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def cross_fit_scores(new_model, inputs: np.ndarray, labels: np.ndarray, patient_ids: np.ndarray,
                     weights: np.ndarray = None, seed: int = 42) -> np.ndarray:
    """Score every row with a model trained on other patients only."""
    weights = np.ones(len(labels)) if weights is None else weights
    scores = np.full(len(labels), np.nan, dtype='float64')

    splitter = StratifiedGroupKFold(n_splits=N_CROSS_FIT_SPLITS, shuffle=True, random_state=seed)
    for train_rows, held_out_rows in splitter.split(inputs, labels, groups=patient_ids):
        model = new_model()
        fit_weights = {_final_step_name(model) + 'sample_weight': weights[train_rows]}
        model.fit(inputs[train_rows], labels[train_rows], **fit_weights)

        scores[held_out_rows] = model.predict_proba(inputs[held_out_rows])[:, 1]

    return scores


def _final_step_name(model) -> str:
    """sklearn pipelines route fit arguments by step name: 'logisticregression__sample_weight'."""
    if not hasattr(model, 'steps'):
        return ''

    return model.steps[-1][0] + '__'
