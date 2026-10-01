"""Shared inference contract for the existing dissatisfaction-risk model.

The model artifact and its preprocessing are unchanged; this module only
centralizes the feature contract, input validation, model loading, and
scoring so the offline pipeline, API, and evaluation use one code path.

The output of score_risk() is the model's predict_proba value. Evaluation
(models/risk_model_evaluation.json) shows it is not a validated or
calibrated probability; treat it as an experimental model score.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from support_analytics.topic_labels import EXPECTED_TOPIC_IDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "dissatisfaction_risk_model.pkl"

RISK_FEATURES = (
    "Ticket Priority",
    "Ticket Channel",
    "sentiment_label",
    "topic_id",
    "sentiment_score",
)

PRIORITY_CATEGORIES = ("Critical", "High", "Low", "Medium")
CHANNEL_CATEGORIES = ("Chat", "Email", "Phone", "Social media")
SENTIMENT_LABELS = ("NEGATIVE", "POSITIVE")

CATEGORY_VALUES = {
    "Ticket Priority": frozenset(PRIORITY_CATEGORIES),
    "Ticket Channel": frozenset(CHANNEL_CATEGORIES),
    "sentiment_label": frozenset(SENTIMENT_LABELS),
    "topic_id": EXPECTED_TOPIC_IDS,
}

# Outcome and post-resolution fields that must never be model inputs.
EXCLUDED_COLUMNS = (
    "Customer Satisfaction Rating",
    "Has CSAT",
    "is_dissatisfied",
    "Resolution",
    "Time to Resolution",
    "Ticket Status",
)


class RiskInputError(ValueError):
    """Raised when risk-model inputs violate the feature contract."""


@lru_cache(maxsize=None)
def _load_model(path):
    return joblib.load(path)


def load_risk_model(path=MODEL_PATH):
    """Load the committed dissatisfaction-risk pipeline (cached per path)."""
    return _load_model(str(Path(path).resolve()))


def build_risk_features(records):
    """Return the model input frame, validating the feature contract.

    Columns are selected in the fitted order without changing values or
    dtypes, so scores are identical to scoring the source columns directly.
    """
    missing_columns = [
        column
        for column in RISK_FEATURES
        if column not in records.columns
    ]
    if missing_columns:
        raise RiskInputError(
            f"Missing risk feature columns: {', '.join(missing_columns)}"
        )

    features = records.loc[:, list(RISK_FEATURES)].copy()

    missing_values = features.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise RiskInputError(
            f"Missing values in risk features: {missing_values.to_dict()}"
        )

    for column, allowed in CATEGORY_VALUES.items():
        unknown = sorted(set(features[column]) - allowed, key=str)
        if unknown:
            raise RiskInputError(
                f"Unknown values in '{column}': {unknown}"
            )

    scores = features["sentiment_score"]
    if not ((scores >= 0) & (scores <= 1)).all():
        raise RiskInputError("sentiment_score values must be within [0, 1].")

    return features


def build_single_risk_features(
    priority,
    channel,
    sentiment_label,
    topic_id,
    sentiment_score,
):
    """Build the one-row model input used by the API."""
    return build_risk_features(
        pd.DataFrame(
            {
                "Ticket Priority": [priority],
                "Ticket Channel": [channel],
                "sentiment_label": [sentiment_label],
                "topic_id": [topic_id],
                "sentiment_score": [sentiment_score],
            }
        )
    )


def score_risk(model, features):
    """Return model scores (predict_proba for class 1) for validated features."""
    if tuple(features.columns) != RISK_FEATURES:
        raise RiskInputError(
            "Risk features must be built with build_risk_features()."
        )

    scores = model.predict_proba(features)[:, 1]

    if not np.all((scores >= 0) & (scores <= 1)):
        raise RiskInputError("Model returned scores outside [0, 1].")

    return scores
