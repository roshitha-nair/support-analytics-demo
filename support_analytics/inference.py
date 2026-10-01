"""Shared single-ticket inference used by the local API and future demo apps.

Combines sentiment, topic inference, and the experimental dissatisfaction-risk
model score. Models are loaded lazily on first use and cached, so importing
this module does not import Transformers/Torch or load any model weights.

Topic scores are the compatibility classifier's predict_proba values. They are
not calibrated probabilities; treat them only as relative model confidence.

The risk output is the experimental model score from support_analytics.risk:
it is not a calibrated probability and showed no measurable predictive
discrimination (models/risk_model_evaluation.json).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from support_analytics.risk import (
    build_single_risk_features,
    load_risk_model,
    score_risk,
)
from support_analytics.topic_labels import get_topic_label, load_topic_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOPIC_INFERENCE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "topic_inference_demo.pkl"
)

# The pretrained sentiment model used in notebooks/02_sentiment_analysis.ipynb.
SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Number of ranked topic candidates returned with each prediction.
TOPIC_CANDIDATE_COUNT = 3


@dataclass(frozen=True)
class TopicCandidate:
    topic_id: int
    topic_label: str
    score: float


@dataclass(frozen=True)
class TopicPrediction:
    topic_id: int
    topic_label: str
    # Relative model confidence for topic_id (uncalibrated predict_proba).
    top_score: float
    # True when the registry marks the predicted topic as a noise cluster.
    is_noise_cluster: bool
    # Highest-scoring topics, best first; the first is the prediction.
    candidates: tuple


@dataclass(frozen=True)
class TicketAnalysis:
    sentiment_label: str
    sentiment_score: float
    topic_id: int
    topic_label: str
    dissatisfaction_risk_score: float
    topic: TopicPrediction


@lru_cache(maxsize=None)
def load_sentiment_pipeline():
    """Load the sentiment pipeline once; Transformers is imported only here."""
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
    )


@lru_cache(maxsize=None)
def load_topic_inference_model():
    """Load the persisted topic compatibility classifier once."""
    return joblib.load(TOPIC_INFERENCE_MODEL_PATH)


def predict_sentiment(text):
    """Return (sentiment_label, sentiment_score) for one ticket text."""

    result = load_sentiment_pipeline()(
        text,
        truncation=True,
    )[0]

    return result["label"], float(result["score"])


def predict_topic(text):
    """Return the inferred topic, its relative confidence, and ranked candidates."""

    # The compatibility classifier was trained on the cleaned ticket
    # descriptions without topic-modeling cleanup, so it receives the
    # ticket text unchanged.
    model = load_topic_inference_model()

    topic_id = int(model.predict([text])[0])

    # Labels come from the shared topic-label registry. Unknown IDs raise
    # instead of silently receiving a fallback label.
    topic_label = get_topic_label(topic_id)

    scores = model.predict_proba([text])[0]
    ranked = sorted(
        zip(model.classes_, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:TOPIC_CANDIDATE_COUNT]

    candidates = tuple(
        TopicCandidate(
            topic_id=int(candidate_id),
            topic_label=get_topic_label(int(candidate_id)),
            score=float(score),
        )
        for candidate_id, score in ranked
    )

    return TopicPrediction(
        topic_id=topic_id,
        topic_label=topic_label,
        top_score=float(scores[list(model.classes_).index(topic_id)]),
        is_noise_cluster=load_topic_registry()[topic_id].is_noise_cluster,
        candidates=candidates,
    )


def analyze_ticket(text, priority, channel):
    """Return sentiment, topic, and the experimental model score for one ticket."""

    text = text.strip()

    if not text:
        raise ValueError("Ticket text cannot be empty.")

    sentiment_label, sentiment_score = predict_sentiment(text)

    topic = predict_topic(text)

    # Build and score the risk-model input through the shared contract
    # used by the offline scoring pipeline. The result is an experimental
    # model score, not a validated probability.
    model_input = build_single_risk_features(
        priority=priority,
        channel=channel,
        sentiment_label=sentiment_label,
        topic_id=topic.topic_id,
        sentiment_score=sentiment_score,
    )

    dissatisfaction_risk_score = float(
        score_risk(load_risk_model(), model_input)[0]
    )

    return TicketAnalysis(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        topic_id=topic.topic_id,
        topic_label=topic.topic_label,
        dissatisfaction_risk_score=dissatisfaction_risk_score,
        topic=topic,
    )
