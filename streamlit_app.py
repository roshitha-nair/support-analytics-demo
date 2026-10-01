"""Streamlit demo: analyze a hypothetical customer-support ticket.

The app calls the shared inference layer (support_analytics.inference) in the
same process. Models load lazily on the first analysis and are cached once per
server process (st.cache_resource around the shared module's loaders), so
reruns and new sessions do not load them again. On first use the sentiment
model is downloaded from the Hugging Face Hub.

The app reads only model artifacts. It never reads ticket data, and it does
not store, print, or log submitted ticket text.

Run locally with:  streamlit run streamlit_app.py
"""

from functools import lru_cache
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from support_analytics import inference
from support_analytics.risk import (
    CHANNEL_CATEGORIES,
    PRIORITY_CATEGORIES,
    load_risk_model,
)
from support_analytics.topic_labels import load_topic_registry


PROJECT_ROOT = Path(__file__).resolve().parent
EVALUATION_PATH = PROJECT_ROOT / "models" / "risk_model_evaluation.json"

# Same input limit as the local API (app/api.py).
MAX_TEXT_LENGTH = 5000

# UI presentation heuristic only, not a model threshold and not validated.
# Topic predictions whose top score falls below this value are described as
# "low-confidence / effectively unclassified"; the prediction itself is shown
# unchanged. Chosen from the held-out split used to train the classifier:
# below 0.25 the classifier agreed with the BERTopic assignments on fewer than
# 40% of tickets (about 28% below 0.20), versus about 56% from 0.25 upward.
# Hand-written tickets typically score 0.13-0.18.
TOPIC_LOW_CONFIDENCE_DISPLAY_THRESHOLD = 0.25

# Hypothetical, newly written examples. They are not taken from the dataset.
EXAMPLE_TICKETS = {
    "Positive feedback": (
        "Thanks for sorting out my order so quickly yesterday. Everything is "
        "working now and the support agent was very helpful."
    ),
    "Account login problem": (
        "I reset my password twice but I still can't sign in to my account. "
        "It keeps saying my login details are incorrect."
    ),
    "Duplicate billing charge": (
        "My card was charged twice for the same subscription this month. "
        "Please refund the duplicate payment."
    ),
    "Technical fault after update": (
        "Since the latest software update my laptop restarts on its own every "
        "few minutes and I lose unsaved work."
    ),
}

NO_EXAMPLE = "Write your own ticket"

PRIVACY_NOTICE = (
    "**Privacy notice:** Do not enter personal or confidential information. "
    "This demo does not store submitted ticket text. Model inference runs "
    "within the application, but the hosting platform may retain standard "
    "request/application logs."
)


@lru_cache(maxsize=None)
def load_evaluation_summary():
    """Return the headline evaluation figures from the committed evaluation file."""
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    ci_low, ci_high = evaluation["test_metrics"]["roc_auc_bootstrap_95ci"]

    return {
        "roc_auc": evaluation["test_metrics"]["roc_auc"],
        "roc_auc_ci_low": ci_low,
        "roc_auc_ci_high": ci_high,
        "observed_rate": evaluation["dataset"]["prevalence"],
        "csat_observed": evaluation["dataset"]["csat_observed"],
        "dissatisfied": evaluation["dataset"]["positive"],
    }


@st.cache_resource(show_spinner="Loading models (first use only)...")
def load_models():
    """Load the shared inference models once per Streamlit server process.

    Wraps the shared module's own cached loaders; inference itself is
    unchanged and still runs through inference.analyze_ticket().
    """
    inference.load_sentiment_pipeline()
    inference.load_topic_inference_model()
    load_risk_model()
    return True


def use_example():
    """Copy the selected example into the ticket text box."""
    example = st.session_state["example"]

    if example in EXAMPLE_TICKETS:
        st.session_state["ticket_text"] = EXAMPLE_TICKETS[example]


def render_sentiment(result):
    st.subheader("Sentiment")

    label_column, score_column = st.columns(2)
    label_column.metric("Sentiment label", result.sentiment_label.title())
    score_column.metric(
        "Model confidence in this label",
        f"{result.sentiment_score:.4f}",
    )

    st.caption(
        "General-purpose binary sentiment from a pretrained SST-2 model "
        "(DistilBERT fine-tuned on movie-review sentences). It was not trained "
        "on customer-support tickets. The confidence shows how strongly the "
        "model favours its label; it does not measure whether the customer "
        "is dissatisfied."
    )


def render_topic(topic):
    st.subheader("Topic: exploratory topic classification")

    if topic.top_score < TOPIC_LOW_CONFIDENCE_DISPLAY_THRESHOLD:
        st.markdown("**Low-confidence / effectively unclassified**")
        st.markdown(
            f"Closest topic: {topic.topic_label} (topic {topic.topic_id})"
        )
    else:
        st.markdown(f"**{topic.topic_label}** (topic {topic.topic_id})")

    st.markdown(
        f"Top predicted topic score: **{topic.top_score:.3f}**. "
        "Relative model confidence — not a calibrated probability."
    )

    if topic.top_score < TOPIC_LOW_CONFIDENCE_DISPLAY_THRESHOLD:
        st.caption(
            f"This page describes top scores below "
            f"{TOPIC_LOW_CONFIDENCE_DISPLAY_THRESHOLD:.2f} as low-confidence. "
            "That is a display rule for readability, not a model threshold; "
            "the model's prediction is shown unchanged."
        )

    if topic.topic_id == -1:
        st.caption(
            "Topic -1 is BERTopic's outlier group for tickets that did not "
            "fit any discovered topic."
        )

    if topic.is_noise_cluster:
        st.info(
            "The project's topic registry marks this topic as a **noise "
            "cluster**: it groups tickets by shared wording or scraped "
            "content rather than a coherent support issue."
        )

    registry = load_topic_registry()

    st.markdown(f"**Top {len(topic.candidates)} candidate topics**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Topic": candidate.topic_label,
                    "Topic ID": candidate.topic_id,
                    "Relative score": f"{candidate.score:.3f}",
                    "Noise cluster": (
                        "Yes"
                        if registry[candidate.topic_id].is_noise_cluster
                        else "No"
                    ),
                }
                for candidate in topic.candidates
            ]
        ),
        hide_index=True,
    )
    st.caption(
        "Topic labels are human-interpreted exploratory themes, not a "
        "validated issue taxonomy."
    )


def render_experimental_score(score):
    summary = load_evaluation_summary()

    with st.expander(
        "Experimental Model Score — Read With Caution",
        expanded=False,
    ):
        st.markdown(f"**Experimental model score:** {score:.3f}")
        st.markdown(
            "This is a model-derived score from an experimental Gradient "
            "Boosting model. Its evaluation showed **no demonstrated "
            f"predictive discrimination**: ROC-AUC {summary['roc_auc']:.3f} "
            f"(95% CI {summary['roc_auc_ci_low']:.3f}–"
            f"{summary['roc_auc_ci_high']:.3f}), where 0.5 is no better than "
            "chance. It is not a probability, it does not indicate that this "
            "customer is or will be dissatisfied, and it must not be used to "
            "prioritize or triage tickets."
        )

        st.markdown("##### Observed dissatisfaction")
        st.markdown(
            f"{summary['observed_rate']:.2%} of the "
            f"{summary['csat_observed']:,} CSAT-observed tickets "
            f"({summary['dissatisfied']:,} tickets) had a satisfaction "
            "rating of 2 or lower. This is a historical rate across the "
            "dataset, not a property of this ticket."
        )

        st.markdown("##### Experimental model score")
        st.markdown(
            "A model-derived score whose predictive discrimination was not "
            "demonstrated. Differences between scores, including high scores "
            "for clearly positive messages, should not be interpreted."
        )


def run_analysis(text, priority, channel):
    text = text.strip()

    if not text:
        st.warning("Enter some ticket text to analyze.")
        return

    if len(text) > MAX_TEXT_LENGTH:
        st.warning(f"Ticket text must be at most {MAX_TEXT_LENGTH:,} characters.")
        return

    try:
        load_models()
        with st.spinner("Analyzing the ticket (models load on first use)..."):
            result = inference.analyze_ticket(
                text,
                priority=priority,
                channel=channel,
            )
    except Exception as error:
        # Report only the error type so the ticket text never appears in output.
        st.error(f"The analysis could not be completed ({type(error).__name__}).")
        return

    render_sentiment(result)
    render_topic(result.topic)
    render_experimental_score(result.dissatisfaction_risk_score)


def render_analyzer():
    st.info(PRIVACY_NOTICE)

    st.selectbox(
        "Start from an example (optional)",
        [NO_EXAMPLE, *EXAMPLE_TICKETS],
        key="example",
        on_change=use_example,
    )

    st.text_area(
        "Ticket text",
        key="ticket_text",
        max_chars=MAX_TEXT_LENGTH,
        height=150,
        placeholder="Describe a hypothetical customer-support issue...",
    )

    priority_column, channel_column = st.columns(2)
    priority = priority_column.selectbox(
        "Priority",
        PRIORITY_CATEGORIES,
        index=PRIORITY_CATEGORIES.index("Medium"),
        key="priority",
    )
    channel = channel_column.selectbox(
        "Channel",
        CHANNEL_CATEGORIES,
        index=CHANNEL_CATEGORIES.index("Email"),
        key="channel",
    )

    if st.button("Analyze ticket", type="primary", key="analyze"):
        run_analysis(st.session_state["ticket_text"], priority, channel)


def render_about():
    summary = load_evaluation_summary()
    registry = load_topic_registry()
    noise_count = sum(topic.is_noise_cluster for topic in registry.values())

    st.markdown("### Sentiment")
    st.markdown(
        f"""
- Pretrained DistilBERT model fine-tuned on SST-2 (`{inference.SENTIMENT_MODEL_NAME}`).
- Binary, general-purpose sentiment: positive or negative.
- SST-2 consists of movie-review sentences; the model was not trained on customer-support tickets.
- The confidence shown is the model's confidence in its label, not a measure of customer dissatisfaction.
"""
    )

    st.markdown("### Topic classification")
    st.markdown(
        f"""
- The project used BERTopic for exploratory topic discovery across the ticket dataset.
- For interactive use, this demo uses a TF-IDF + Logistic Regression classifier as a stand-in. It agrees with the BERTopic assignments on approximately 65.7% of tickets in its held-out evaluation split.
- Topic labels are human-interpreted exploratory labels, not a validated ground-truth issue taxonomy.
- Scores are relative and uncalibrated. This page describes top scores below {TOPIC_LOW_CONFIDENCE_DISPLAY_THRESHOLD:.2f} as low-confidence; that is a display rule, not a validated threshold.
- {noise_count} of the {len(registry)} registry topics are marked as noise clusters: groupings driven by shared wording or scraped content rather than a coherent support issue.
"""
    )

    st.markdown("### Experimental model score")
    st.markdown(
        f"""
- Gradient Boosting model using priority, channel, sentiment and topic.
- Evaluated against CSAT-grounded dissatisfaction (satisfaction rating of 2 or lower) on {summary['csat_observed']:,} CSAT-observed tickets.
- Held-out ROC-AUC {summary['roc_auc']:.3f} (95% CI {summary['roc_auc_ci_low']:.3f}–{summary['roc_auc_ci_high']:.3f}): no demonstrated predictive discrimination.
- It is included only as an experimental, model-audit output. It is not a validated probability and not a signal for prioritizing tickets in production.
"""
    )

    st.markdown("### Privacy")
    st.markdown(
        "Submitted text is analyzed in memory and is not stored by this "
        "demo. The app loads only model files, never the ticket dataset."
    )


st.set_page_config(page_title="Customer Support Ticket Analyzer")

st.title("Customer Support Ticket Analyzer")
st.markdown(
    "Enter a hypothetical support ticket to see how the project's models "
    "read it: general-purpose sentiment, an exploratory topic, and an "
    "experimental model score. This is a portfolio demonstration, not a "
    "production tool."
)

st.session_state.setdefault("ticket_text", "")

analyzer_tab, about_tab = st.tabs(["Ticket Analyzer", "About these models"])

with analyzer_tab:
    render_analyzer()

with about_tab:
    render_about()
