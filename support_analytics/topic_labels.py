"""Load the topic-label registry: the single source of topic_id -> topic_label.

Topic IDs are frozen outputs of the committed BERTopic run. This module only
interprets them; it never assigns or changes topic IDs.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "models" / "topic_labels.json"

SCHEMA_VERSION = 1
EXPECTED_TOPIC_IDS = frozenset(range(-1, 29))

REQUIRED_FIELDS = {
    "id": int,
    "label": str,
    "previous_label": str,
    "group": str,
    "is_noise_cluster": bool,
    "confidence": str,
    "evidence": str,
    "review_status": str,
}

CONFIDENCE_VALUES = {"high", "medium", "low"}
REVIEW_STATUS_VALUES = {"proposed", "approved"}


class TopicRegistryError(ValueError):
    """Raised when the topic-label registry is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class TopicLabel:
    id: int
    label: str
    previous_label: str
    group: str
    is_noise_cluster: bool
    confidence: str
    evidence: str
    review_status: str


def _validate_topic(entry, position):
    if not isinstance(entry, dict):
        raise TopicRegistryError(f"Topic entry {position} is not an object.")

    missing = sorted(set(REQUIRED_FIELDS) - set(entry))
    if missing:
        raise TopicRegistryError(
            f"Topic entry {position} is missing fields: {', '.join(missing)}"
        )

    for field, expected_type in REQUIRED_FIELDS.items():
        value = entry[field]

        # bool is a subclass of int, so reject it explicitly for integer IDs.
        if expected_type is int and isinstance(value, bool):
            raise TopicRegistryError(f"Topic entry {position} has a boolean id.")

        if not isinstance(value, expected_type):
            raise TopicRegistryError(
                f"Topic entry {position} field '{field}' must be "
                f"{expected_type.__name__}, got {type(value).__name__}."
            )

    if not entry["label"].strip() or entry["label"] != entry["label"].strip():
        raise TopicRegistryError(
            f"Topic {entry['id']} has an empty or untrimmed label."
        )

    if entry["confidence"] not in CONFIDENCE_VALUES:
        raise TopicRegistryError(
            f"Topic {entry['id']} has invalid confidence '{entry['confidence']}'."
        )

    if entry["review_status"] not in REVIEW_STATUS_VALUES:
        raise TopicRegistryError(
            f"Topic {entry['id']} has invalid review_status "
            f"'{entry['review_status']}'."
        )

    return TopicLabel(**{field: entry[field] for field in REQUIRED_FIELDS})


def parse_topic_registry(document):
    """Validate a registry document and return {topic_id: TopicLabel}."""
    if not isinstance(document, dict):
        raise TopicRegistryError("Topic registry must be a JSON object.")

    if document.get("schema_version") != SCHEMA_VERSION:
        raise TopicRegistryError(
            f"Unsupported topic registry schema_version "
            f"{document.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )

    entries = document.get("topics")
    if not isinstance(entries, list):
        raise TopicRegistryError("Topic registry must contain a 'topics' list.")

    topics = [
        _validate_topic(entry, position)
        for position, entry in enumerate(entries)
    ]

    topic_ids = [topic.id for topic in topics]
    duplicate_ids = sorted({i for i in topic_ids if topic_ids.count(i) > 1})
    if duplicate_ids:
        raise TopicRegistryError(f"Duplicate topic IDs in registry: {duplicate_ids}")

    labels = [topic.label for topic in topics]
    duplicate_labels = sorted({x for x in labels if labels.count(x) > 1})
    if duplicate_labels:
        raise TopicRegistryError(f"Duplicate topic labels in registry: {duplicate_labels}")

    if set(topic_ids) != EXPECTED_TOPIC_IDS:
        missing = sorted(EXPECTED_TOPIC_IDS - set(topic_ids))
        unexpected = sorted(set(topic_ids) - EXPECTED_TOPIC_IDS)
        raise TopicRegistryError(
            f"Registry topic IDs do not match the frozen topic run "
            f"(missing: {missing}, unexpected: {unexpected})."
        )

    return {topic.id: topic for topic in sorted(topics, key=lambda t: t.id)}


def read_topic_registry(path):
    """Read and validate a registry file from an explicit path (no caching)."""
    path = Path(path)

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TopicRegistryError(f"Topic registry not found: {path}") from error
    except json.JSONDecodeError as error:
        raise TopicRegistryError(f"Topic registry is not valid JSON: {error}") from error

    return parse_topic_registry(document)


@lru_cache(maxsize=None)
def load_topic_registry():
    """Return the validated project registry as {topic_id: TopicLabel}."""
    return read_topic_registry(REGISTRY_PATH)


def topic_label_map():
    """Return {topic_id: label} for every frozen topic ID."""
    return {
        topic_id: topic.label
        for topic_id, topic in load_topic_registry().items()
    }


def get_topic_label(topic_id):
    """Return the label for a topic ID, raising KeyError for unknown IDs."""
    registry = load_topic_registry()

    try:
        is_integer = not isinstance(topic_id, bool) and int(topic_id) == topic_id
    except (TypeError, ValueError):
        is_integer = False

    if not is_integer:
        raise KeyError(f"Invalid topic ID: {topic_id!r}")

    try:
        return registry[int(topic_id)].label
    except KeyError:
        raise KeyError(f"Unknown topic ID: {topic_id!r}") from None
