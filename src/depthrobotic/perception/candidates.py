"""perception / candidates: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.skills.pick_place.locate import name_tokens, parse_relation
import re


class RecordingProposer:
    """Wraps the GroundingDINO proposer so the boxes behind v17's pool entries are recoverable for drawing."""

    def __init__(self, inner):
        self.inner = inner
        self.boxes = []

    def detect(self, image, phrases, prefix="G"):
        boxes, record = self.inner.detect(image, phrases, prefix)
        self.boxes.extend(boxes)
        return (boxes, record)

    def metadata(self):
        return self.inner.metadata()


ELEVATED_MIN_DZ = 0.035


def _target_and_relation(desc, instruction):
    parsed = parse_relation(desc, instruction)
    return (parsed["target"].strip() if parsed else str(desc).strip(), parsed)


GDINO_MIN_BOXES = 3


def generic_variants(description):
    """Generic synonyms used ONLY to widen recall when the specific phrase finds too little.  They are
    never used for ranking: in the gate they outscored the real target and drove rank-1 from 6 to 2.
    """
    text = re.sub("\\s+", " ", str(description).lower().strip())
    out: list[str] = []
    if "bowl" in text or "cup" in text or "mug" in text:
        out += ["bowl", "mug", "cup"]
    if "pot" in text:
        out += ["pot"]
    if "book" in text:
        out += ["book"]
    if "bottle" in text:
        out += ["bottle"]
    if any(
        (
            w in text
            for w in (
                "soup",
                "sauce",
                "ketchup",
                "milk",
                "butter",
                "pudding",
                "juice",
                "cheese",
                "dressing",
                "cream",
                "box",
                "carton",
                "can",
            )
        )
    ):
        out += ["food package", "can", "bottle", "carton", "box"]
    seen: list[str] = []
    for v in out:
        if v not in seen and v != text:
            seen.append(v)
    return seen[:5]


def _tokens(text):
    return {t for t in name_tokens(text) if len(t) >= 3}


def _relation_from_record(record):
    """Find the anchor-first grounding record without adding another VLM call."""
    for call in reversed(record.get("tool_calls", [])):
        if not isinstance(call, dict):
            continue
        for value in call.values():
            if (
                isinstance(value, dict)
                and value.get("parsed")
                and value.get("anchor_points")
            ):
                return value
    return None


RELATION_RADIUS = {
    "on": 260.0,
    "on top of": 260.0,
    "in": 260.0,
    "inside": 260.0,
    "next to": 360.0,
    "near": 360.0,
    "beside": 360.0,
    "by": 360.0,
    "between": 300.0,
}
MAX_ALTERNATES = 8
MARK_STYLE = "circle"
MIN_BUDGET_FOR_LAST_RESORT = 130
OCR_BOILERPLATE = {
    "json",
    "object",
    "recoverable",
    "response",
    "parse",
    "error",
    "visible",
    "image",
    "centre",
    "center",
    "words",
    "printed",
    "label",
    "product",
    "cannot",
    "unable",
    "sorry",
    "unknown",
    "none",
    "text",
}


def content_tokens(text, min_len=3):
    return {t for t in name_tokens(text) if len(t) >= min_len} - OCR_BOILERPLATE


def clean_model_text(raw):
    """safe_ask appends 'PARSE_ERROR: ...' to every plain-text answer (the prompt asks for prose, so
    JSON parsing always fails).  v12 stripped only the token 'parse_error', leaving boilerplate words
    that then matched as if they were printed on the object."""
    text = str(raw or "")
    cut = text.upper().find("PARSE_ERROR")
    if cut >= 0:
        text = text[:cut]
    return text.lower()


MAX_TEXT_SCAN = 12
