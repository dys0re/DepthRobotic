"""perception / parsing: v30 control behavior with explicit dependencies."""

from __future__ import annotations
import json
import numpy as np
import re


def parse_or_recover_json(response):
    start = response.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(response[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    recovered = {}
    for key in ("box", "point", "contact", "motion", "destination", "start", "end"):
        match = re.search(
            f"""["\\']{key}["\\']\\s*:\\s*\\[([^\\]\\}}\\n]+)""", response
        )
        if not match:
            continue
        numbers = re.findall("[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)", match.group(1))
        if numbers:
            recovered[key] = [float(number) for number in numbers]
    held = re.search(
        "[\"\\']held[\"\\']\\s*:\\s*(true|false)", response, flags=re.IGNORECASE
    )
    if held:
        recovered["held"] = held.group(1).lower() == "true"
    if recovered:
        return recovered
    raise ValueError(f"No recoverable JSON object in response: {response!r}")


def safe_ask_structured(grounder, image, prompt, max_tokens=160):
    try:
        response, _ = grounder.generate_response(image, prompt, max_tokens)
        return (response, parse_or_recover_json(response))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        response = locals().get("response", "")
        return (f"{response}\nPARSE_ERROR: {error}".strip(), {})


def valid_box(value):
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        box = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(box).all()
        and (box >= 0).all()
        and (box <= 1000).all()
        and (box[2] > box[0] + 5)
        and (box[3] > box[1] + 5)
    )


def normalize_box(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if np.isfinite(box).all() and (box >= 0).all() and (box <= 1.0).all():
        box *= 1000.0
    result = box.tolist()
    return result if valid_box(result) else None


def valid_point(value):
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(point).all() and (point >= 0).all() and (point <= 1000).all()
    )


def normalize_point(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(point).all() or (point < 0).any():
        return None
    if (point <= 1.0).all():
        point *= 1000.0
    result = point.tolist()
    return result if valid_point(result) else None


def parse_json_object(text):
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in response: {text!r}")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError(f"Response JSON is not an object: {value!r}")
    return value


def valid_vector(value):
    if not isinstance(value, list) or len(value) != 2:
        return False
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.linalg.norm(vector) >= 0.1 and np.all(np.abs(vector) <= 1.5))
