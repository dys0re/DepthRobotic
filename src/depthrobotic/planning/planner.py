"""planning / planner: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.perception.parsing import safe_ask_structured
from depthrobotic.prompts import plan_prompt
from typing import Any
from typing import Callable
import re


def _split_clauses(text):
    parts, buf = ([], "")
    for chunk in re.split(
        " and (?=(?:put|place|close|open|turn|pick|stack|push|move)\\b)", text
    ):
        if buf and re.match("(?:place|put) it ", chunk):
            buf = buf + " and " + chunk
        else:
            if buf:
                parts.append(buf)
            buf = chunk
    if buf:
        parts.append(buf)
    return parts


PLAN_ACTIONS = ("pick_place", "articulate", "push_object", "toggle")


def parse_model_plan(grounder, image, instruction):
    response, value = safe_ask_structured(
        grounder, image, plan_prompt(instruction), 256
    )
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > 4:
        raise RuntimeError(f"Invalid DepthVLM plan: {response}")
    normalized = []
    for step in steps:
        if not isinstance(step, dict) or step.get("action") not in PLAN_ACTIONS:
            raise RuntimeError(f"Invalid plan step: {step!r}")
        if step["action"] in ("pick_place", "push_object"):
            if not isinstance(step.get("source"), str) or not isinstance(
                step.get("destination"), str
            ):
                raise RuntimeError(f"Missing source/destination: {step!r}")
        elif step["action"] == "articulate":
            if not isinstance(step.get("target"), str) or step.get("mode") not in (
                "open",
                "close",
            ):
                raise RuntimeError(f"Invalid articulation step: {step!r}")
        elif step["action"] == "toggle":
            if not isinstance(step.get("target"), str) or step.get("mode") not in (
                "on",
                "off",
            ):
                raise RuntimeError(f"Invalid toggle step: {step!r}")
        text = " " + re.sub("[^a-z]+", " ", instruction.lower()) + " "
        explicitly_requested = {
            "pick_place": any(
                (word in text for word in (" pick ", " place ", " put ", " move "))
            ),
            "push_object": " push " in text,
            "articulate": f" {step.get('mode', '')} " in text,
            "toggle": f"turn {step.get('mode', '')}" in text
            or f"switch {step.get('mode', '')}" in text,
        }[step["action"]]
        if explicitly_requested:
            normalized.append(step)
    if not normalized:
        raise RuntimeError(
            f"DepthVLM plan contains no instruction-supported action: {response}"
        )
    return (response, normalized)


def robust_parse_plan(grounder, image, instruction):
    text = re.sub("\\s+", " ", instruction.lower().strip())
    compound_patterns = (
        (
            re.fullmatch("turn on the (.+?) and put the (.+?) on it", text),
            lambda match: [
                {"action": "toggle", "target": match.group(1), "mode": "on"},
                {
                    "action": "pick_place",
                    "source": match.group(2),
                    "destination": f"on {match.group(1)}",
                },
            ],
        ),
        (
            re.fullmatch("open the (.+?) and put the (.+?) inside", text),
            lambda match: [
                {"action": "articulate", "target": match.group(1), "mode": "open"},
                {
                    "action": "pick_place",
                    "source": match.group(2),
                    "destination": f"inside {match.group(1)}",
                },
            ],
        ),
        (
            re.fullmatch("put the (.+?) in the (.+?) and close it", text),
            lambda match: [
                {
                    "action": "pick_place",
                    "source": match.group(1),
                    "destination": f"inside {match.group(2)}",
                },
                {"action": "articulate", "target": match.group(2), "mode": "close"},
            ],
        ),
        (
            re.fullmatch("put both the (.+?) and the (.+?) (in|on) the (.+?)", text),
            lambda match: [
                {
                    "action": "pick_place",
                    "source": match.group(1),
                    "destination": f"{match.group(3)} {match.group(4)}",
                },
                {
                    "action": "pick_place",
                    "source": match.group(2),
                    "destination": f"{match.group(3)} {match.group(4)}",
                },
            ],
        ),
        (
            re.fullmatch(
                "put the (.+?) (on|in) the (.+?) and put the (.+?) (on|in|to the right of) the (.+?)",
                text,
            ),
            lambda match: [
                {
                    "action": "pick_place",
                    "source": match.group(1),
                    "destination": f"{match.group(2)} {match.group(3)}",
                },
                {
                    "action": "pick_place",
                    "source": match.group(4),
                    "destination": f"{match.group(5)} {match.group(6)}",
                },
            ],
        ),
    )
    for match, builder in compound_patterns:
        if match is not None:
            return ("DETERMINISTIC_COMPOUND_GRAMMAR", builder(match))
    try:
        return parse_model_plan(grounder, image, instruction)
    except RuntimeError as original_error:
        match = re.fullmatch(
            "(?:put|place) (?:the )?(.+?) (in|on|onto) (?:the )?(.+)", text
        )
        if match:
            source, relation, target = match.groups()
            destination = f"{relation} {target}"
            return (
                f"GRAMMAR_FALLBACK after {original_error}",
                [
                    {
                        "action": "pick_place",
                        "source": source,
                        "destination": destination,
                    }
                ],
            )
        raise


PICK_WORDS = (" pick ", " place ", " put ", " move ", " stack ")


def vlm_plan_widened(grounder, image, instruction):
    """generic.parse_plan with the lexical gate widened to the ontology's verb families."""
    response, value = safe_ask_structured(
        grounder, image, plan_prompt(instruction), 256
    )
    steps = value.get("steps") if isinstance(value, dict) else None
    if not isinstance(steps, list) or not steps or len(steps) > 4:
        raise RuntimeError(f"Invalid DepthVLM plan: {response}")
    text = " " + re.sub("[^a-z]+", " ", instruction.lower()) + " "
    normalized = []
    for step in steps:
        if not isinstance(step, dict) or step.get("action") not in PLAN_ACTIONS:
            raise RuntimeError(f"Invalid plan step: {step!r}")
        if step["action"] in ("pick_place", "push_object"):
            if not isinstance(step.get("source"), str) or not isinstance(
                step.get("destination"), str
            ):
                raise RuntimeError(f"Missing source/destination: {step!r}")
        elif step["action"] == "articulate":
            if not isinstance(step.get("target"), str) or step.get("mode") not in (
                "open",
                "close",
            ):
                raise RuntimeError(f"Invalid articulation step: {step!r}")
        elif step["action"] == "toggle":
            if not isinstance(step.get("target"), str) or step.get("mode") not in (
                "on",
                "off",
            ):
                raise RuntimeError(f"Invalid toggle step: {step!r}")
        requested = {
            "pick_place": any((w in text for w in PICK_WORDS)),
            "push_object": " push " in text or " slide " in text,
            "articulate": f" {step.get('mode', '')} " in text,
            "toggle": f"turn {step.get('mode', '')}" in text
            or f"switch {step.get('mode', '')}" in text,
        }[step["action"]]
        if requested:
            normalized.append(step)
    if not normalized:
        raise RuntimeError(
            f"DepthVLM plan contains no instruction-supported action: {response}"
        )
    return (response, normalized)


REL = "(on top of|onto|on|inside|into|in|to the right of|to the left of|to the front of|to the back of|next to|behind|in front of|under|below|above)"


def _object_of(step):
    if step["action"] in ("articulate", "toggle"):
        return step["target"]
    dest = step["destination"]
    for rel in sorted(
        re.findall("\\(([^()]+)\\)", REL)[0].split("|"), key=len, reverse=True
    ):
        if dest.startswith(rel + " "):
            return dest[len(rel) + 1 :]
    return dest


Step = dict[str, Any]
_ART = "(?:the |a |an )?"


def _dest(rel, target):
    rel = {"onto": "on", "into": "in"}.get(rel, rel)
    return f"{rel} {target}"


def _pick(src, rel, tgt):
    return {
        "action": "pick_place",
        "source": src.strip(),
        "destination": _dest(rel, tgt.strip()),
    }


CLAUSE_TEMPLATES: list[
    tuple[str, Callable[[re.Match, str | None], list[Step] | None]]
] = [
    (
        f"(open|close) {_ART}(.+)",
        lambda m, prev: (
            [
                {
                    "action": "articulate",
                    "target": prev if m[2].strip() == "it" else m[2].strip(),
                    "mode": m[1],
                }
            ]
            if m[2].strip() != "it" or prev
            else None
        ),
    ),
    (
        f"turn (on|off) {_ART}(.+)",
        lambda m, prev: (
            [
                {
                    "action": "toggle",
                    "target": prev if m[2].strip() == "it" else m[2].strip(),
                    "mode": m[1],
                }
            ]
            if m[2].strip() != "it" or prev
            else None
        ),
    ),
    (
        f"(?:put|place) (?:the |a |an )?(.+?) {REL} (?:the |a |an )?(.+)",
        lambda m, prev: (
            [_pick(m[1], m[2], prev if m[3].strip() == "it" else m[3])]
            if m[3].strip() != "it" or prev
            else None
        ),
    ),
    (
        f"pick up {_ART}(.+?) and (?:place|put) it {REL} {_ART}(.+)",
        lambda m, prev: [_pick(m[1], m[2], m[3])],
    ),
]
TEMPLATES: list[tuple[str, Callable[[re.Match], list[Step]]]] = [
    (
        f"stack {_ART}(.+?) on {_ART}(.+?) and place them (in|on|inside) {_ART}(.+)",
        lambda m: [_pick(m[1], "on top of", m[2]), _pick(m[2], m[3], m[4])],
    ),
    (
        f"stack {_ART}(.+?) on(?: top of)? {_ART}(.+)",
        lambda m: [_pick(m[1], "on top of", m[2])],
    ),
    (
        f"pick up {_ART}(.+?) and (?:place|put) it {REL} {_ART}((?:(?! and ).)+)",
        lambda m: [_pick(m[1], m[2], m[3])],
    ),
    (
        f"(?:put|place) {_ART}(.+?) {REL} {_ART}((?:(?! and ).)+)",
        lambda m: [_pick(m[1], m[2], m[3])],
    ),
    (
        f"move {_ART}(.+?) {REL} {_ART}((?:(?! and ).)+)",
        lambda m: [_pick(m[1], m[2], m[3])],
    ),
    (
        f"(open|close) {_ART}((?:(?! and ).)+)",
        lambda m: [{"action": "articulate", "target": m[2].strip(), "mode": m[1]}],
    ),
    (
        f"turn (on|off) {_ART}((?:(?! and ).)+)",
        lambda m: [{"action": "toggle", "target": m[2].strip(), "mode": m[1]}],
    ),
    (
        f"push {_ART}(.+?) to {_ART}(.+)",
        lambda m: [
            {
                "action": "push_object",
                "source": m[1].strip(),
                "destination": m[2].strip(),
            }
        ],
    ),
]


def _norm(instruction):
    text = re.sub("\\s+", " ", instruction.lower().strip())
    return text.rstrip(".").strip()


def grammar_plan(instruction):
    text = _norm(instruction)
    for pattern, build in TEMPLATES:
        m = re.fullmatch(pattern, text)
        if m:
            return build(m)
    clauses = _split_clauses(text)
    if len(clauses) < 2:
        return None
    steps: list[Step] = []
    prev_obj: str | None = None
    for clause in clauses:
        parsed = None
        for pattern, build in CLAUSE_TEMPLATES:
            m = re.fullmatch(pattern, clause.strip())
            if m:
                parsed = build(m, prev_obj)
                break
        if not parsed:
            return None
        steps.extend(parsed)
        prev_obj = _object_of(parsed[-1])
    return steps or None


def plan(grounder, image, instruction):
    try:
        reason, steps = robust_parse_plan(grounder, image, instruction)
        n_clauses = len(_split_clauses(_norm(instruction)))
        if len(steps) < n_clauses:
            grammar = grammar_plan(instruction)
            if grammar and len(grammar) >= n_clauses:
                return (
                    f"GRAMMAR_V2 (base plan covered {len(steps)}/{n_clauses} clauses: {reason[:60]})",
                    grammar,
                )
        return (reason, steps)
    except RuntimeError as base_error:
        steps = grammar_plan(instruction)
        if steps:
            return (f"GRAMMAR_V2 after {str(base_error)[:80]}", steps)
        try:
            response, steps = vlm_plan_widened(grounder, image, instruction)
            return (
                f"VLM_WIDENED_GATE after {str(base_error)[:80]} :: {response[:120]}",
                steps,
            )
        except RuntimeError:
            raise base_error
