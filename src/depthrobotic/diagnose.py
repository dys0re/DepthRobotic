"""Failure-stage and symptom analysis from recorded observations."""

from __future__ import annotations
from collections import Counter

SYMPTOM_RULES: dict[str, tuple[str, ...]] = {
    "source_uncertain": (
        "identity is unconfirmed",
        "could not confirm",
        "may not be the requested",
    ),
    "source_localization_error": ("Could not find", "outside the reachable workspace"),
    "motion_failure": ("Motion failed", "did not reach target", "No wrist center"),
    "high_contact": ("fingers landed on top", "stall_high"),
    "empty_grasp": ("empty grasp", "closed on nothing", "regrasp empty or dropped"),
    "slip": ("slipped out", "slipped during"),
    "grasp_width_infeasible": ("wider than the 8 cm", "cannot enclose"),
    "placement_unmeasured": (
        "Placement could not be measured",
        "Completion of the specified object relation is unverified",
    ),
    "placement_outside": ("NOT inside", "outside the ~3 cm"),
    "placement_blocked": ("Could not place", "will not fit"),
    "budget_refusal": ("Not enough step budget", "not enough budget"),
    "attempt_limit": ("Max grasp attempts",),
}
FRESH_VIEW_MARKER = "A fresh view is available."
GRASP_TOOLS = frozenset({"grasp", "retry_grasp"})
PLACE_TOOLS = frozenset({"place_object", "retry_place"})
STAGES = (
    "no_pick_place_executor",
    "approach_only",
    "grasp_attempted_no_placement",
    "placement_attempted",
)


def _executor_records(result: dict) -> list[dict]:
    return [
        p
        for p in result.get("perceptions") or []
        if isinstance(p, dict) and "tool_calls" in p
    ]


def _returned_calls(records: list[dict]) -> list[dict]:
    """Tool calls that produced a result, in order."""
    return [
        call
        for record in records
        for call in record["tool_calls"]
        if isinstance(call, dict) and "tool" in call and ("result" in call)
    ]


def stage_of(result: dict) -> str:
    """How far the episode got. Mutually exclusive, so these counts sum."""
    records = _executor_records(result)
    if not records:
        return "no_pick_place_executor"
    tools = {
        call.get("tool")
        for record in records
        for call in record["tool_calls"]
        if isinstance(call, dict) and "round" in call
    }
    if tools & PLACE_TOOLS:
        return "placement_attempted"
    if tools & GRASP_TOOLS:
        return "grasp_attempted_no_placement"
    return "approach_only"


def symptoms_of(result: dict) -> list[str]:
    """Overlapping log observations. Not causes, and not exclusive to failures."""
    records = _executor_records(result)
    returned = _returned_calls(records)
    texts = [str(call["result"]) for call in returned]
    found = {
        name
        for name, phrases in SYMPTOM_RULES.items()
        if any((phrase in text for text in texts for phrase in phrases))
    }
    if "did not reach target" in str(result.get("error") or ""):
        found.add("motion_failure")
    steps, budget = (result.get("sim_steps", 0), result.get("step_budget", 0))
    if steps >= budget:
        found.add("horizon_exhausted")
    if not result.get("success") and budget - steps >= 100:
        found.add("failed_with_100_steps_left")
    kinds = Counter(
        (
            recovery.get("failure")
            for record in records
            for call in record["tool_calls"]
            if isinstance(call, dict)
            and isinstance((recovery := call.get("recovery")), dict)
            if recovery.get("failure")
        )
    )
    if any((count >= 2 for count in kinds.values())):
        found.add("repeated_failure_kind")
    for record in records:
        returned_here = [
            call
            for call in record["tool_calls"]
            if isinstance(call, dict) and "tool" in call and ("result" in call)
        ]
        for index, call in enumerate(returned_here):
            if FRESH_VIEW_MARKER in str(call["result"]):
                found.add("active_observation")
                following = (
                    returned_here[index + 1] if index + 1 < len(returned_here) else None
                )
                if following is not None and following.get("tool") == "done":
                    found.add("observe_then_done")
    return sorted(found)


def first_issue(result: dict) -> dict | None:
    """The first tool result that reported something going wrong."""
    markers = (
        "Grasp FAILED",
        "Motion failed",
        "Error:",
        "cannot enclose",
        "NOT inside",
        "Placement could not be measured",
    )
    for call in _returned_calls(_executor_records(result)):
        text = str(call["result"])
        if any((marker in text for marker in markers)):
            return {
                "tool": call["tool"],
                "text": text.split("Attempts so far:")[0][:240],
            }
    return None


def classify(result: dict) -> dict:
    """The diagnosis attached to one episode result."""
    plan = result.get("plan") or []
    return {
        "stage": stage_of(result),
        "symptoms": symptoms_of(result),
        "plan_family": "+".join((step.get("action", "?") for step in plan)),
        "multi_step": len(plan) > 1,
        "remaining_steps": result.get("step_budget", 0) - result.get("sim_steps", 0),
        "first_issue": first_issue(result),
    }


def tally(results: list[dict], *, failures_only: bool = True) -> dict:
    """Stage and symptom counts over a run, for paired comparison between runs."""
    rows = (
        [r for r in results if not r.get("success")] if failures_only else list(results)
    )
    diagnoses = [r.get("diagnosis") or classify(r) for r in rows]
    return {
        "episodes": len(rows),
        "stages": dict(Counter((d["stage"] for d in diagnoses))),
        "symptoms": dict(Counter((s for d in diagnoses for s in d["symptoms"]))),
    }


def delta(baseline: list[dict], candidate: list[dict]) -> dict[str, tuple[int, int]]:
    """Per-symptom failure counts, baseline -> candidate.

    This is the sensitive instrument: on forty episodes a change can move
    ``empty_grasp`` from 25 to 12 while the total moves by one, and the total
    alone cannot distinguish that from noise.
    """
    before, after = (tally(baseline), tally(candidate))
    keys = sorted({**before["symptoms"], **after["symptoms"]}) + sorted(
        {**before["stages"], **after["stages"]}
    )
    out = {}
    for key in dict.fromkeys(keys):
        source = (
            "symptoms"
            if key in {**before["symptoms"], **after["symptoms"]}
            else "stages"
        )
        out[key] = (before[source].get(key, 0), after[source].get(key, 0))
    return out
