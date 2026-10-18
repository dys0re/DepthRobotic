"""Integrity checks for episode results and benchmark populations."""

from __future__ import annotations
from math import isclose
from . import protocol

REQUIRED_EPISODE_FIELDS = (
    "suite",
    "task_id",
    "episode",
    "instruction",
    "success",
    "sim_steps",
    "step_budget",
)


class ValidationError(AssertionError):
    """A result that must not be used as evidence."""


def check_episode(result: dict, *, min_sim_steps: int = 1) -> None:
    """Validate one finished episode. Raises rather than warns.

    Called immediately after each episode so a broken run stops at episode one
    instead of after three hours.
    """
    missing = [key for key in REQUIRED_EPISODE_FIELDS if key not in result]
    if missing:
        raise ValidationError(
            f"Episode result missing fields {missing}: {result.get('suite')}"
        )
    case = f"{result['suite']}:{result['task_id']} ep{result['episode']}"
    if type(result["success"]) is not bool:
        raise ValidationError(f"{case} success must be a boolean")
    for key in ("task_id", "episode", "sim_steps", "step_budget"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValidationError(f"{case} {key} must be a nonnegative integer")
    if result["step_budget"] < 1:
        raise ValidationError(f"{case} step_budget must be positive")
    error = result.get("error") or ""
    if (
        error.startswith(
            (
                "NameError:",
                "AttributeError:",
                "ImportError:",
                "ModuleNotFoundError:",
                "FileNotFoundError:",
                "OSError:",
                "OutOfMemoryError:",
            )
        )
        or "CUDA out of memory" in error
    ):
        raise ValidationError(
            f"{case} infrastructure error must not count as a task failure: {error}"
        )
    steps = result["sim_steps"]
    if steps < min_sim_steps:
        raise ValidationError(
            f"{case} advanced the simulator {steps} steps. An episode that never ran is a broken run, not a failed episode; it must not be counted. error={result.get('error')!r}"
        )
    if result["success"] and result.get("error"):
        raise ValidationError(
            f"{case} is marked successful and also carries error {result['error']!r}"
        )
    if steps > int(result["step_budget"]):
        raise ValidationError(
            f"{case} used {steps} steps beyond the {result['step_budget']}-step budget this run enforces"
        )
    official = result.get("official_horizon")
    if official is not None and int(result["step_budget"]) < int(official):
        raise ValidationError(
            f"{case} ran with a {result['step_budget']}-step budget, below the official horizon of {official}. A shortened horizon is not comparable with any reported number."
        )
    if result.get("within_official_horizon") and (not result["success"]):
        raise ValidationError(
            f"{case} is within the official horizon but not a success"
        )


def check_payload(payload: dict) -> None:
    """Validate a run payload's internal consistency."""
    results = payload.get("results", [])
    if payload.get("episodes") != len(results):
        raise ValidationError(
            f"episodes={payload.get('episodes')} but {len(results)} results are present"
        )
    counted = sum((1 for row in results if row["success"]))
    if payload.get("successes") != counted:
        raise ValidationError(
            f"successes={payload.get('successes')} but {counted} results are successful"
        )
    by_suite_total = sum(
        (entry["successes"] for entry in payload.get("by_suite", {}).values())
    )
    if by_suite_total != counted:
        raise ValidationError(
            f"by_suite sums to {by_suite_total} successes but results contain {counted}. Aggregates must be derived from the results list, never recomputed by hand."
        )
    by_suite_episodes = sum(
        (entry["episodes"] for entry in payload.get("by_suite", {}).values())
    )
    if by_suite_episodes != len(results):
        raise ValidationError(
            f"by_suite covers {by_suite_episodes} episodes but {len(results)} were run"
        )
    expected_complete = len(results) == payload.get("planned_episodes")
    if bool(payload.get("complete")) != expected_complete:
        raise ValidationError(
            f"complete={payload.get('complete')} contradicts {len(results)}/{payload.get('planned_episodes')} episodes"
        )
    suites = set((r["suite"] for r in results)) | set(payload.get("by_suite", {}))
    for suite in suites:
        rows = [r for r in results if r["suite"] == suite]
        entry = payload.get("by_suite", {}).get(suite, {})
        wins = sum((bool(r["success"]) for r in rows))
        if entry.get("successes") != wins or entry.get("episodes") != len(rows):
            raise ValidationError(f"by_suite entry for {suite} disagrees with its rows")
        expected_rate = wins / len(rows) if rows else 0.0
        if "success_rate" in entry and (
            not isclose(entry["success_rate"], expected_rate)
        ):
            raise ValidationError(f"by_suite rate for {suite} disagrees with its rows")
    if "success_rate" in payload:
        expected_rate = counted / len(results) if results else 0.0
        if not isclose(payload["success_rate"], expected_rate):
            raise ValidationError("success_rate disagrees with results")
    config = payload.get("configuration", {})
    if config.get("cases"):
        cases = config["cases"]
        cases = (
            protocol.parse_cases(cases)
            if isinstance(cases, str)
            else [tuple(c) for c in cases]
        )
        first = config.get("first_episode", 0)
        population = {
            (s, t, e)
            for s, t in cases
            for e in range(first, first + config["episodes"])
        }
        actual = {(r["suite"], r["task_id"], r["episode"]) for r in results}
        if not actual <= population or payload.get("planned_episodes") != len(
            population
        ):
            raise ValidationError("Results do not match the configured population")
        if payload.get("complete") and actual != population:
            raise ValidationError(
                "Complete run does not cover the configured population"
            )
    duplicates = len(results) - len(
        {(r["suite"], r["task_id"], r["episode"]) for r in results}
    )
    if duplicates:
        raise ValidationError(f"{duplicates} duplicate (suite, task, episode) rows")


def require_reportable(payload: dict, *, allow_dirty: bool = False) -> None:
    """Refuse to quote a score from a run that is not a complete measurement."""
    check_payload(payload)
    if not payload.get("complete"):
        raise ValidationError(
            f"Run covers {payload['episodes']}/{payload['planned_episodes']} planned episodes. A truncated run is not a score; finish or resume it before reporting."
        )
    git = payload.get("provenance", {}).get("git", {})
    if git.get("dirty") and (not allow_dirty):
        raise ValidationError(
            f"Run was produced from a dirty working tree, so its code cannot be identified. Dirty paths: {git.get('dirty_paths')}"
        )
    if not git.get("commit") and (not allow_dirty):
        raise ValidationError(
            "Run records no git commit; its code cannot be identified"
        )
