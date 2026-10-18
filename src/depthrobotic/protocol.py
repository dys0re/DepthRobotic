"""LIBERO populations, horizons, and observation conventions."""

from __future__ import annotations

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_40 = tuple((f"{suite}:{task}" for suite in SUITES for task in range(10)))
LIBERO_90 = tuple((f"libero_90:{task}" for task in range(90)))
TASK_BUDGETS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
PLAN_ACTIONS = ("pick_place", "articulate", "push_object", "toggle")
IMAGE_SIZE = 832
CAMERA_NAME = "agentview"
CAMERA_FOVY = 45.0
ORIENT_FIX = 0
SETTLE_STEPS = 10
SETTLE_GRIPPER = -1.0
SETTLE_PHASE = "initial_settle"
SUPPORT_Z_OFFSET = 0.24
SEED = 0
TORCH_THREADS = 4
MAX_TOOL_ROUNDS = 12
MAX_GRASP_ATTEMPTS = 3


def budget_for(suite: str) -> int:
    """Official action horizon for a suite, or raise for an unknown suite."""
    try:
        return TASK_BUDGETS[suite]
    except KeyError:
        raise ValueError(
            f"No official horizon registered for suite {suite!r}"
        ) from None


def parse_cases(text: str) -> list[tuple[str, int]]:
    """Parse "suite:task,suite:task" into validated, de-duplicated pairs."""
    text = {
        "libero_40": ",".join(LIBERO_40),
        "libero40": ",".join(LIBERO_40),
        "libero_90": ",".join(LIBERO_90),
        "libero90": ",".join(LIBERO_90),
    }.get(text, text)
    cases: list[tuple[str, int]] = []
    for part in (piece.strip() for piece in text.split(",")):
        if not part:
            continue
        suite, _, task = part.partition(":")
        if not task.isdigit():
            raise ValueError(f"Malformed case {part!r}; expected suite:task")
        budget_for(suite)
        task_id = int(task)
        limit = 90 if suite == "libero_90" else 10
        if not 0 <= task_id < limit:
            raise ValueError(f"Task id out of range for {suite}: {task_id}")
        cases.append((suite, task_id))
    if len(set(cases)) != len(cases):
        raise ValueError("Duplicate cases in the population")
    if not cases:
        raise ValueError("Empty case population")
    return cases
