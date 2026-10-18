"""signals: v30 control behavior with explicit dependencies."""

from __future__ import annotations


class TaskCompleted(Exception):
    """Raised immediately after the benchmark success predicate becomes true."""

    def __init__(self, obs):
        super().__init__("LIBERO success predicate became true")
        self.obs = obs


class StepBudgetExceeded(Exception):
    """Raised before an action would exceed the suite's official horizon."""
