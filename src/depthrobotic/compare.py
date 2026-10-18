"""Paired task comparisons and exact McNemar statistics."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import comb


def _outcomes(payload: dict) -> dict[tuple[str, int, int], dict]:
    return {(r["suite"], r["task_id"], r["episode"]): r for r in payload["results"]}


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for ``wins`` vs ``losses`` discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum((comb(n, i) for i in range(k + 1))) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class Comparison:
    """A paired comparison of two runs over their shared population."""

    baseline_name: str
    candidate_name: str
    shared: int
    baseline_successes: int
    candidate_successes: int
    gains: list[tuple[str, int, int]] = field(default_factory=list)
    losses: list[tuple[str, int, int]] = field(default_factory=list)
    baseline_only: list = field(default_factory=list)
    candidate_only: list = field(default_factory=list)

    @property
    def p_value(self) -> float:
        return mcnemar_exact(len(self.gains), len(self.losses))

    @property
    def promotable(self) -> bool:
        """Significant at 0.05 with no regressions."""
        return not self.losses and self.gains and (self.p_value < 0.05)

    def report(self) -> str:

        def cases(rows):
            return ", ".join((f"{s}:{t}#{e}" for s, t, e in rows)) or "none"

        lines = [
            f"paired population : {self.shared} episodes",
            f"{self.baseline_name:>18s} : {self.baseline_successes}/{self.shared}",
            f"{self.candidate_name:>18s} : {self.candidate_successes}/{self.shared}",
            f"gains  (+{len(self.gains)}) : {cases(self.gains)}",
            f"losses (-{len(self.losses)}) : {cases(self.losses)}",
            f"exact McNemar p   : {self.p_value:.4f}",
            f"promotable        : {('yes' if self.promotable else 'no')}",
        ]
        if self.baseline_only or self.candidate_only:
            lines.append(
                f"NOT PAIRED        : {len(self.baseline_only)} baseline-only, {len(self.candidate_only)} candidate-only episodes excluded"
            )
        if not self.promotable and self.gains and (not self.losses):
            need = next(
                (n for n in range(len(self.gains), 40) if mcnemar_exact(n, 0) < 0.05),
                None,
            )
            if need:
                lines.append(
                    f"note              : {need} clean wins would reach p<0.05"
                )
        return "\n".join(lines)


def compare(
    baseline: dict,
    candidate: dict,
    *,
    baseline_name="baseline",
    candidate_name="candidate",
) -> Comparison:
    """Compare two payloads over the episodes they have in common."""
    left, right = (_outcomes(baseline), _outcomes(candidate))
    shared = sorted(set(left) & set(right))
    result = Comparison(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        shared=len(shared),
        baseline_successes=sum((1 for key in shared if left[key]["success"])),
        candidate_successes=sum((1 for key in shared if right[key]["success"])),
        baseline_only=sorted(set(left) - set(right)),
        candidate_only=sorted(set(right) - set(left)),
    )
    for key in shared:
        was, now = (bool(left[key]["success"]), bool(right[key]["success"]))
        if now and (not was):
            result.gains.append(key)
        elif was and (not now):
            result.losses.append(key)
    return result


def reference_outcomes(payload: dict) -> dict[str, bool]:
    """Flatten a payload to ``"suite:task#episode" -> success`` for gate fixtures."""
    return {
        f"{s}:{t}#{e}": bool(r["success"])
        for (s, t, e), r in _outcomes(payload).items()
    }
