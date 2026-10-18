#!/usr/bin/env python3
"""Failure taxonomy for a run, and the paired difference between two runs.

    tools/analyze.py results/v30_libero40_gate_r1/results.json
    tools/analyze.py results/live_obs_libero40_r1/results.json \
                     --against results/v30_libero40_gate_r1/results.json

One number per run is not enough to steer by: on forty episodes its 95%
interval is about +-2.6 episodes. Paired symptom counts move first and move
visibly, and they are also what catches a change that buys three episodes while
introducing a new way to fail.

Works on runs recorded before ``diagnosis`` became a field: the classification
is derived from the tool results and events, which were always written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = 0 if total == 0 else round(width * count / total)
    return "#" * filled + "." * (width - filled)


def report(path: Path) -> None:
    from depthrobotic import diagnose

    data = json.loads(path.read_text())
    failures = diagnose.tally(data["results"])
    everything = diagnose.tally(data["results"], failures_only=False)

    print(f"{path}")
    print(
        f"  {data['successes']}/{data['episodes']} successes, {failures['episodes']} failures"
    )
    if not data.get("complete", True):
        print("  INCOMPLETE -- not reportable")

    print("\n  stage reached (mutually exclusive, failures only)")
    for stage in diagnose.STAGES:
        count = failures["stages"].get(stage, 0)
        if count:
            print(f"    {stage:32} {count:>3}  {_bar(count, failures['episodes'])}")

    print("\n  symptoms (overlapping log observations, NOT causes)")
    print(f"    {'':32} {'fail':>4} {'all':>4}")
    for symptom, count in sorted(failures["symptoms"].items(), key=lambda kv: -kv[1]):
        seen = everything["symptoms"].get(symptom, 0)
        note = "   <- also in successes" if seen > count else ""
        print(
            f"    {symptom:32} {count:>4} {seen:>4}  {_bar(count, failures['episodes'])}{note}"
        )


def compare(baseline: Path, candidate: Path) -> None:
    from depthrobotic import compare as compare_module
    from depthrobotic import diagnose

    before = json.loads(baseline.read_text())
    after = json.loads(candidate.read_text())

    print(
        compare_module.compare(
            before,
            after,
            baseline_name=baseline.parent.name,
            candidate_name=candidate.parent.name,
        ).report()
    )

    print("\nfailure-mode delta (baseline -> candidate, failures only)")
    print("  a change that helps should shrink a symptom without growing another.")
    moved = {
        k: v
        for k, v in diagnose.delta(before["results"], after["results"]).items()
        if v[0] != v[1]
    }
    if not moved:
        print("    no symptom or stage count changed")
        return
    for key, (was, now) in sorted(moved.items(), key=lambda kv: kv[1][0] - kv[1][1]):
        arrow = "down" if now < was else "UP  "
        print(f"    {key:32} {was:>3} -> {now:<3}  {arrow}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--against", type=Path, default=None, metavar="BASELINE_RESULTS"
    )
    args = parser.parse_args()

    if args.against:
        compare(args.against, args.run)
    else:
        report(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
