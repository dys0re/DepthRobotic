"""Compare benchmark runs against fixed per-task reference outcomes and action counts."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from depthrobotic import compare, validate

DEFAULT_REFERENCE = REPO_ROOT / "tests" / "reference" / "v30_libero40.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, type=Path)
    parser.add_argument(
        "--require",
        choices=["outcomes", "exact"],
        default="outcomes",
        help="outcomes: same success bits; exact: also identical step counts",
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="gate a run still in progress"
    )
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    reference = json.loads(args.reference.read_text())
    validate.check_payload(run)
    if not args.allow_partial:
        validate.require_reportable(run)
    result = compare.compare(
        reference, run, baseline_name="v30 reference", candidate_name="this run"
    )
    print(result.report())
    steps = {
        (r["suite"], r["task_id"], r["episode"]): r.get("sim_steps")
        for r in reference["results"]
    }
    run_steps = {
        (r["suite"], r["task_id"], r["episode"]): r.get("sim_steps")
        for r in run["results"]
    }
    shared = sorted(set(steps) & set(run_steps))
    drifted = [(k, steps[k], run_steps[k]) for k in shared if steps[k] != run_steps[k]]
    print(f"\nstep counts       : {len(shared) - len(drifted)}/{len(shared)} identical")
    for (suite, task, ep), was, now in drifted[:12]:
        print(f"  {suite}:{task}#{ep}  {was} -> {now}")
    if len(drifted) > 12:
        print(f"  ... and {len(drifted) - 12} more")
    population_ok = (
        result.shared > 0
        and (not result.candidate_only)
        and (args.allow_partial or not result.baseline_only)
    )
    protocol_ok = all(
        (
            next(
                (
                    r["step_budget"]
                    for r in reference["results"]
                    if (r["suite"], r["task_id"], r["episode"]) == key
                )
            )
            == next(
                (
                    r["step_budget"]
                    for r in run["results"]
                    if (r["suite"], r["task_id"], r["episode"]) == key
                )
            )
            for key in shared
        )
    )
    if not population_ok:
        print(
            "Population mismatch: a reproduction must cover the reference population."
        )
    if not protocol_ok:
        print("Budget mismatch: standard and relaxed runs are different measurements.")
    outcomes_ok = (
        population_ok and protocol_ok and (not result.gains) and (not result.losses)
    )
    exact_ok = outcomes_ok and (not drifted)
    print()
    if args.require == "exact":
        passed = exact_ok
        print(f"GATE (exact)    : {('PASS' if passed else 'FAIL')}")
    else:
        passed = outcomes_ok
        print(f"GATE (outcomes) : {('PASS' if passed else 'FAIL')}")
        if outcomes_ok and drifted:
            print(
                "  outcomes match but step counts drifted: the harness changed something the policy observes. Investigate before trusting a later comparison."
            )
    if not outcomes_ok:
        print(
            f"  {len(result.gains)} episodes now succeed that did not, {len(result.losses)} no longer succeed. The port is not faithful yet; each difference must be explained, not averaged away."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
