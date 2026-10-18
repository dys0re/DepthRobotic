"""Run a declared v30 configuration against a fixed LIBERO population."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import config as config_module


def main(argv=None) -> int:
    default = config_module.REPO_ROOT / "configs" / "baseline_v30.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        required=True,
        help="Unique run name; existing runs are never overwritten",
    )
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--baseline", type=Path, default=default)
    parser.add_argument(
        "--cases", help="libero_40, libero_90, or comma-separated suite:task pairs"
    )
    parser.add_argument("--change", action="append", default=[], metavar="FIELD")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Exploratory run; not reportable"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and assets without loading models",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    config = config_module.load(
        args.config,
        name=args.name,
        cases=args.cases,
        out_dir=str(args.out_dir) if args.out_dir else None,
    )
    baseline = config_module.load(args.baseline, name=args.name, cases=args.cases)
    changes = config_module.require_declared_diff(config, baseline, set(args.change))
    config.validate_assets()
    if config.run_dir.exists():
        raise SystemExit(f"Refusing to overwrite a prior run: {config.run_dir}")
    provenance = config_module.git_provenance()
    if not args.check and provenance.get("dirty") and (not args.allow_dirty):
        raise SystemExit(
            "Working tree is dirty. Commit before a measured run, or use --allow-dirty for an exploratory run."
        )
    print(
        json.dumps(
            {
                "run": config.name,
                "episodes": config.planned_episodes(),
                "policy": config.policy,
                "orient_fix": config.orient_fix,
                "declared_changes": changes,
                "commit": provenance.get("commit"),
                "preflight": "passed",
            },
            indent=2,
        ),
        flush=True,
    )
    if args.check:
        return 0
    from .runtime import initialize

    initialize(config)
    from .agent import Agent, load_models
    from .context import AgentSettings
    from .benchmark.episode import run_one
    from .benchmark.recording import EpisodeRecorder
    from .results import ResultWriter
    from . import validate

    settings = AgentSettings.from_config(config)
    policy = Agent(load_models(config), settings)
    writer = ResultWriter(config, policy_provenance=policy.provenance())
    config_module.dump(config, config.run_dir / "config.yaml")
    recorder = (
        None if args.no_record else EpisodeRecorder(config.run_dir / "videos", settings)
    )
    writer.flush()
    for suite, task in config.resolved_cases():
        for episode in range(
            config.first_episode, config.first_episode + config.episodes
        ):
            print(f"[{config.name}] {suite}:{task} ep{episode}", flush=True)
            result = run_one(policy, suite, task, episode, config, recorder=recorder)
            try:
                validate.check_episode(result)
            except validate.ValidationError:
                from .results import json_default

                (config.run_dir / "invalid_episode.json").write_text(
                    json.dumps(result, indent=2, default=json_default) + "\n"
                )
                raise
            payload = writer.append(result)
            print(
                f"RESULT {suite}:{task} success={result['success']} steps={result['sim_steps']} seconds={result['wall_seconds']:.1f} error={result.get('error')} running={payload['successes']}/{payload['episodes']}",
                flush=True,
            )
    payload = writer.flush()
    validate.check_payload(payload)
    print(
        json.dumps(
            {k: payload[k] for k in ("successes", "episodes", "complete", "by_suite")},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
