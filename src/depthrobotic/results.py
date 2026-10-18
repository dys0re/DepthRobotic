"""Validated, atomic run artifacts with immutable launch provenance."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from . import config as config_module, protocol, validate


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def source_manifest():
    package = Path(__file__).resolve().parent
    return {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }


def artifact_manifest(config):
    """Fingerprint the actual third-party implementations and frozen model weights."""
    files = []
    for directory in (config.source_path, config.eupe_repo):
        files.extend(Path(directory).rglob("*.py"))
    for directory in (config.model_path, config.gdino_path):
        base = Path(directory)
        files.extend(base.glob("*.safetensors"))
        files.extend(base.glob("*.json"))
    files.append(Path(config.eupe_checkpoint))
    manifest = {}
    for path in sorted(set(files)):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest[str(path)] = {
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    return manifest


def build_payload(results, config, *, policy_provenance=None, provenance=None):
    planned = config.planned_episodes()
    suites = list(dict.fromkeys((suite for suite, _ in config.resolved_cases())))
    by_suite = {}
    for suite in suites:
        selected = [r for r in results if r["suite"] == suite]
        wins = sum((bool(r["success"]) for r in selected))
        by_suite[suite] = {
            "successes": wins,
            "episodes": len(selected),
            "success_rate": wins / len(selected) if selected else 0.0,
        }
    successes = sum((bool(r["success"]) for r in results))
    return {
        "protocol": "depthrobotic-v30-libero-r1",
        "policy": config.policy,
        "successes": successes,
        "episodes": len(results),
        "planned_episodes": planned,
        "complete": len(results) == planned,
        "budget_scale": config.budget_scale,
        "official_protocol": config.budget_scale == 1.0,
        "successes_within_official_horizon": sum(
            (bool(r.get("within_official_horizon")) for r in results)
        ),
        "success_rate": successes / len(results) if results else 0.0,
        "by_suite": by_suite,
        "official_suite_horizons": protocol.TASK_BUDGETS,
        "success_predicate_usage": "termination and scoring only",
        "configuration": asdict(config),
        "provenance": (
            provenance
            if provenance is not None
            else {
                "git": config_module.git_provenance(),
                "environment": config_module.environment_provenance(),
                "source_sha256": source_manifest(),
                "policy": policy_provenance or {},
            }
        ),
        "results": results,
    }


class ResultWriter:

    def __init__(self, config, policy_provenance=None):
        self.config = config
        self.path = config.run_dir / "results.json"
        config.run_dir.mkdir(parents=True, exist_ok=False)
        self.results = []
        self.provenance = build_payload(
            [], config, policy_provenance=policy_provenance
        )["provenance"]
        self.provenance["artifacts"] = artifact_manifest(config)

    def append(self, result):
        validate.check_episode(result)
        prospective = self.results + [result]
        payload = build_payload(prospective, self.config, provenance=self.provenance)
        validate.check_payload(payload)
        self.results = prospective
        return self._write(payload)

    def flush(self):
        payload = build_payload(self.results, self.config, provenance=self.provenance)
        validate.check_payload(payload)
        return self._write(payload)

    def _write(self, payload):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")
        tmp.replace(self.path)
        return payload


def load(path):
    return json.loads(Path(path).read_text())
