"""Run configuration, validation, and provenance."""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import subprocess
import sys
import yaml
from . import protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
NON_BEHAVIOURAL = frozenset({"name", "out_dir", "record", "notes"})


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines what a run measures."""

    name: str
    cases: str = ",".join(protocol.LIBERO_40)
    episodes: int = 1
    first_episode: int = 0
    orient_fix: int = protocol.ORIENT_FIX
    image_size: int = protocol.IMAGE_SIZE
    camera_fovy: float = protocol.CAMERA_FOVY
    support_z_offset: float = protocol.SUPPORT_Z_OFFSET
    max_tool_rounds: int = protocol.MAX_TOOL_ROUNDS
    live_observation: bool = False
    live_image_history: int = 4
    budget_scale: float = 1.0
    seed: int = protocol.SEED
    torch_threads: int = protocol.TORCH_THREADS
    policy: str = "v30"
    model_path: str = str(REPO_ROOT / "weights" / "DepthVLM-4B")
    source_path: str = str(REPO_ROOT / "vendor" / "DepthVLM")
    gdino_path: str = str(REPO_ROOT / "weights" / "grounding-dino-tiny")
    eupe_repo: str = str(REPO_ROOT / "vendor" / "EUPE")
    eupe_checkpoint: str = str(REPO_ROOT / "weights" / "EUPE-ViT-B.pt")
    workers: int = 1
    out_dir: str = str(REPO_ROOT / "results")
    notes: str = ""

    def resolved_cases(self) -> list[tuple[str, int]]:
        return protocol.parse_cases(self.cases)

    def planned_episodes(self) -> int:
        return len(self.resolved_cases()) * self.episodes

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def validate(self) -> None:
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Run name must be alphanumeric with _ or -: {self.name!r}"
            )
        if self.episodes < 1 or self.first_episode < 0:
            raise ValueError("episodes must be >= 1 and first_episode >= 0")
        if self.orient_fix not in (0, 1):
            raise ValueError("orient_fix must be 0 or 1")
        if self.workers != 1:
            raise ValueError("v30 reproduction uses exactly one worker")
        if self.live_image_history < 1:
            raise ValueError("live_image_history must be >= 1")
        if not math.isfinite(self.budget_scale) or self.budget_scale < 1.0:
            raise ValueError(
                "budget_scale must be >= 1.0: shortening the official horizon would make the run incomparable with every reported number and with the VLA baselines, in the direction that flatters nothing."
            )
        self.resolved_cases()
        if self.image_size < 32 or self.max_tool_rounds < 1 or self.torch_threads < 1:
            raise ValueError(
                "image_size >= 32, max_tool_rounds >= 1, torch_threads >= 1 required"
            )
        if not math.isfinite(self.camera_fovy) or not 0 < self.camera_fovy < 180:
            raise ValueError("camera_fovy must be finite and between 0 and 180 degrees")
        if not math.isfinite(self.support_z_offset) or self.support_z_offset <= 0:
            raise ValueError("support_z_offset must be finite and positive")
        if self.policy != "v30":
            raise ValueError("Only the v30 policy is implemented")

    def budget_for(self, suite: str) -> int:
        """The horizon this run enforces. `protocol.budget_for` stays official."""
        return round(protocol.budget_for(suite) * self.budget_scale)

    def grasp_attempts(self) -> int:
        """Attempt cap, scaled with the horizon so one knob moves both.

        Raising the horizon alone would just trade `budget_refusal` for
        `attempt_limit`, which already fires in 5 of 24 baseline failures.
        """
        return max(1, round(protocol.MAX_GRASP_ATTEMPTS * self.budget_scale))

    def validate_assets(self) -> None:
        self.resolved_cases()
        for label in (
            "model_path",
            "source_path",
            "gdino_path",
            "eupe_repo",
            "eupe_checkpoint",
        ):
            path = Path(getattr(self, label))
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")


def load(path: str | Path, **overrides) -> RunConfig:
    """Load a YAML config, apply overrides, and validate."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(RunConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
    merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
    if not merged.get("cases"):
        merged["cases"] = "libero_40"
    for key in (
        "model_path",
        "source_path",
        "gdino_path",
        "eupe_repo",
        "eupe_checkpoint",
        "out_dir",
    ):
        if key in merged and (not Path(merged[key]).is_absolute()):
            merged[key] = str((REPO_ROOT / merged[key]).resolve())
    config = RunConfig(**merged)
    config.validate()
    return config


def dump(config: RunConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(asdict(config), sort_keys=True, allow_unicode=True)
    )


def git_provenance() -> dict[str, object]:
    """Return the current commit, branch, and working-tree state."""

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        dirty = git("status", "--porcelain")
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty),
            "dirty_paths": dirty.splitlines()[:20],
        }
    except Exception as error:
        return {"commit": None, "error": f"{type(error).__name__}: {error}"}


def environment_provenance() -> dict[str, object]:
    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def diff(config: RunConfig, baseline: RunConfig) -> dict[str, tuple[object, object]]:
    """Behavioural fields where ``config`` differs from ``baseline``."""
    return {
        f.name: (getattr(baseline, f.name), getattr(config, f.name))
        for f in fields(RunConfig)
        if f.name not in NON_BEHAVIOURAL
        and getattr(config, f.name) != getattr(baseline, f.name)
    }


def require_declared_diff(
    config: RunConfig, baseline: RunConfig, declared: set[str] | None = None
) -> dict[str, tuple[object, object]]:
    """Refuse to run on differences from the baseline that were not declared.

    An experiment states which fields it changes; anything else that has drifted
    -- a weight path, the image convention, the worker count -- stops the run
    before the GPU is touched, instead of producing a number that cannot be
    attributed.
    """
    changes = diff(config, baseline)
    undeclared = {k: v for k, v in changes.items() if k not in (declared or set())}
    if undeclared:
        lines = "\n".join(
            (f"  {k}: baseline={v[0]!r} -> run={v[1]!r}" for k, v in undeclared.items())
        )
        raise ValueError(
            f"Undeclared differences from the baseline:\n{lines}\nDeclare them with --change <field> (one variable per experiment) or correct the config."
        )
    return changes
