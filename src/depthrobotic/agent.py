"""Plan-and-execute orchestration; all model and skill dependencies are injected."""

from __future__ import annotations
from pathlib import Path
from .context import AgentSettings, Models, SkillContext


def load_models(config) -> Models:
    from .models.depthvlm import DepthVLM
    from .models.eupe import EUPEFeatures
    from .models.grounding_dino import GroundingDino
    from .perception.candidates import RecordingProposer

    features = EUPEFeatures(Path(config.eupe_repo), Path(config.eupe_checkpoint))
    detector = GroundingDino(Path(config.gdino_path))
    detector._ensure()
    if detector.metadata()["error"]:
        raise RuntimeError(f"GroundingDINO unavailable: {detector.metadata()['error']}")
    vlm = DepthVLM(Path(config.model_path), Path(config.source_path))
    return Models(vlm=vlm, detector=RecordingProposer(detector), features=features)


class Agent:

    def __init__(
        self, models: Models, settings: AgentSettings, *, planner=None, skills=None
    ):
        from .planning.planner import plan
        from .skills.registry import SKILLS

        self.models = models
        self.settings = settings
        self.planner = planner if planner is not None else plan
        self.skills = dict(SKILLS if skills is None else skills)

    def begin_episode(self, episode):
        self.context = SkillContext(
            self.models,
            self.settings,
            case_label=f"{episode.suite}_{episode.task_id}_ep{episode.episode}",
        )
        self._feature_calls_at_start = self.models.features.calls

    def plan(self, image, instruction):
        return self.planner(self.models.vlm, image, instruction)

    def execute(self, episode, step, config):
        action = step.get("action")
        if action not in self.skills:
            raise ValueError(f"Unsupported plan action: {action!r}")
        return self.skills[action](
            episode.env,
            episode.obs,
            self.models.vlm,
            episode.instruction,
            step,
            config.image_size,
            episode.support_z,
            episode.events,
            episode.perceptions,
            context=self.context,
        )

    def episode_diagnostics(self, episode):
        return {"eupe_calls": self.models.features.calls - self._feature_calls_at_start}

    def provenance(self):
        return {
            "mode": "v30",
            "architecture": "explicit-context-skills",
            "max_grasp_attempts": self.settings.max_grasp_attempts,
            "learned_models": ["DepthVLM-4B", "GroundingDINO", "EUPE-ViT-B"],
            "gdino": self.models.detector.metadata(),
            "eupe": self.models.features.metadata(),
        }
