"""Episode-local dependencies shared by geometric skills."""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Protocol, Any


class VisionLanguageModel(Protocol):
    def generate_response(
        self, image, prompt: str, max_new_tokens: int = 160
    ) -> tuple[str, int]: ...
    def generate_with_tools(
        self, messages: list, tools: list, max_new_tokens: int = 300
    ) -> str: ...

    def ask(
        self, image, prompt: str, max_new_tokens: int = 160
    ) -> tuple[str, dict, int]: ...

    def predict_depth(self, image, focal_px: float): ...


class FeatureExtractor(Protocol):
    resolution: int
    calls: int

    def features(self, image): ...


class ObjectDetector(Protocol):

    def detect(self, image, phrases: list[str], prefix: str = "D"): ...


@dataclass(frozen=True)
class Models:
    vlm: VisionLanguageModel
    detector: ObjectDetector
    features: FeatureExtractor


@dataclass(frozen=True)
class AgentSettings:
    orient_fix: bool = False
    max_grasp_attempts: int = 3
    max_tool_rounds: int = 12
    live_observation: bool = False
    live_image_history: int = 4
    release_mode: str = "act"
    basket_clearance: float = 0.12

    @classmethod
    def from_config(cls, config):
        return cls(
            orient_fix=bool(config.orient_fix),
            max_grasp_attempts=config.grasp_attempts(),
            max_tool_rounds=config.max_tool_rounds,
            live_observation=config.live_observation,
            live_image_history=config.live_image_history,
        )


@dataclass
class SkillContext:
    models: Models
    settings: AgentSettings
    case_label: str = "episode"
    frames_dir: Path | None = None
    source_candidates: list = field(default_factory=list)
    append_observation: Any = field(init=False, repr=False)

    def __post_init__(self):
        from .observation import make_appender
        from .perception.images import observation_image

        self.append_observation = make_appender(
            partial(observation_image, context=self), self.settings.live_image_history
        )
