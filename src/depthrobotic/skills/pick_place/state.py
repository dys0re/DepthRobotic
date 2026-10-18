"""skills / pick_place / state: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import numpy as np
import re
import torch
import torch.nn.functional as F


@dataclass
class ObjectState:
    instance_id: str
    description: str
    camera: str = "agentview"
    pixel_frame: str = "upright_0_1000"
    point: list | None = None
    box: list | None = None
    surface_z: float | None = None
    world: list | None = None
    observed_step: int = -1
    identity: str = "UNKNOWN"
    association: str = "UNKNOWN"
    geometry_valid: bool = False
    dirty_reason: str | None = None

    def observe(
        self,
        step,
        *,
        point=None,
        box=None,
        world=None,
        surface_z=None,
        association="MATCHED",
        identity=None,
    ):
        if point is not None:
            self.point = list(point)
        if box is not None:
            self.box = list(box)
        if world is not None:
            self.world = [float(x) for x in world]
        if surface_z is not None:
            self.surface_z = float(surface_z)
        self.observed_step = int(step)
        self.association = association
        if identity is not None:
            self.identity = identity
        if world is not None and surface_z is not None:
            self.geometry_valid = True
            self.dirty_reason = None

    def invalidate(self, reason):
        self.geometry_valid = False
        self.association = "UNKNOWN"
        self.dirty_reason = reason


@dataclass
class TaskState:
    source: ObjectState
    target: ObjectState
    held: str = "UNKNOWN"
    held_offset: list | None = None
    relation: str = "UNKNOWN"
    transitions: list = field(default_factory=list)

    def record(self, stage, step, **evidence):
        self.transitions.append({"stage": stage, "step": int(step), **evidence})

    def contact(self, reason, step):
        self.target.invalidate(reason)
        self.relation = "UNKNOWN"
        self.record("invalidate", step, reason=reason)

    def snapshot(self):
        return asdict(self)

    def set_relation(self, geometric_inside):
        if geometric_inside is False:
            self.relation = "OUTSIDE"
        elif (
            geometric_inside is True
            and self.source.identity in {"LABEL_MATCH", "NAME_MATCH"}
            and (self.source.association == "MATCHED")
            and self.target.geometry_valid
            and (self.held == "RELEASED")
        ):
            self.relation = "OBSERVED_SATISFIED"
        else:
            self.relation = "UNKNOWN"
        return self.relation


def region_descriptor(features, box):
    side = int(round(len(features) ** 0.5))
    xy = torch.stack(
        torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1
    )
    xy = (xy.reshape(-1, 2)[:, [1, 0]].float() + 0.5) * 1000 / side
    b = torch.tensor(box, dtype=torch.float32)
    keep = ((xy >= b[:2]) & (xy <= b[2:])).all(-1)
    if not keep.any():
        return None
    return F.normalize(features[keep].mean(0), dim=0)


def valid_region(box, point):
    try:
        b, p = (np.asarray(box, float), np.asarray(point, float))
        return bool(
            b.shape == (4,)
            and p.shape == (2,)
            and np.isfinite(b).all()
            and np.isfinite(p).all()
            and (b[:2] < b[2:]).all()
            and (b >= 0).all()
            and (b <= 1000).all()
            and (p >= b[:2]).all()
            and (p <= b[2:]).all()
        )
    except (TypeError, ValueError):
        return False


def semantic_evidence(description, name, label=""):
    generic = {
        "the",
        "a",
        "an",
        "object",
        "item",
        "container",
        "bottle",
        "can",
        "box",
        "package",
        "carton",
        "food",
        "product",
        "black",
        "white",
        "red",
        "green",
        "blue",
        "small",
        "large",
        "left",
        "right",
    }

    def tokens(x):
        return set(re.findall("[a-z]+", str(x).lower())) - generic

    noun = re.split(
        "\\b(?:between|next to|on top of|on|in|behind|beside|at the|to the)\\b",
        str(description).lower(),
        maxsplit=1,
    )[0]
    wanted, printed, named = (tokens(noun), tokens(label), tokens(name))
    empty = {"none", "nothing", "unknown", "empty", "visible", "no", "text", "label"}
    printed -= empty
    named -= empty
    if wanted and wanted <= printed:
        return "LABEL_MATCH"
    if printed and printed < wanted:
        return "UNKNOWN"
    if printed and wanted and (not wanted <= printed):
        return "LABEL_CONFLICT"
    if wanted and wanted <= named:
        return "NAME_MATCH"
    return "UNKNOWN"
