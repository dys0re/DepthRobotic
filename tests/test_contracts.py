"""Behavior contracts measured from the reference, independent of its source tree."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from depthrobotic.context import AgentSettings, Models, SkillContext
from depthrobotic.skills.pick_place.executor import PickPlaceExecutor
from depthrobotic.skills.pick_place.react import PICK_PLACE_TOOLS
from depthrobotic.planning.planner import grammar_plan
from depthrobotic.geometry.camera import raw_ray
from depthrobotic.perception.parsing import normalize_point, normalize_box
from depthrobotic.skills.articulate.handle import _parse_boxes

FIXTURE = json.loads(
    (Path(__file__).parent / "reference/v30_contracts.json").read_text()
)


class EmptyGrounder:
    def generate_response(self, image, prompt, max_tokens=160):
        return "{}", 0


class Features:
    calls = 0
    resolution = 512

    def features(self, image):
        raise AssertionError("Invalid regions must not invoke the feature model")


def context():
    return SkillContext(Models(EmptyGrounder(), None, Features()), AgentSettings())


def test_tool_schemas_preserve_reference_prompts():
    assert PICK_PLACE_TOOLS == FIXTURE["tools"]


@pytest.mark.parametrize("instruction,expected", FIXTURE["grammar"].items())
def test_compound_grammar(instruction, expected):
    assert grammar_plan(instruction) == expected


@pytest.mark.parametrize("case", FIXTURE["rays"])
def test_pixel_convention(case):
    np.testing.assert_array_equal(
        raw_ray(case["point"], case["focal"], case["size"], context=context()),
        case["ray"],
    )


@pytest.mark.parametrize("case", FIXTURE["parsing"])
def test_coordinate_normalization(case):
    assert normalize_point(case["value"]) == case["point"]
    if isinstance(case["value"], list) and len(case["value"]) == 4:
        assert normalize_box(case["value"]) == case["box"]


@pytest.mark.parametrize("case", FIXTURE["handle_boxes"])
def test_handle_box_recovery(case):
    assert _parse_boxes(case["response"]) == case["boxes"]


def test_controller_initialization_and_dispatch_match_reference():
    ctx = context()
    obs = {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.1]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
    }
    executor = PickPlaceExecutor(
        SimpleNamespace(steps=0, budget=280),
        obs,
        ctx.models.vlm,
        "put the bowl on the plate",
        "bowl",
        "on plate",
        832,
        0.86,
        [],
        [],
        context=ctx,
    )
    assert executor.record == FIXTURE["controller"]["initial_record"]
    for case in FIXTURE["controller"]["dispatch"]:
        text, image = executor.dispatch(case["name"], case["args"])
        assert text == case["text"]
        assert image is None
        assert executor.record == case["record"]


def test_candidate_cache_is_episode_local():
    a, b = context(), context()
    a.source_candidates.append([200, 300])
    assert b.source_candidates == []


def test_relaxed_reference_is_complete_and_separate():
    from depthrobotic import validate, protocol

    data = json.loads(
        (Path(__file__).parent / "reference/v30_2x_libero40.json").read_text()
    )
    validate.check_payload(data)
    assert data["complete"] and data["successes"] == 20
    assert data["budget_scale"] == 2.0
    assert data["configuration"]["live_observation"] is True
    assert {(r["suite"], r["task_id"]) for r in data["results"]} == set(
        protocol.parse_cases("libero_40")
    )
    for result in data["results"]:
        validate.check_episode(result)
        assert result["step_budget"] == 2 * protocol.budget_for(result["suite"])
