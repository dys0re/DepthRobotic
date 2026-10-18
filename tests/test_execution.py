"""Check terminal signals, multi-step orchestration and cleanup without a simulator."""

from types import SimpleNamespace
import numpy as np
import pytest

from depthrobotic.agent import Agent
from depthrobotic.context import Models, AgentSettings
from depthrobotic.config import RunConfig
from depthrobotic.benchmark.env import BudgetedEnv, Episode
from depthrobotic.benchmark import episode as runner
from depthrobotic.signals import TaskCompleted, StepBudgetExceeded


class Simulator:
    def __init__(self, success_after=None):
        self.n = 0
        self.success_after = success_after
        self.closed = False
        self.obs = {
            "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }

    def step(self, action):
        self.n += 1
        return self.obs, 0, False, {}

    def _check_success(self):
        return self.success_after is not None and self.n >= self.success_after

    def close(self):
        self.closed = True


def test_success_on_last_budgeted_action_wins():
    base = Simulator(success_after=3)
    env = BudgetedEnv(base, 3)
    env.step(np.zeros(7))
    env.step(np.zeros(7))
    with pytest.raises(TaskCompleted):
        env.step(np.zeros(7))
    assert env.steps == 3
    with pytest.raises(StepBudgetExceeded):
        env.step(np.zeros(7))
    assert base.n == 3


def test_budget_stops_without_extra_actions():
    base = Simulator()
    env = BudgetedEnv(base, 1)
    with pytest.raises(StepBudgetExceeded):
        env.step(np.zeros(7))
    with pytest.raises(StepBudgetExceeded):
        env.step(np.zeros(7))
    assert base.n == 1


def test_observer_failure_does_not_change_success():
    env = BudgetedEnv(Simulator(success_after=1), 1)

    def bad_observer(*args):
        raise RuntimeError("video writer failed")

    env.observers.append(bad_observer)
    with pytest.raises(TaskCompleted):
        env.step(np.zeros(7))


def episode_for(base, budget):
    env = BudgetedEnv(base, budget)
    return Episode(
        "libero_spatial", 0, 0, "put the bowl on the plate", env, base, base.obs, 0.76
    )


def models():
    return Models(None, None, SimpleNamespace(calls=0))


@pytest.mark.parametrize(
    "success_after,budget,expected,steps", [(1, 280, True, 1), (None, 280, False, 1)]
)
def test_agent_outcome_and_close(monkeypatch, success_after, budget, expected, steps):
    base = Simulator(success_after)
    ep = episode_for(base, budget)
    monkeypatch.setattr(runner, "open_episode", lambda *args: ep)

    def planner(*args):
        return "test", [{"action": "pick_place"}]

    def skill(env, obs, *args, **kwargs):
        return env.step(np.zeros(7))[0]

    agent = Agent(
        models(), AgentSettings(), planner=planner, skills={"pick_place": skill}
    )
    result = runner.run_one(agent, "libero_spatial", 0, 0, RunConfig("unit"))
    assert result["success"] is expected
    assert result["sim_steps"] == steps
    assert result["error"] is None
    assert base.closed


def test_multi_step_plan_passes_current_observation(monkeypatch):
    base = Simulator()
    ep = episode_for(base, 280)
    monkeypatch.setattr(runner, "open_episode", lambda *args: ep)
    calls = []

    def planner(*args):
        return "test", [{"action": "articulate"}, {"action": "pick_place"}]

    def skill(env, obs, grounder, instruction, step, *args, **kwargs):
        calls.append((step["action"], obs.get("version", 0)))
        env.step(np.zeros(7))
        return dict(obs, version=obs.get("version", 0) + 1)

    agent = Agent(
        models(),
        AgentSettings(),
        planner=planner,
        skills={"articulate": skill, "pick_place": skill},
    )
    result = runner.run_one(agent, "libero_spatial", 0, 0, RunConfig("unit"))
    assert calls == [("articulate", 0), ("pick_place", 1)]
    assert result["sim_steps"] == 2
    assert base.closed


def test_construction_failure_is_not_counted(monkeypatch):
    def broken(*args):
        raise FileNotFoundError("initial state missing")

    monkeypatch.setattr(runner, "open_episode", broken)
    with pytest.raises(FileNotFoundError):
        runner.run_one(None, "libero_spatial", 0, 0, RunConfig("unit"))


@pytest.mark.parametrize(
    "modules",
    [
        [
            "depthrobotic.perception.candidates",
            "depthrobotic.skills.registry",
            "depthrobotic.agent",
        ],
        [
            "depthrobotic.models.depthvlm",
            "depthrobotic.models.eupe",
            "depthrobotic.skills.registry",
        ],
    ],
)
def test_independent_import_order(modules):
    import subprocess
    import sys

    script = "import importlib; " + "; ".join(
        f"importlib.import_module({name!r})" for name in modules
    )
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
