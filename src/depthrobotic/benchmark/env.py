"""LIBERO environment initialization and action-budget enforcement."""

from __future__ import annotations
from depthrobotic.control.motion import success
from depthrobotic.signals import StepBudgetExceeded, TaskCompleted


class BudgetedEnv:
    """Count every simulator step and check success after every action."""

    def __init__(self, env, budget):
        self.env = env
        self.budget = budget
        self.steps = 0
        self.observers = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        if self.steps >= self.budget:
            raise StepBudgetExceeded(f"Official {self.budget}-step horizon reached")
        result = self.env.step(action)
        self.steps += 1
        for observer in self.observers:
            try:
                observer(result[0], self.steps)
            except Exception:
                continue
        if success(self.env):
            raise TaskCompleted(result[0])
        if self.steps >= self.budget:
            raise StepBudgetExceeded(f"Official {self.budget}-step horizon reached")
        return result


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from .. import protocol
from ..control.motion import hold


@dataclass
class Episode:
    """A constructed, settled episode ready for the policy."""

    suite: str
    task_id: int
    episode: int
    instruction: str
    env: Any
    base_env: Any
    obs: dict
    support_z: float
    events: list = field(default_factory=list)
    perceptions: list = field(default_factory=list)

    @property
    def sim_steps(self) -> int:
        return int(self.env.steps)

    @property
    def step_budget(self) -> int:
        """What this run enforces. Equals the official horizon at budget_scale 1."""
        return int(self.env.budget)

    @property
    def official_horizon(self) -> int:
        """The benchmark's horizon, recorded whatever this run enforces.

        Keeping both is what lets a budget-relaxed run still report the
        official-protocol number, and what stops the two being merged.
        """
        return protocol.budget_for(self.suite)

    def succeeded(self) -> bool:
        return bool(success(self.env))

    def close(self) -> None:
        self.base_env.close()


def open_episode(suite: str, task_id: int, episode: int, config) -> Episode:
    """Build, seed and settle one episode. Raises before the policy runs."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite]()
    if not 0 <= task_id < bench.n_tasks:
        raise ValueError(f"{suite}:{task_id} is out of range (n_tasks={bench.n_tasks})")
    init_states = bench.get_task_init_states(task_id)
    if not 0 <= episode < len(init_states):
        raise ValueError(f"{suite}:{task_id} has no initial state {episode}")
    task = bench.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    base_env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=config.image_size,
        camera_widths=config.image_size,
    )
    env = BudgetedEnv(base_env, config.budget_for(suite))
    events: list = []
    try:
        env.seed(episode)
        env.reset()
        obs = env.set_init_state(init_states[episode])
        camera_id = env.sim.model.camera_name2id(protocol.CAMERA_NAME)
        env.sim.model.cam_fovy[camera_id] = config.camera_fovy
        obs = hold(
            env,
            obs,
            protocol.SETTLE_GRIPPER,
            protocol.SETTLE_STEPS,
            events,
            protocol.SETTLE_PHASE,
        )
        env.steps = 0
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        env.home_xy = eef[:2].copy()
        support_z = float(eef[2] - config.support_z_offset)
    except BaseException:
        base_env.close()
        raise
    return Episode(
        suite=suite,
        task_id=task_id,
        episode=episode,
        instruction=task.language,
        env=env,
        base_env=base_env,
        obs=obs,
        support_z=support_z,
        events=events,
    )
