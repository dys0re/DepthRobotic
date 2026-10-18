"""Episode execution, terminal outcomes, diagnostics, and cleanup."""

from __future__ import annotations
import time
from .. import diagnose
from .env import Episode, open_episode
from ..signals import TaskCompleted
from ..perception.images import raw_image


def _build_result(
    ep: Episode,
    config,
    *,
    success: bool,
    plan_response: str,
    plan: list[dict],
    error: str | None = None,
) -> dict:
    """The single result schema. Every exit path goes through here."""
    return {
        "suite": ep.suite,
        "task_id": ep.task_id,
        "episode": ep.episode,
        "instruction": ep.instruction,
        "success": bool(success),
        "error": error,
        "plan": plan,
        "plan_response": plan_response,
        "perceptions": ep.perceptions,
        "events": ep.events,
        "sim_steps": ep.sim_steps,
        "step_budget": ep.step_budget,
        "official_horizon": ep.official_horizon,
        "within_official_horizon": bool(success)
        and ep.sim_steps <= ep.official_horizon,
        "camera_fovy": config.camera_fovy,
        "orient_fix": config.orient_fix,
        "support_z": ep.support_z,
    }


def run_one(
    policy, suite: str, task_id: int, episode: int, config, recorder=None
) -> dict:
    """Run one episode end to end and return its result.

    Construction failures propagate: an episode that cannot be built is a broken
    run, not a failed episode, and must not enter a denominator.
    """
    ep = open_episode(suite, task_id, episode, config)
    if recorder is not None:
        recorder.start(suite, task_id, episode)
        ep.env.observers.append(recorder.observe)
    started = time.perf_counter()
    plan_response, plan = ("", [])
    try:
        policy.begin_episode(ep)
        plan_response, plan = policy.plan(
            raw_image(ep.obs, context=policy.context), ep.instruction
        )
        for plan_step in plan:
            ep.obs = policy.execute(ep, plan_step, config)
            if ep.succeeded():
                break
        result = _build_result(
            ep, config, success=ep.succeeded(), plan_response=plan_response, plan=plan
        )
    except TaskCompleted:
        result = _build_result(
            ep, config, success=True, plan_response=plan_response, plan=plan
        )
    except Exception as error:
        result = _build_result(
            ep,
            config,
            success=False,
            plan_response=plan_response,
            plan=plan,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        result_seconds = time.perf_counter() - started
        try:
            diagnostics = policy.episode_diagnostics(ep)
        except Exception as error:
            diagnostics = {"diagnostics_error": f"{type(error).__name__}: {error}"}
        ep.close()
    result["wall_seconds"] = result_seconds
    result["diagnosis"] = diagnose.classify(result)
    for key, value in diagnostics.items():
        if key in result:
            raise ValueError(
                f"Policy diagnostics may not overwrite result field {key!r}"
            )
        result[key] = value
    if recorder is not None:
        result["video"] = recorder.finish(result)
    return result
