"""control / motion: v30 control behavior with explicit dependencies."""

from __future__ import annotations
import numpy as np

WORKSPACE_Y = (-0.42, 0.42)
WORKSPACE_X = (-0.32, 0.18)


def in_workspace(xy):
    return (
        WORKSPACE_X[0] <= float(xy[0]) <= WORKSPACE_X[1]
        and WORKSPACE_Y[0] <= float(xy[1]) <= WORKSPACE_Y[1]
    )


PRECISE_TOL = 0.006


def move_to_precise(
    env, obs, target, gripper, events, phase, max_steps=45, tol=PRECISE_TOL
):
    """Like move_to but with a configurable tolerance and no exception on residual."""
    target = np.asarray(target, dtype=float)
    for step in range(max_steps):
        delta = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
        if float(np.linalg.norm(delta)) < tol:
            events.append(
                {
                    "phase": phase,
                    "steps": step,
                    "reached": True,
                    "eef": obs["robot0_eef_pos"].tolist(),
                }
            )
            return obs
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
    remaining = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
    events.append(
        {
            "phase": phase,
            "steps": max_steps,
            "reached": False,
            "remaining_xyz": remaining.tolist(),
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    return obs


def descend_hold_xy(
    env, obs, target, gripper, events, phase, max_steps=32, z_tol=0.005
):
    """Descend to target z while continuously servoing xy. Stops on z arrival or on contact stall
    (descent rate < 35% of the fastest rate seen, 3 steps running). Returns (obs, reached, contact_limited).
    """
    target = np.asarray(target, dtype=float)
    stalls = 0
    z0 = prev_z = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
    reached = False
    contact = False
    max_dz = 0.0
    for step in range(max_steps):
        cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
        delta = target - cur
        if abs(float(delta[2])) < z_tol and float(np.linalg.norm(delta[:2])) < 0.012:
            reached = True
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
        z = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
        commanded_down = delta[2] < -0.012
        dz = prev_z - z
        max_dz = max(max_dz, dz)
        if (
            commanded_down
            and step >= 2
            and (z0 - z > 0.01)
            and (dz < max(0.0015, 0.35 * max_dz))
        ):
            stalls += 1
        else:
            stalls = 0
        prev_z = z
        if stalls >= 3:
            contact = True
            break
    else:
        step = max_steps
    remaining = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
    events.append(
        {
            "phase": phase,
            "steps": step + 1,
            "reached": reached,
            "contact_limited": contact,
            "remaining_xyz": remaining.tolist(),
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    return (obs, reached, contact)


def descend_gentle_contact(
    env, obs, target, events, phase, action_z=-0.35, max_steps=14
):
    """Final approach of a HELD object onto a surface/floor: slow constant descent with xy servo until the object's
    contact stalls the hand (2 consecutive slow steps) or the target height is reached. v12: the full-speed
    contact descent kept pushing after contact and drove tall objects up into the palm slot, where the open gripper
    could no longer release them (object:7 carton), and pressed bowls onto plates so hard that the opening was
    blocked (spatial:1). Returns (obs, reached, contact)."""
    target = np.asarray(target, dtype=float)
    prev_z = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
    trace = []
    stalls = 0
    reached = False
    contact = False
    for step in range(max_steps):
        cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
        if cur[2] - target[2] < 0.003:
            reached = True
            break
        action = np.zeros(7, dtype=np.float32)
        action[:2] = np.clip((target[:2] - cur[:2]) / 0.05, -1.0, 1.0)
        action[2] = action_z
        action[6] = 1.0
        obs, _, _, _ = env.step(action)
        z = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
        dz = prev_z - z
        prev_z = z
        trace.append(round(dz, 4))
        if step >= 2 and dz < max(0.0012, 0.5 * max(trace)):
            stalls += 1
        else:
            stalls = 0
        if stalls >= 2:
            contact = True
            break
    events.append(
        {
            "phase": phase,
            "steps": len(trace),
            "reached": reached,
            "contact_limited": contact,
            "eef": obs["robot0_eef_pos"].tolist(),
            "dz_trace": trace[-8:],
        }
    )
    return (obs, reached, contact)


def move_to(env, obs, target, gripper, events, phase, max_steps=80):
    for step in range(max_steps):
        delta = target - np.asarray(obs["robot0_eef_pos"])
        if float(np.linalg.norm(delta)) < 0.02:
            events.append(
                {
                    "phase": phase,
                    "steps": step,
                    "reached": True,
                    "eef": obs["robot0_eef_pos"].tolist(),
                }
            )
            return obs
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
    events.append(
        {
            "phase": phase,
            "steps": max_steps,
            "reached": False,
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    raise RuntimeError(f"Motion phase {phase!r} did not reach target {target.tolist()}")


PROBE_ACTION_Z = -0.8
PROBE_TIP_OFFSET = -0.004


def hold(env, obs, gripper, steps, events, phase):
    action = np.zeros(7, dtype=np.float32)
    action[6] = gripper
    for _ in range(steps):
        obs, _, _, _ = env.step(action)
    events.append(
        {
            "phase": phase,
            "steps": steps,
            "eef": obs["robot0_eef_pos"].tolist(),
            "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
        }
    )
    return obs


def probe_top_height(
    env,
    obs,
    xy,
    start_z,
    support_z,
    events,
    phase,
    max_steps=45,
    close_first=True,
    open_after=True,
    action_z=PROBE_ACTION_Z,
):
    """Descend at xy with closed fingers until they stall on something. Returns (obs, contact_z or None).
    A robot-side measurement of the surface height under the grasp point; replaces the agentview depth
    estimate for grasp heights. Stall = descent rate drops below 40% of the free-descent rate seen so far.
    """
    if close_first:
        obs = hold(env, obs, 1.0, 6, events, f"{phase}:close")
    obs = move_to_precise(
        env,
        obs,
        np.r_[xy, start_z],
        1.0,
        events,
        f"{phase}:align",
        max_steps=25,
        tol=0.02,
    )
    contact_z = None
    stalls = 0
    z0 = prev_z = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
    trace = []
    for step in range(max_steps):
        cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
        action = np.zeros(7, dtype=np.float32)
        action[:2] = np.clip((np.asarray(xy) - cur[:2]) / 0.05, -1.0, 1.0)
        action[2] = action_z
        action[6] = 1.0
        obs, _, _, _ = env.step(action)
        z = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
        dz = prev_z - z
        trace.append(round(dz, 4))
        prev_z = z
        if z < support_z - 0.06:
            break
        if step < 3:
            continue
        free_rate = max(trace)
        if z0 - z > 0.012 and dz < max(0.0012, 0.6 * free_rate):
            stalls += 1
        else:
            stalls = 0
        if stalls >= 2:
            contact_z = z - PROBE_TIP_OFFSET
            break
    events.append(
        {
            "phase": phase,
            "steps": len(trace),
            "contact_z": contact_z,
            "eef": obs["robot0_eef_pos"].tolist(),
            "dz_trace": trace[-8:],
        }
    )
    if open_after:
        retreat_z = (contact_z if contact_z is not None else prev_z) + 0.06
        obs = move_to_precise(
            env,
            obs,
            np.r_[xy, retreat_z],
            -1.0,
            events,
            f"{phase}:retreat",
            max_steps=25,
            tol=0.015,
        )
        obs = hold(env, obs, -1.0, 4, events, f"{phase}:open")
    return (obs, contact_z)


def descend_for_grasp(env, obs, target, events, phase, max_steps=35):
    target = np.asarray(target, dtype=float)
    reached = False
    for step in range(max_steps):
        delta = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
        if float(np.linalg.norm(delta)) < 0.018:
            reached = True
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
        action[6] = -1.0
        obs, _, _, _ = env.step(action)
    else:
        step = max_steps
    remaining = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
    contact_limited = (
        float(np.linalg.norm(remaining[:2])) < 0.025
        and abs(float(remaining[2])) < 0.055
    )
    events.append(
        {
            "phase": phase,
            "steps": step,
            "reached": reached,
            "contact_limited": contact_limited and (not reached),
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    if not (reached or contact_limited):
        raise RuntimeError(f"Grasp descent did not reach {target.tolist()}")
    return obs


def descend_for_place(
    env,
    obs,
    target,
    events,
    phase,
    container_like,
    max_steps=40,
    z_tolerance_override=None,
):
    target = np.asarray(target, dtype=float)
    reached = False
    for step in range(max_steps):
        delta = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
        if float(np.linalg.norm(delta)) < 0.02:
            reached = True
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
        action[6] = 1.0
        obs, _, _, _ = env.step(action)
    else:
        step = max_steps
    remaining = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
    xy_tolerance = 0.055 if container_like else 0.025
    z_tolerance = (
        z_tolerance_override
        if z_tolerance_override is not None
        else 0.06 if container_like else 0.04
    )
    contact_limited = (
        float(np.linalg.norm(remaining[:2])) < xy_tolerance
        and abs(float(remaining[2])) < z_tolerance
    )
    events.append(
        {
            "phase": phase,
            "steps": step,
            "reached": reached,
            "contact_limited": contact_limited and (not reached),
            "remaining_xyz": remaining.tolist(),
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    if not (reached or contact_limited):
        raise RuntimeError(f"Place descent did not reach {target.tolist()}")
    return obs


def move_contact(env, obs, target, gripper, events, phase, max_steps=40):
    """Move toward a contact target while accepting surface-limited Z motion."""
    target = np.asarray(target, dtype=float)
    reached = False
    contacted = False
    for step in range(max_steps):
        delta = target - np.asarray(obs["robot0_eef_pos"], dtype=float)
        reached = float(np.linalg.norm(delta)) < 0.02
        contacted = (
            float(np.linalg.norm(delta[:2])) < 0.025 and abs(float(delta[2])) < 0.055
        )
        if reached or contacted:
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / 0.05, -1.0, 1.0)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
    else:
        step = max_steps
    events.append(
        {
            "phase": phase,
            "steps": step,
            "reached": reached,
            "contact_limited": contacted and (not reached),
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    if not (reached or contacted):
        raise RuntimeError(
            f"Contact phase {phase!r} did not reach target {target.tolist()}"
        )
    return obs


def success(env):
    for candidate in (env, getattr(env, "env", None)):
        check = getattr(candidate, "_check_success", None) or getattr(
            candidate, "check_success", None
        )
        if check is not None:
            try:
                return bool(check())
            except Exception:
                pass
    return False


def rotate_wrist(env, obs, direction, steps, events, phase):
    action = np.zeros(7, dtype=np.float32)
    action[5] = float(np.sign(direction))
    action[6] = 1.0
    for _ in range(steps):
        obs, _, _, _ = env.step(action)
    events.append(
        {
            "phase": phase,
            "steps": steps,
            "eef": obs["robot0_eef_pos"].tolist(),
            "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
        }
    )
    return obs
