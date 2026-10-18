"""skills / articulate / door: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import hold
from depthrobotic.geometry.camera import deproject
from depthrobotic.geometry.depth import frame_geometry, sample_point_depth
from depthrobotic.perception.images import raw_image
from depthrobotic.perception.parsing import normalize_box, safe_ask_structured
from depthrobotic.prompts import _door_prompt
from depthrobotic.skills.articulate.handle import (
    _ray,
    _ray_plane,
    _unflip_box,
    _upright,
    pose_servo,
)
from typing import Any
import numpy as np


def robot_facing(env, normal_xy, point_xy):
    """The +-normal that points from the surface toward the robot's base (the side a hand can reach)."""
    base = np.asarray(
        env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")], dtype=float
    )[:2]
    n = np.asarray(normal_xy, dtype=float)
    s = float(np.dot(base - np.asarray(point_xy, dtype=float), n))
    return n if s >= 0 else -n


def measure_door_front(
    env,
    obs,
    grounder,
    furniture,
    instruction,
    size,
    support_z,
    furniture_box=None,
    *,
    context,
):
    """Door geometry from the FRONT camera, which sees an open door's inner face head-on: the panel box gives the
    plane (DepthVLM depth at its centre -- a large flat surface, where the depth is reliable -- and the depth face
    normal), its side-edge rays cut with that plane give the hinge edge and the free edge, its top/bottom edges the
    mid-height.  The hinge edge is the side nearer the appliance body.  Returns (geometry or None, record).
    """
    rec: dict[str, Any] = {}
    if grounder is None:
        return (None, rec)
    image = raw_image(obs, context=context)
    q_image, flipped = _upright(image, context=context)
    response, value = safe_ask_structured(
        grounder, q_image, _door_prompt(furniture, instruction), 96
    )
    box = value.get("bbox_2d") or value.get("bbox") if isinstance(value, dict) else None
    box = normalize_box(box) if box is not None else None
    box = _unflip_box(box, flipped)
    rec["door_response"] = response[:120]
    rec["door_box"] = None if box is None else [round(v, 1) for v in box]
    if box is None or box[2] - box[0] < 15 or box[3] - box[1] < 15:
        return (None, rec)
    depth, cam2world, focal, _, scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    c = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    z_cam = sample_point_depth(depth, c, radius=6) * scale
    centre = np.asarray(
        deproject(c, z_cam, cam2world, focal, size, context=context), dtype=float
    )
    rec["centre_world"] = centre.round(3).tolist()
    aspect = (box[3] - box[1]) / max(box[2] - box[0], 1e-06)
    n_axis = np.array([1.0, 0.0]) if aspect < 1.0 else np.array([0.0, 1.0])
    rec["box_aspect"] = round(float(aspect), 2)
    plane_n = np.r_[n_axis, 0.0]
    rays = {
        "left": _ray([box[0], c[1]], cam2world, focal, size, context=context),
        "right": _ray([box[2], c[1]], cam2world, focal, size, context=context),
        "top": _ray([c[0], box[1]], cam2world, focal, size, context=context),
        "bottom": _ray([c[0], box[3]], cam2world, focal, size, context=context),
    }
    cuts = {k: _ray_plane(r[0], r[1], centre, plane_n) for k, r in rays.items()}
    if any((p is None for p in cuts.values())):
        rec["fail"] = "edge_ray_misses_plane"
        return (None, rec)
    base = np.asarray(
        env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")], dtype=float
    )[:2]
    d_left, d_right = (
        float(np.linalg.norm(cuts["left"][:2] - base)),
        float(np.linalg.norm(cuts["right"][:2] - base)),
    )
    rec["edge_base_dist_left_right"] = [round(d_left, 3), round(d_right, 3)]
    hinge_side = "left" if d_left > d_right else "right"
    free_side = "right" if hinge_side == "left" else "left"
    hinge, free = (cuts[hinge_side], cuts[free_side])
    n_out = robot_facing(env, n_axis, centre[:2])
    length = float(np.linalg.norm(free[:2] - hinge[:2]))
    z_mid = float((cuts["top"][2] + cuts["bottom"][2]) / 2)
    z_top = float(max(cuts["top"][2], cuts["bottom"][2]))
    rec.update(
        {
            "normal_axis": n_axis.tolist(),
            "outer_normal": n_out.tolist(),
            "hinge_side": hinge_side,
            "hinge_world": hinge[:2].round(3).tolist(),
            "free_world": free[:2].round(3).tolist(),
            "door_length_m": round(length, 3),
            "z_mid": round(z_mid, 3),
            "z_top": round(z_top, 3),
        }
    )
    if (
        length < 0.08
        or length > 0.6
        or (not support_z - 0.05 <= z_mid <= support_z + 0.5)
    ):
        rec["fail"] = "implausible_geometry"
        return (None, rec)
    return (
        {
            "hinge": hinge[:2].copy(),
            "free": np.r_[free[:2], z_mid],
            "n_out": n_out,
            "z_top": z_top,
            "centre": centre,
            "plane_n": plane_n,
            "rays": {
                "hinge": rays[hinge_side],
                "free": rays[free_side],
                "top": rays["top"],
                "bottom": rays["bottom"],
            },
        },
        rec,
    )


PROBE_REACH_M = 0.15


def recut(geom, delta_m):
    """Re-cut the stored edge rays on the plane shifted by delta along the outer normal (after a tactile contact)."""
    centre = (
        np.asarray(geom["centre"], dtype=float)
        + np.r_[np.asarray(geom["n_out"], dtype=float) * delta_m, 0.0]
    )
    cuts = {
        k: _ray_plane(r[0], r[1], centre, geom["plane_n"])
        for k, r in geom["rays"].items()
    }
    if any((p is None for p in cuts.values())):
        return None
    z_mid = float((cuts["top"][2] + cuts["bottom"][2]) / 2)
    return {
        "hinge": cuts["hinge"][:2],
        "free": np.r_[cuts["free"][:2], z_mid],
        "z_top": float(max(cuts["top"][2], cuts["bottom"][2])),
    }


PEG_CLEARANCE_M = 0.024
PROBE_NEAR_HINGE_M = 0.05
APPROACH_LIFT_M = 0.15
ARC_MAX_DEG = 110.0
PROBE_BACKOFF_M = 0.04
ARC_STEP_DEG = 10.0
PUSH_RADIUS_M = 0.12


def _frame(handle, hinge, n_out):
    handle, hinge, n_out = (
        np.asarray(handle, dtype=float),
        np.asarray(hinge, dtype=float),
        np.asarray(n_out, dtype=float),
    )
    length = float(np.linalg.norm(handle[:2] - hinge))
    e0 = (handle[:2] - hinge) / max(length, 1e-06)
    n_out = n_out - e0 * float(np.dot(n_out, e0))
    n_out = n_out / max(float(np.linalg.norm(n_out)), 1e-06)
    return (
        hinge,
        e0,
        n_out,
        -n_out,
        min(PUSH_RADIUS_M, 0.5 * length),
        float(handle[2]),
    )


def close_door_by_arc(
    env, obs, handle, hinge, n_out, base_rot, events, mode, record, geom=None
):
    """Top-down closed gripper on the outer face, PUSH_RADIUS_M from the hinge, following the closing arc."""
    hinge, e0, n_out, n_in, r, z = _frame(handle, hinge, n_out)
    if geom is not None:
        z = max(z, float(geom["z_top"]) - 0.06)

    def peg(phi):
        e = np.cos(phi) * e0 + np.sin(phi) * n_in
        n = np.cos(phi) * n_out + np.sin(phi) * e0
        return np.r_[hinge + r * e + PEG_CLEARANCE_M * n, z]

    record["door_push"] = {
        "hinge": hinge.round(3).tolist(),
        "axis": e0.round(3).tolist(),
        "outer_normal": n_out.round(3).tolist(),
        "radius_m": round(r, 3),
        "z": round(z, 3),
    }
    obs = hold(env, obs, 1.0, 6, events, f"door_peg_close:{mode}")
    p_probe = np.r_[hinge + e0 * PROBE_NEAR_HINGE_M + PEG_CLEARANCE_M * n_out, z]
    start = np.r_[p_probe[:2] + n_out * PROBE_BACKOFF_M, z]
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[start[:2], z + APPROACH_LIFT_M],
        base_rot,
        1.0,
        events,
        f"door_above:{mode}",
        max_steps=50,
    )
    obs, reached, res_m, _, _ = pose_servo(
        env,
        obs,
        start,
        base_rot,
        1.0,
        events,
        f"door_descend:{mode}",
        max_steps=40,
        stop_on_stall=True,
    )
    record["door_push"]["descend"] = {"reached": reached, "residual_m": round(res_m, 3)}
    goal = np.r_[start[:2] + n_in * (PROBE_BACKOFF_M + PROBE_REACH_M), z]
    obs, p_reached, _, _, p_steps = pose_servo(
        env,
        obs,
        goal,
        base_rot,
        1.0,
        events,
        f"door_face_probe:{mode}",
        max_steps=60,
        stop_on_stall=True,
    )
    contact = np.asarray(obs["robot0_eef_pos"], dtype=float)
    if not p_reached:
        delta = float(np.dot(contact[:2] - p_probe[:2], n_out))
        record["door_push"]["face_probe"] = {
            "contact": contact.round(3).tolist(),
            "plane_shift_m": round(delta, 3),
            "steps": p_steps,
        }
        fixed = recut(geom, delta) if geom is not None else None
        if fixed is not None:
            hinge, e0, n_out, n_in, r, z_new = _frame(
                fixed["free"], fixed["hinge"], n_out
            )
            record["door_push"]["recut"] = {
                "hinge": hinge.round(3).tolist(),
                "free": fixed["free"].round(3).tolist(),
                "z": round(z_new, 3),
            }
            z = float(np.clip(z_new, z - 0.1, z + 0.1))
        else:
            hinge = hinge + n_out * delta
    else:
        record["door_push"]["face_probe"] = {"contact": None, "steps": p_steps}
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2] + n_out * 0.015, z],
        base_rot,
        1.0,
        events,
        f"door_backoff:{mode}",
        max_steps=15,
    )
    p0 = peg(0.0)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[p0[:2] + n_out * 0.015, z],
        base_rot,
        1.0,
        events,
        f"door_to_arc_start:{mode}",
        max_steps=30,
    )
    stalls, last, swing, backoff = (
        0,
        np.asarray(obs["robot0_eef_pos"], dtype=float).copy(),
        0.0,
        0.0,
    )
    for k in range(1, int(ARC_MAX_DEG / ARC_STEP_DEG) + 1):
        phi = np.radians(ARC_STEP_DEG * k)
        target = peg(phi)
        if backoff > 0:
            n_phi = np.cos(phi) * n_out + np.sin(phi) * e0
            target = np.r_[target[:2] + n_phi * backoff, target[2]]
        obs, *_ = pose_servo(
            env,
            obs,
            target,
            base_rot,
            1.0,
            events,
            f"door_arc:{int(ARC_STEP_DEG * k)}",
            max_steps=30,
            stop_on_stall=True,
        )
        now = np.asarray(obs["robot0_eef_pos"], dtype=float)
        rel = now[:2] - hinge
        swing = float(
            np.degrees(np.arctan2(float(np.dot(rel, n_in)), float(np.dot(rel, e0))))
        )
        if float(np.linalg.norm(now[:2] - last[:2])) < 0.01:
            stalls += 1
            backoff = min(backoff + 0.015, 0.045)
            if stalls >= 3:
                break
        else:
            stalls = 0
        last = now
    record["door_push"]["swing_deg"] = round(swing, 1)
    record["travel_m"] = round(float(np.radians(max(swing, 0.0)) * r), 3)
    record["grip"] = "push"
    record["stalled"] = stalls >= 2
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2], cur[2] + 0.12],
        base_rot,
        -1.0,
        events,
        f"door_lift:{mode}",
        max_steps=30,
    )
    return obs


def fit_hinge(path_pts, bar0, out):
    """Hinge and radius from the pulled eef path.  The hinge axis lies BEHIND the bar (its standoff off the face
    plus the hinge recess: 2-6 cm along -out) on a line parallel to the closed face; that band resolves the
    short-arc ambiguity (a 50 deg arc alone admits a family of circles).  Returns (hinge_xy, radius, spread).
    """
    pts = np.asarray(path_pts, dtype=float)
    out = np.asarray(out, dtype=float)
    axis = np.array([out[1], -out[0]])
    best = None
    for L in np.linspace(-0.45, 0.45, 181):
        if abs(L) < 0.08:
            continue
        for s in np.linspace(0.02, 0.06, 9):
            H = np.asarray(bar0, dtype=float)[:2] + axis * L - out * s
            dists = np.linalg.norm(pts - H, axis=1)
            spread = float(np.std(dists))
            if best is None or spread < best[0]:
                best = (spread, H, float(np.mean(dists)))
    return (best[1], best[2], best[0])


FINISH_RADIUS_FRACTION = 0.65
FINISH_APPROACH_RISE_M = 0.15
FINISH_SWEEP_START_DEG = 30.0
FINISH_BEHIND_M = 0.03
FINISH_MAX_DEG = 100.0
FINISH_STEP_DEG = 8.0


def finish_door_open(env, obs, path_pts, bar0, out, z, base_rot, events, mode, record):
    """Continue a door's opening past the pull's reach limit: after the pinch is released, a top-down closed hand
    inside the swing pushes the door's inner face outward along the fitted arc until the door stalls (or 125 deg).
    Triggered only by an observed stall of the pull short of the target travel."""
    pts = np.asarray(path_pts, dtype=float)
    if len(pts) < 3:
        record["finish"] = {"skipped": "too_few_path_points"}
        return obs
    H, R, spread = fit_hinge(pts, bar0, out)
    e0 = (pts[0] - H) / max(float(np.linalg.norm(pts[0] - H)), 1e-06)
    e_last = (pts[-1] - H) / max(float(np.linalg.norm(pts[-1] - H)), 1e-06)
    sense = float(np.sign(e0[0] * e_last[1] - e0[1] * e_last[0]) or 1.0)
    rec: dict[str, Any] = {
        "hinge": H.round(3).tolist(),
        "radius_m": round(R, 3),
        "fit_spread_m": round(spread, 4),
        "sense": sense,
        "steps": [],
    }
    record["finish"] = rec

    def rot2(v, a):
        return np.array(
            [np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]]
        )

    r = FINISH_RADIUS_FRACTION * R

    def peg(phi):
        e = rot2(e0, phi)
        tng = rot2(e, sense * np.pi / 2)
        return np.r_[H + r * e - tng * FINISH_BEHIND_M, z]

    obs = hold(env, obs, 1.0, 6, events, f"finish_peg_close:{mode}")
    phi0 = sense * np.radians(FINISH_SWEEP_START_DEG)
    p0 = peg(phi0)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[p0[:2], z + FINISH_APPROACH_RISE_M],
        base_rot,
        1.0,
        events,
        f"finish_above:{mode}",
        max_steps=30,
    )
    obs, reached, res_m, _, _ = pose_servo(
        env,
        obs,
        p0,
        base_rot,
        1.0,
        events,
        f"finish_descend:{mode}",
        max_steps=35,
        stop_on_stall=True,
    )
    rec["descend"] = {"reached": reached, "residual_m": round(res_m, 3)}
    stalls, swept = (0, 0.0)
    for k in range(1, int(FINISH_MAX_DEG / FINISH_STEP_DEG) + 1):
        phi = phi0 + sense * np.radians(FINISH_STEP_DEG * k)
        if abs(phi) > np.radians(FINISH_MAX_DEG):
            break
        before = np.asarray(obs["robot0_eef_pos"], dtype=float)[:2].copy()
        obs, *_ = pose_servo(
            env,
            obs,
            peg(phi),
            base_rot,
            1.0,
            events,
            f"finish_arc:{int(np.degrees(abs(phi)))}",
            max_steps=25,
            stop_on_stall=True,
        )
        moved = float(
            np.linalg.norm(np.asarray(obs["robot0_eef_pos"], dtype=float)[:2] - before)
        )
        rec["steps"].append([round(float(np.degrees(phi)), 1), round(moved, 3)])
        swept = float(np.degrees(abs(phi)))
        if moved < 0.01:
            stalls += 1
            if stalls >= 2:
                break
        else:
            stalls = 0
    rec["swept_to_deg"] = round(swept, 1)
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2], cur[2] + 0.12],
        base_rot,
        -1.0,
        events,
        f"finish_lift:{mode}",
        max_steps=25,
    )
    return obs
