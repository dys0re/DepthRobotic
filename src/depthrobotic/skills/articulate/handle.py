"""skills / articulate / handle: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import WORKSPACE_X, WORKSPACE_Y, hold
from depthrobotic.geometry.camera import (
    camera_geometry,
    deproject,
    intersect_horizontal_plane,
    project_world_to_raw,
    raw_ray,
)
from depthrobotic.geometry.depth import frame_geometry, sample_point_depth
from depthrobotic.perception.images import (
    crop_from_normalized,
    crop_point_to_global,
    observation_image,
    raw_image,
)
from depthrobotic.perception.parsing import (
    normalize_box,
    normalize_point,
    safe_ask_structured,
)
from depthrobotic.prompts import _furniture_prompt, _handles_prompt, _point_prompt
from depthrobotic.skills.simple import execute_contact_articulation
from scipy.spatial.transform import Rotation
from typing import Any
import json
import numpy as np
import re


def _snap(v):
    v = np.asarray(v, dtype=float)
    if abs(v[0]) >= abs(v[1]):
        return np.array([np.sign(v[0]) or 1.0, 0.0])
    return np.array([0.0, np.sign(v[1]) or 1.0])


REVOLUTE_WORDS = ("microwave", "door", "cabinet door", "fridge", "oven")


def _unflip_box(b, flipped):
    return b if not flipped or b is None else [b[0], 1000.0 - b[3], b[2], 1000.0 - b[1]]


def _parse_boxes(response):
    text = response.replace("```json", "").replace("```", "")
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    start = min(starts) if starts else -1
    while start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            nxt = [
                i
                for i in (text.find("[", start + 1), text.find("{", start + 1))
                if i >= 0
            ]
            start = min(nxt) if nxt else -1
            continue
        items = value if isinstance(value, list) else [value]
        boxes = []
        for item in items:
            box = (
                item.get("bbox_2d") or item.get("bbox") or item.get("box")
                if isinstance(item, dict)
                else item if isinstance(item, list) and len(item) == 4 else None
            )
            box = normalize_box(box) if box is not None else None
            if box is not None and box[2] - box[0] > 2 and (box[3] - box[1] > 2):
                boxes.append(box)
        return boxes
    return []


def _upright(image, *, context):
    """The articulation primitive shows the VLM world-upright images (its wrist views are de-rolled for the same
    reason).  Under the historical raw frame (ORIENT_FIX=0) the front view is vertically mirrored, and the VLM then
    boxes the background counter's drawer fronts instead of the cabinet's bars on the far-side scenes; flipping the
    image for the query and mapping its answers back keeps the rest of the pipeline in its own frame.
    """
    from PIL import Image as PILImage

    return (
        (image, False)
        if context.settings.orient_fix
        else (image.transpose(PILImage.FLIP_TOP_BOTTOM), True)
    )


NORMAL_SAMPLE_PX = 30.0


def face_normal_from_depth(depth, point, cam2world, focal, size, scale, *, context):
    """Outward horizontal normal of the surface around an image point, from three deprojected depth samples."""

    def world_at(px):
        z_cam = sample_point_depth(depth, px, radius=4) * scale
        return deproject(px, z_cam, cam2world, focal, size, context=context)

    a = world_at(point)
    b = world_at([min(point[0] + NORMAL_SAMPLE_PX, 999.0), point[1]])
    c = world_at([point[0], min(point[1] + NORMAL_SAMPLE_PX, 999.0)])
    n = np.cross(b - a, c - a)
    if float(np.linalg.norm(n)) < 1e-09:
        return None
    n = n / float(np.linalg.norm(n))
    if float(np.dot(n, cam2world[:3, 3] - a)) < 0:
        n = -n
    if abs(float(n[2])) > 0.8:
        return None
    return n[:2] / max(float(np.linalg.norm(n[:2])), 1e-09)


ORDINALS = {
    "top": "max",
    "upper": "max",
    "bottom": "min",
    "lower": "min",
    "middle": "mid",
    "center": "mid",
}


def _handles_prompt_retry(furniture):
    return f'This crop of the front camera shows the {furniture}, which has several drawers or doors stacked vertically, each with its own handle bar. Return one bounding box per handle bar, all of them, including a drawer that is pulled out. JSON only, normalized 0..1000 within this crop: [{{"bbox_2d":[x1,y1,x2,y2]}}, ...]'


def _unflip_point(p, flipped):
    return p if not flipped or p is None else [float(p[0]), 1000.0 - float(p[1])]


def _ray(point, cam2world, focal, size, *, context):
    origin = np.asarray(cam2world[:3, 3], dtype=float)
    direction = cam2world[:3, :3] @ raw_ray(point, focal, size, context=context)
    return (origin, direction / float(np.linalg.norm(direction)))


def _furniture_phrase(target):
    m = re.search("\\bof (?:the )?(.+)$", target.strip().lower())
    phrase = m.group(1).strip() if m else target.strip().lower()
    return re.sub("^(the|a|an)\\s+", "", phrase)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ground_handle(env, obs, grounder, target, instruction, size, support_z, *, context):
    """Anchor-first: box the furniture, list every handle bar inside its crop, deproject each with DepthVLM depth,
    and let geometry resolve the ordinal ("top drawer" = highest bar in world z, whatever the camera makes of it).
    """
    image = raw_image(obs, context=context)
    furniture = _furniture_phrase(target)
    rec: dict[str, Any] = {"furniture": furniture}
    q_image, flipped = _upright(image, context=context)
    f_resp, f_val = safe_ask_structured(
        grounder, q_image, _furniture_prompt(furniture, instruction), 96
    )
    fbox = (
        f_val.get("bbox_2d") or f_val.get("bbox") if isinstance(f_val, dict) else None
    )
    fbox = normalize_box(fbox) if fbox is not None else None
    fbox = _unflip_box(fbox, flipped)
    rec["vlm_frame"] = "upright"
    rec["furniture_box"] = None if fbox is None else [round(v, 1) for v in fbox]
    depth, cam2world, focal, _, scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    candidates: list[dict[str, Any]] = []
    if fbox is not None and fbox[2] - fbox[0] > 20 and (fbox[3] - fbox[1] > 20):
        center = [(fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2]
        frac = float(
            min(0.95, max(fbox[2] - fbox[0], fbox[3] - fbox[1]) / 1000.0 * 1.3 + 0.05)
        )
        crop, bounds = crop_from_normalized(image, center, frac)
        q_crop, _ = _upright(crop, context=context)
        h_resp, _ = grounder.generate_response(q_crop, _handles_prompt(furniture), 200)
        rec["handles_response"] = h_resp[:160]
        cboxes = [_unflip_box(b, flipped) for b in _parse_boxes(h_resp)]
        h_resp2, _ = grounder.generate_response(
            q_crop, _handles_prompt_retry(furniture), 200
        )
        rec["handles_response_2"] = h_resp2[:160]
        for extra in [_unflip_box(b, flipped) for b in _parse_boxes(h_resp2)]:
            if all((_iou(extra, b) < 0.3 for b in cboxes)):
                cboxes.append(extra)
        for cbox in cboxes:
            tl = crop_point_to_global([cbox[0], cbox[1]], bounds, image.size)
            br = crop_point_to_global([cbox[2], cbox[3]], bounds, image.size)
            gbox = [tl[0], tl[1], br[0], br[1]]
            c = [(gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2]
            try:
                z_cam = sample_point_depth(depth, c, radius=4) * scale
                w = deproject(c, z_cam, cam2world, focal, size, context=context)
            except Exception:
                continue
            if np.isfinite(w).all():
                if not (
                    WORKSPACE_X[0] - 0.05 <= float(w[0]) <= WORKSPACE_X[1] + 0.05
                    and WORKSPACE_Y[0] - 0.05 <= float(w[1]) <= WORKSPACE_Y[1] + 0.05
                ):
                    rec["n_handles_outside_workspace"] = (
                        rec.get("n_handles_outside_workspace", 0) + 1
                    )
                    continue
                o, d = _ray(c, cam2world, focal, size, context=context)
                candidates.append(
                    {"box": gbox, "center": c, "world": w, "origin": o, "direction": d}
                )
    rec["n_handles"] = len(candidates)
    chosen = None
    if candidates:
        word = next((w for w in target.lower().split() if w in ORDINALS), None)
        order = sorted(candidates, key=lambda c: float(c["world"][2]))
        if word and ORDINALS[word] == "max":
            chosen = order[-1]
        elif word and ORDINALS[word] == "min":
            chosen = order[0]
        elif word:
            chosen = order[len(order) // 2]
        else:
            chosen = max(
                candidates,
                key=lambda c: (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]),
            )
        rec["handle_z_candidates"] = [
            round(float(c["world"][2]) - support_z, 3) for c in order
        ]
        rec["ordinal"] = word
        point, box = (chosen["center"], chosen["box"])
        z = float(np.clip(chosen["world"][2], support_z - 0.05, support_z + 0.4))
    else:
        response, value = safe_ask_structured(
            grounder, q_image, _point_prompt(target, instruction, "front"), 64
        )
        point = _unflip_point(
            normalize_point(value.get("point") if isinstance(value, dict) else None),
            flipped,
        )
        if point is None:
            raise RuntimeError(f"Invalid handle grounding: {response[:120]}")
        rec["front_response"] = response[:100]
        z_cam = sample_point_depth(depth, point, radius=5) * scale
        z = float(
            np.clip(
                deproject(point, z_cam, cam2world, focal, size, context=context)[2],
                support_z - 0.05,
                support_z + 0.4,
            )
        )
        box = None
        o, d = _ray(point, cam2world, focal, size, context=context)
        chosen = {"box": None, "center": point, "origin": o, "direction": d}
        candidates = [chosen]
    world = np.asarray(
        intersect_horizontal_plane(point, z, cam2world, focal, size, context=context),
        dtype=float,
    )
    rec.update(
        {
            "front_point": [round(v, 1) for v in point],
            "depth_z_above_support": round(z - support_z, 3),
        }
    )
    out, normals = (None, {})
    bar_vertical = None
    if box is not None and box[2] - box[0] > 4 and (box[3] - box[1] > 4):
        aspect = (box[3] - box[1]) / max(box[2] - box[0], 1e-06)
        bar_vertical = aspect > 1.5 if aspect > 1.5 or aspect < 0.67 else None
        rec["bar_box"] = [round(v, 1) for v in box]
        rec["bar_pixel_aspect"] = round(float(aspect), 2)
        if "drawer" in target.lower():
            votes = {"y": 0, "x": 0}
            for c_ in candidates:
                b_ = c_.get("box")
                if b_ is None or b_[2] - b_[0] <= 4 or b_[3] - b_[1] <= 4:
                    continue
                a_ = (b_[3] - b_[1]) / max(b_[2] - b_[0], 1e-06)
                if a_ > 1.3:
                    votes["y"] += 1
                elif a_ < 0.77:
                    votes["x"] += 1
            rec["normal_aspect_votes"] = votes
            if votes["y"] != votes["x"]:
                normals["aspect"] = (
                    np.array([0.0, 1.0])
                    if votes["y"] > votes["x"]
                    else np.array([1.0, 0.0])
                )
    if fbox is not None:
        try:
            n = face_normal_from_depth(
                depth,
                [(fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2],
                cam2world,
                focal,
                size,
                scale,
                context=context,
            )
            if n is not None:
                normals["furniture_depth"] = _snap(n)
        except Exception:
            pass
    try:
        n = face_normal_from_depth(
            depth, point, cam2world, focal, size, scale, context=context
        )
        if n is not None:
            normals["bar_depth"] = _snap(n)
    except Exception as error:
        rec["normal_error"] = f"{type(error).__name__}: {error}"[:60]
    rec["normal_cues"] = {k: v.tolist() for k, v in normals.items()}
    for key in ("aspect", "furniture_depth", "bar_depth"):
        if key in normals:
            out = normals[key]
            rec["normal_source"] = key
            break
    rec["face_normal"] = None if out is None else out.round(3).tolist()
    rec["bar_vertical"] = bar_vertical
    front_rays = [
        {
            "center": c["center"],
            "box": c.get("box"),
            "origin": c["origin"],
            "direction": c["direction"],
            "world_depth": (
                None if c.get("world") is None else np.asarray(c["world"], dtype=float)
            ),
        }
        for c in candidates
    ]
    return {
        "world": world,
        "out": out,
        "bar_vertical": bar_vertical,
        "rec": rec,
        "front_rays": front_rays,
        "chosen": candidates.index(chosen) if chosen in candidates else 0,
    }


def pose_servo(
    env,
    obs,
    target_pos,
    target_rot,
    gripper,
    events,
    phase,
    max_steps=60,
    pos_tol=0.008,
    rot_tol=0.05,
    clip=0.8,
    stop_on_stall=False,
    min_stall_steps=5,
    stop_on_no_progress=False,
):
    """Drive position and orientation together. Returns (obs, reached, residual_m, residual_rad, steps).
    stop_on_stall: end once the hand has moved < 2 mm over the last 4 steps (after min_stall_steps, so a
    rotation-dominated start is not mistaken for a block).  stop_on_no_progress: end once the combined residual has
    not improved by 5 mm / 0.03 rad over 12 steps -- an unreachable pose keeps the arm drifting without converging,
    which a stall test never catches."""
    target_pos = np.asarray(target_pos, dtype=float)
    used, recent, hist = (0, [], [])
    for _ in range(max_steps):
        cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
        dp = target_pos - cur
        err = (
            target_rot * Rotation.from_quat(obs["robot0_eef_quat"]).inv()
        ).as_rotvec()
        if float(np.linalg.norm(dp)) < pos_tol and float(np.linalg.norm(err)) < rot_tol:
            break
        action = np.r_[
            np.clip(dp / 0.05, -clip, clip), np.clip(err / 0.5, -clip, clip), gripper
        ].astype(np.float32)
        obs, _, _, _ = env.step(action)
        used += 1
        recent.append(cur.copy())
        hist.append((float(np.linalg.norm(dp)), float(np.linalg.norm(err))))
        if (
            stop_on_stall
            and len(recent) >= max(5, min_stall_steps)
            and (float(np.linalg.norm(recent[-1] - recent[-5])) < 0.002)
        ):
            break
        if stop_on_no_progress and len(hist) >= 12:
            old, new = (hist[-12], hist[-1])
            if old[0] - new[0] < 0.015 and old[1] - new[1] < 0.03:
                break
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    res_m = float(np.linalg.norm(target_pos - cur))
    res_rad = float(
        (target_rot * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).magnitude()
    )
    reached = res_m < pos_tol * 2 and res_rad < rot_tol * 2
    events.append(
        {
            "phase": phase,
            "steps": used,
            "reached": bool(reached),
            "residual_m": round(res_m, 4),
            "residual_rad": round(res_rad, 3),
            "eef": cur.tolist(),
        }
    )
    return (obs, reached, res_m, res_rad, used)


def _retreat(env, obs, out, rot, base_rot, events, mode):
    obs = hold(env, obs, -1.0, 6, events, f"handle_release:{mode}")
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2] + out * 0.1, cur[2]],
        rot,
        -1.0,
        events,
        f"handle_retreat:{mode}",
        max_steps=25,
    )
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2], cur[2] + 0.1],
        base_rot,
        -1.0,
        events,
        f"handle_restore:{mode}",
        max_steps=35,
    )
    return obs


TOPDOWN_LEAN_DEG = 45.0
TOPDOWN_SLACK_M = 0.0
TRAVEL_M = {"prismatic": 0.16, "revolute": 0.25}
SEGMENT_M = 0.06


def push_segments(
    env, obs, into, z, rot, gripper, events, mode, record, already=0.0, budget=None
):
    """Push along `into` in segments at height z until the drawer stalls twice or the travel budget is spent."""
    budget = TRAVEL_M["prismatic"] + 0.04 if budget is None else budget
    start = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
    last, stalls, path, travel = (start.copy(), 0, [], 0.0)
    while already + travel < budget and stalls < 2:
        cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
        goal = np.r_[cur[:2] + into * SEGMENT_M, z]
        obs, *_ = pose_servo(
            env,
            obs,
            goal,
            rot,
            gripper,
            events,
            f"handle_push:{int(travel * 100)}",
            max_steps=30,
            stop_on_stall=True,
        )
        now = np.asarray(obs["robot0_eef_pos"], dtype=float)
        path.append((now[:2] - last[:2]).round(3).tolist())
        progress = float(np.dot((now - start)[:2], into))
        if progress - travel < 0.01:
            stalls += 1
        travel, last = (progress, now)
    record.update(
        {
            "grip": "push",
            "travel_m": round(already + travel, 3),
            "stalled": stalls >= 2,
            "path_segments": path,
        }
    )
    return obs


def push_topdown(
    env, obs, handle, out, into, base_rot, events, mode, record, support_z
):
    """Prismatic close of a LOW drawer (the pitched hand's body would meet the table): a top-down closed gripper,
    yawed so its 20 cm width runs along the face, pushes the drawer face beside the bar.  Along the push axis the
    hand is then ~6 cm thick and clears the protruding bar of the drawer above until the drawer is nearly shut.
    """
    axis = np.array([out[1], -out[0]])
    yaw = float(np.arctan2(axis[1], axis[0]))
    rot = Rotation.from_rotvec([0.0, 0.0, yaw + np.pi / 2]) * base_rot
    best = None
    for sign in (1.0, -1.0):
        cand = (
            Rotation.from_rotvec(np.r_[axis, 0.0] * np.radians(sign * TOPDOWN_LEAN_DEG))
            * rot
        )
        score = float(np.dot(cand.apply([0.0, 0.0, 1.0])[:2], into))
        if best is None or score > best[0]:
            best = (score, cand)
    rot_lean = best[1]
    lateral = record.get("bar_half_length_m", 0.065) + 0.03
    robot = np.asarray(env.home_xy, dtype=float)
    side = 1.0 if float(np.dot(axis, robot - handle[:2])) >= 0 else -1.0
    record["push_variant"] = "topdown_yawed_leaned"
    obs = hold(env, obs, 1.0, 6, events, f"handle_close_fingers:{mode}")
    home = np.r_[np.asarray(env.home_xy, dtype=float), support_z + 0.24]
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    cur_rot = Rotation.from_quat(obs["robot0_eef_quat"])
    if (
        float(np.linalg.norm(cur - home)) > 0.05
        or float((base_rot * cur_rot.inv()).magnitude()) > 0.15
    ):
        obs, *_ = pose_servo(
            env,
            obs,
            home,
            base_rot,
            1.0,
            events,
            f"handle_home_reset:{mode}",
            max_steps=45,
        )
    spot, found, attempts = (None, False, [])
    for k in range(3):
        spot = np.r_[
            handle[:2]
            + axis * side * lateral
            + out * (0.08 + TOPDOWN_SLACK_M + 0.1 * k),
            handle[2] + 0.025,
        ]
        obs, *_ = pose_servo(
            env,
            obs,
            np.r_[spot[:2], spot[2] + 0.12],
            rot,
            1.0,
            events,
            f"handle_above:{mode}:{k}",
            max_steps=60,
        )
        obs, reached, res_m, _, _ = pose_servo(
            env,
            obs,
            spot,
            rot,
            1.0,
            events,
            f"handle_descend:{mode}:{k}",
            max_steps=40,
            stop_on_stall=True,
        )
        if res_m > 0.015:
            attempts.append(
                {
                    "spot": spot.round(3).tolist(),
                    "descend_residual_m": round(res_m, 3),
                    "front_found": False,
                    "landed": True,
                }
            )
            cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
            obs, *_ = pose_servo(
                env,
                obs,
                np.r_[cur[:2], spot[2] + 0.12],
                rot,
                1.0,
                events,
                f"handle_lift_retry:{mode}:{k}",
                max_steps=25,
            )
            continue
        z_now = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
        obs, probe_reached, _, _, probe_steps = pose_servo(
            env,
            obs,
            np.r_[spot[:2] + into * 0.3, z_now],
            rot,
            1.0,
            events,
            f"handle_find_front:{mode}:{k}",
            max_steps=60,
            stop_on_stall=True,
        )
        contact = np.asarray(obs["robot0_eef_pos"], dtype=float)
        found = not probe_reached
        attempts.append(
            {
                "spot": spot.round(3).tolist(),
                "descend_residual_m": round(res_m, 3),
                "front_found": found,
                "contact": contact.round(3).tolist(),
                "probe_steps": probe_steps,
            }
        )
        if found:
            break
        obs, *_ = pose_servo(
            env,
            obs,
            np.r_[contact[:2], spot[2] + 0.12],
            rot,
            1.0,
            events,
            f"handle_lift_retry:{mode}:{k}",
            max_steps=25,
        )
    record["push_spot"] = spot.round(3).tolist()
    record["descend_attempts"] = attempts
    record["advance"] = {"reached": found, "residual_m": None}
    if not found:
        record["abort"] = "drawer_front_not_found"
        return _retreat(env, obs, out, rot, base_rot, events, mode)
    z_push = float(np.asarray(obs["robot0_eef_pos"], dtype=float)[2])
    probe_push = float(np.dot(contact[:2] - spot[:2], into))
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, lean_reached, lean_res, lean_rad, _ = pose_servo(
        env, obs, cur, rot_lean, 1.0, events, f"handle_lean:{mode}", max_steps=40
    )
    obs = push_segments(
        env, obs, into, z_push, rot_lean, 1.0, events, mode, record, budget=0.12
    )
    record["push_phases"] = {
        "probe_push_m": round(probe_push, 3),
        "lean": {
            "reached": lean_reached,
            "residual_m": round(lean_res, 3),
            "residual_rad": round(lean_rad, 3),
        },
        "leaned": {
            "travel_m": record.get("travel_m"),
            "stalled": record.get("stalled"),
        },
    }
    record["travel_m"] = round(
        max(probe_push, 0.0) + float(record.get("travel_m") or 0.0), 3
    )
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2] + out * 0.03, cur[2] + 0.12],
        rot_lean,
        -1.0,
        events,
        f"handle_lift:{mode}",
        max_steps=30,
    )
    obs, *_ = pose_servo(
        env, obs, home, base_rot, -1.0, events, f"handle_restore:{mode}", max_steps=40
    )
    return obs


PITCH_DOWN_DEG = 25.0


def _largest_cluster(points, radius):
    """Indices of the largest group of points that all lie within `radius` of one member (None points excluded)."""
    idx = [i for i, p in enumerate(points) if p is not None]
    if len(idx) <= 1:
        return set(idx)
    best = set()
    for i in idx:
        group = {
            j
            for j in idx
            if float(
                np.linalg.norm(np.asarray(points[i])[:2] - np.asarray(points[j])[:2])
            )
            <= radius
        }
        if len(group) > len(best):
            best = group
    return best


def _handles_prompt_wrist_retry(furniture):
    return f'This wrist-camera image shows the front of the {furniture} up close. It has several drawers or doors, each with its own handle bar. Return one bounding box per handle bar, all of them. JSON only, normalized 0..1000: [{{"bbox_2d":[x1,y1,x2,y2]}}, ...]'


SURVEY_NOMINAL_BAR_Z_M = 0.15
SURVEY_BASE_SHIFT_M = 0.15
SURVEY_EEF_Z_M = 0.2
TRIANG_MAX_SKEW_M = 0.03
LAST_WRIST_VIEW = {}


def _image_up(env, camera, size, world_point, *, context):
    """Image-space direction of world +z at a scene point (the pixel frame's y is not 'up' in general)."""
    a = np.asarray(
        project_world_to_raw(env, camera, world_point, size, context=context),
        dtype=float,
    )
    b = np.asarray(
        project_world_to_raw(
            env,
            camera,
            np.asarray(world_point, dtype=float) + np.array([0, 0, 0.05]),
            size,
            context=context,
        ),
        dtype=float,
    )
    u = b - a
    return u / max(float(np.linalg.norm(u)), 1e-06)


def deroll(env, image, size, world_point, *, context):
    """Rotate a wrist image so that world +z points up (a pitched hand rolls the camera by ~90 deg, and the VLM
    boxes handle bars far better upright).  Returns (rotated image, function mapping a normalized point in the
    rotated image back to the original frame)."""
    from PIL import Image as PILImage

    u = _image_up(env, "robot0_eye_in_hand", size, world_point, context=context)
    alpha = float(np.degrees(np.arctan2(u[1], u[0])) + 90.0)
    rotated = image.rotate(
        alpha, resample=PILImage.BILINEAR, expand=False, fillcolor=(0, 0, 0)
    )
    w, h = image.size
    c, s = (np.cos(np.radians(alpha)), np.sin(np.radians(alpha)))

    def back(point_norm):
        p = np.array(
            [
                point_norm[0] / 1000.0 * (w - 1) - w / 2.0,
                point_norm[1] / 1000.0 * (h - 1) - h / 2.0,
            ]
        )
        q = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])
        return [
            float((q[0] + w / 2.0) / (w - 1) * 1000.0),
            float((q[1] + h / 2.0) / (h - 1) * 1000.0),
        ]

    return (rotated, back, alpha)


def _handles_prompt_wrist(furniture):
    return f'This wrist-camera image looks straight at the front of the {furniture}. List the bounding boxes of ALL of its handle bars (one per drawer or door). JSON only, normalized 0..1000: [{{"bbox_2d":[x1,y1,x2,y2]}}, ...]'


def wrist_bar_rays(
    env, obs, grounder, furniture, size, prompt_fn=None, up_at=None, *, context
):
    """Every handle bar the wrist camera sees, as a world ray through its box centre (no depth).  With `up_at`
    (a world point near the bars) the image is de-rolled before the VLM sees it."""
    if grounder is None:
        return ([], "")
    image = observation_image(obs, "robot0_eye_in_hand_image", context=context)
    cam2world, focal, _ = camera_geometry(env, "robot0_eye_in_hand", size)
    back = None
    if up_at is not None:
        image, back, alpha = deroll(env, image, size, up_at, context=context)
        LAST_WRIST_VIEW.update({"image": image, "alpha": alpha})
    response, _ = grounder.generate_response(
        image, (prompt_fn or _handles_prompt_wrist)(furniture), 200
    )
    bars = []
    for box in _parse_boxes(response):
        c = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        ends = [[box[0], (box[1] + box[3]) / 2], [box[2], (box[1] + box[3]) / 2]]
        if back is not None:
            c = back(c)
            ends = [back(e) for e in ends]
        o, d = _ray(c, cam2world, focal, size, context=context)
        end_rays = [_ray(e, cam2world, focal, size, context=context)[1] for e in ends]
        bars.append(
            {"box": box, "center": c, "origin": o, "direction": d, "end_rays": end_rays}
        )
    return (bars, response[:160])


def _monotone_matches(front_order, wrist_order, skew):
    """Best rank-consistent pairing of two height-ordered lists: maximise pairs under the skew gate, then minimise
    total skew.  Enumerates the C(n, k) monotone injections of the shorter list into the longer one.
    """
    from itertools import combinations

    a, b = (
        (front_order, wrist_order)
        if len(front_order) <= len(wrist_order)
        else (wrist_order, front_order)
    )
    swapped = len(front_order) > len(wrist_order)
    best, best_key = ([], None)
    for slots in combinations(range(len(b)), len(a)):
        pairs = []
        for ai, bi in zip(a, slots):
            fi, wi = (b[bi], ai) if swapped else (ai, b[bi])
            s = skew.get((fi, wi))
            if s is not None and s <= TRIANG_MAX_SKEW_M:
                pairs.append((fi, wi, s))
        key = (len(pairs), -sum((p[2] for p in pairs)))
        if best_key is None or key > best_key:
            best, best_key = (pairs, key)
    return best


def _ray_plane(origin, direction, plane_point, normal):
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-08:
        return None
    t = float(np.dot(np.asarray(plane_point) - origin, normal)) / denom
    return None if t <= 0 else origin + direction * t


def _rank_by_height(items, up):
    """Indices of `items` (each with a pixel 'center') from the highest in the world to the lowest."""

    def height(i):
        c = items[i].get("center")
        return 0.0 if c is None else float(np.dot(np.asarray(c, dtype=float), up))

    return sorted(range(len(items)), key=lambda i: -height(i))


def _pick_by_ordinal(target, items, default):
    word = next((w for w in target.lower().split() if w in ORDINALS), None)
    if not items:
        return (None, word)
    order = sorted(items, key=lambda b: float(b["world"][2]))
    if word and ORDINALS[word] == "max":
        return (order[-1], word)
    if word and ORDINALS[word] == "min":
        return (order[0], word)
    if word:
        return (order[len(order) // 2], word)
    return (default(items), word)


SURVEY_STANDOFF_M = 0.25
SURVEY_SECOND_SHIFT_M = 0.12


def triangulate_rays(o1, d1, o2, d2):
    """Closest points of two rays: (midpoint, skew distance, range along ray 1, range along ray 2) or None."""
    w0 = np.asarray(o1, dtype=float) - np.asarray(o2, dtype=float)
    b, d, e = (float(np.dot(d1, d2)), float(np.dot(d1, w0)), float(np.dot(d2, w0)))
    denom = 1.0 - b * b
    if denom < 1e-06:
        return None
    t1, t2 = ((b * e - d) / denom, (e - b * d) / denom)
    p1, p2 = (o1 + t1 * d1, o2 + t2 * d2)
    return ((p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2)), t1, t2)


def _ray_vertical_line(o, d, xy):
    """Closest point on the ray to the vertical line through xy (bars of flush drawer fronts share x, y)."""
    tri = triangulate_rays(
        o, d, np.r_[np.asarray(xy, dtype=float), 0.0], np.array([0.0, 0.0, 1.0])
    )
    return None if tri is None else tri[0]


SURVEY_RISE_M = 0.18


def approach_rotation(base, into, bar_vertical, quarter_sign=-1):
    """Point the gripper along `into`.  Pointing down, the fingers close along world y; after pitching they close
    vertically across a horizontal bar; for a vertical bar add a quarter turn about the pointing axis (the -90 deg
    sense is the one the Panda wrist reaches at a microwave standoff; +90 hits joint 7's limit).
    """
    down = np.array([0.0, 0.0, -1.0])
    into = np.asarray(into, dtype=float)
    target_dir = np.r_[into, 0.0] if into.shape[0] == 2 else into
    target_dir = target_dir / max(float(np.linalg.norm(target_dir)), 1e-09)
    axis = np.cross(down, target_dir)
    n = float(np.linalg.norm(axis))
    angle = float(np.arccos(np.clip(np.dot(down, target_dir), -1.0, 1.0)))
    rot = base if n < 1e-09 else Rotation.from_rotvec(axis / n * angle) * base
    if bar_vertical:
        rot = Rotation.from_rotvec(target_dir * (quarter_sign * np.pi / 2)) * rot
    return rot


def _rank_complete(front_rays, front_order, view, support_z, max_skew=0.015):
    """True when the wrist view lists as many bars as the front view and every rank-paired ray pair meets within
    max_skew (a coincidence pairing needs a missing bar to shift the ranks)."""
    bars, order = (view["bars"], view["order"])
    if len(bars) != len(front_rays) or len(bars) < 1:
        return False
    for fi, wi in zip(front_order, order):
        tri = triangulate_rays(
            front_rays[fi]["origin"],
            front_rays[fi]["direction"],
            bars[wi]["origin"],
            bars[wi]["direction"],
        )
        if tri is None or tri[1] > max_skew or tri[2] <= 0.05 or (tri[3] <= 0.05):
            return False
        if not support_z - 0.05 <= float(tri[0][2]) <= support_z + 0.45:
            return False
    return True


def survey_triangulate(
    env,
    obs,
    grounder,
    target,
    furniture,
    size,
    support_z,
    front_rays,
    chosen_idx,
    out,
    rot,
    events,
    mode,
    record,
    *,
    context,
):
    """Second view of the handle bars from the wrist camera at a survey pose, triangulated against the front-camera
    rays.  The front camera looks along -x while a drawer face's normal is +-y: its rays run nearly parallel to the
    face, so a range error (DepthVLM was 10-20 cm off on 1.6 cm bars) slides a depth-based estimate along the ray.
    Two rays from separated viewpoints fix the bar without any depth.  Correspondence is by height rank in each
    view (a stack of bars keeps its order in any view) with the ray skew as a veto: skew alone mis-paired an open
    drawer's bar with the one behind it (their rays crossed within 2 mm).  Returns (obs, handle or None).
    """
    from depthrobotic.skills.articulate.door import robot_facing

    ref = front_rays[chosen_idx]
    o = ref["origin"]
    pre_cluster = _largest_cluster([fr.get("world_depth") for fr in front_rays], 0.3)
    members = [front_rays[i] for i in sorted(pre_cluster)] or [ref]
    d = np.mean([m["direction"] for m in members], axis=0)
    d = d / max(float(np.linalg.norm(d)), 1e-09)
    if abs(float(d[2])) < 1e-06:
        return (obs, None)
    t = (support_z + SURVEY_NOMINAL_BAR_Z_M - float(o[2])) / float(d[2])
    if t <= 0:
        return (obs, None)
    on_ray = o + t * d
    word = next((w for w in target.lower().split() if w in ORDINALS), None)
    cluster = _largest_cluster([fr.get("world_depth") for fr in front_rays], 0.3)
    front_order = [
        i
        for i in _rank_by_height(
            front_rays, _image_up(env, "agentview", size, on_ray, context=context)
        )
        if i in cluster
    ]
    if not front_order:
        front_order = _rank_by_height(
            front_rays, _image_up(env, "agentview", size, on_ray, context=context)
        )
    record["front_cluster"] = sorted(cluster)
    if word and ORDINALS[word] == "max":
        chosen_idx = front_order[0]
    elif word and ORDINALS[word] == "min":
        chosen_idx = front_order[-1]
    elif word:
        chosen_idx = front_order[len(front_order) // 2]
    record["front_height_order"] = front_order
    record["front_chosen"] = chosen_idx
    to_robot = np.asarray(env.home_xy, dtype=float) - on_ray[:2]
    to_robot = to_robot / max(float(np.linalg.norm(to_robot)), 1e-06)
    base_rot = Rotation.from_quat(obs["robot0_eef_quat"])
    base_xy = np.asarray(
        env.sim.data.body_xpos[env.sim.model.body_name2id("robot0_base")], dtype=float
    )[:2]
    nominal_xy = on_ray[:2] + out * SURVEY_STANDOFF_M + to_robot * SURVEY_BASE_SHIFT_M
    to_base = base_xy - nominal_xy
    to_base = to_base / max(float(np.linalg.norm(to_base)), 1e-06)
    rec: dict[str, Any] = {"tries": []}
    alt_out = robot_facing(env, np.array([out[1], -out[0]]), on_ray[:2])
    plans = [
        (0.0, False, out),
        (0.1, True, out),
        (0.2, True, out),
        (0.0, False, alt_out),
        (0.1, True, alt_out),
    ]
    for k, (shift, elevated, out_k) in enumerate(plans):
        if (
            not np.allclose(out_k, out)
            and rec["tries"]
            and (rec["tries"][-1].get("axis") == "assumed")
        ):
            nominal_xy = (
                on_ray[:2] + out_k * SURVEY_STANDOFF_M + to_robot * SURVEY_BASE_SHIFT_M
            )
            to_base = base_xy - nominal_xy
            to_base = to_base / max(float(np.linalg.norm(to_base)), 1e-06)
        survey = np.r_[
            nominal_xy + to_base * shift,
            (
                float(on_ray[2]) + SURVEY_RISE_M
                if elevated
                else support_z + SURVEY_EEF_Z_M
            ),
        ]
        look3 = on_ray - survey
        look = look3[:2] / max(float(np.linalg.norm(look3[:2])), 1e-06)
        survey_rot = approach_rotation(
            base_rot, look3 / max(float(np.linalg.norm(look3)), 1e-06), False
        )
        obs, reached, res_m, res_rad, _ = pose_servo(
            env,
            obs,
            survey,
            survey_rot,
            -1.0,
            events,
            f"handle_survey{(k if k else '')}:{mode}",
            max_steps=35,
            stop_on_stall=True,
            min_stall_steps=12,
            stop_on_no_progress=True,
        )
        rec["tries"].append(
            {
                "pose": survey.round(3).tolist(),
                "elevated": elevated,
                "reached": reached,
                "residual_m": round(res_m, 3),
                "residual_rad": round(res_rad, 3),
                "axis": "assumed" if np.allclose(out_k, out) else "alternative",
            }
        )
        if reached or (res_m <= 0.05 and res_rad <= 0.2):
            break
    rec.update(
        {
            "pose": survey.round(3).tolist(),
            "look": look.round(3).tolist(),
            "reached": reached,
            "residual_m": round(res_m, 3),
            "residual_rad": round(res_rad, 3),
        }
    )
    record["survey_view"] = "elevated" if rec["tries"][-1]["elevated"] else "level"
    record["survey"] = rec
    lateral = np.array([out[1], -out[0]])
    if float(np.dot(lateral, np.asarray(env.home_xy, dtype=float) - survey[:2])) < 0:
        lateral = -lateral
    up_w = _image_up(env, "robot0_eye_in_hand", size, on_ray, context=context)
    views = []
    for v in range(2):
        if (
            v == 1
            and mode == "open"
            and views
            and _rank_complete(front_rays, front_order, views[0], support_z)
        ):
            rec["second"] = "skipped_rank_complete"
            views.append(views[0])
            break
        if v == 1:
            pose2 = np.r_[survey[:2] + lateral * SURVEY_SECOND_SHIFT_M, survey[2]]
            look3b = on_ray - pose2
            obs, reached2, res_m2, res_rad2, _ = pose_servo(
                env,
                obs,
                pose2,
                approach_rotation(
                    base_rot, look3b / max(float(np.linalg.norm(look3b)), 1e-06), False
                ),
                -1.0,
                events,
                f"handle_survey2:{mode}",
                max_steps=40,
                stop_on_stall=True,
                min_stall_steps=12,
            )
            rec["second"] = {
                "pose": pose2.round(3).tolist(),
                "reached": reached2,
                "residual_m": round(res_m2, 3),
                "residual_rad": round(res_rad2, 3),
            }
            up_w = _image_up(env, "robot0_eye_in_hand", size, on_ray, context=context)
        bars, response = wrist_bar_rays(
            env, obs, grounder, furniture, size, up_at=on_ray, context=context
        )
        if len(bars) < len(front_rays):
            more, response2 = wrist_bar_rays(
                env,
                obs,
                grounder,
                furniture,
                size,
                prompt_fn=_handles_prompt_wrist_retry,
                up_at=on_ray,
                context=context,
            )
            if len(more) > len(bars):
                bars, response = (more, response2)
                rec[f"wrist_retry_{v}"] = True
        views.append(
            {
                "bars": bars,
                "order": _rank_by_height(bars, up_w) if bars else [],
                "response": response,
            }
        )
    rec["n_wrist_bars"] = [len(v["bars"]) for v in views]
    rec["response"] = views[0]["response"]
    rec["response2"] = views[1]["response"]
    rec["wrist_height_order"] = [v["order"] for v in views]

    def pair(ra, rb):
        tri = triangulate_rays(
            ra["origin"], ra["direction"], rb["origin"], rb["direction"]
        )
        if tri is None:
            return None
        p, _, t1, t2 = tri
        if (
            t1 <= 0.05
            or t2 <= 0.05
            or (not support_z - 0.05 <= float(p[2]) <= support_z + 0.45)
        ):
            return None
        return p

    W1, W2 = (views[0]["bars"], views[1]["bars"])
    triples = []
    for i, fr in enumerate(front_rays):
        for j, w1 in enumerate(W1):
            p1 = pair(fr, w1)
            if p1 is None:
                continue
            for k, w2 in enumerate(W2):
                p2, p3 = (pair(fr, w2), pair(w1, w2))
                if p2 is None or p3 is None:
                    continue
                pts = np.stack([p1, p2, p3])
                spread = max(
                    (
                        float(np.linalg.norm(pts[a] - pts[b]))
                        for a, b in ((0, 1), (0, 2), (1, 2))
                    )
                )
                if spread <= TRIANG_MAX_SKEW_M:
                    triples.append((spread, i, j, k, pts.mean(axis=0)))
    triples.sort(key=lambda t: t[0])
    used_i, used_j, used_k, tri_bars = (set(), set(), set(), [])
    for spread, i, j, k, p in triples:
        if i in used_i or j in used_j or k in used_k:
            continue
        if not (
            WORKSPACE_X[0] - 0.05 <= float(p[0]) <= WORKSPACE_X[1] + 0.05
            and WORKSPACE_Y[0] - 0.05 <= float(p[1]) <= WORKSPACE_Y[1] + 0.05
        ):
            continue
        (used_i.add(i), used_j.add(j), used_k.add(k))
        tri_bars.append(
            {
                "front": i,
                "wrist": j,
                "wrist2": k,
                "world": np.asarray(p, dtype=float),
                "skew": spread,
            }
        )
    rec["consistency"] = "three_view"
    if not tri_bars and W1:
        skew_of, point_of = ({}, {})
        for j, wb in enumerate(W1):
            for i, fr in enumerate(front_rays):
                tri = triangulate_rays(
                    fr["origin"], fr["direction"], wb["origin"], wb["direction"]
                )
                if tri is not None and pair(fr, wb) is not None:
                    skew_of[i, j], point_of[i, j] = (tri[1], tri[0])
        tri_bars = [
            {
                "front": i,
                "wrist": j,
                "world": np.asarray(point_of[i, j], dtype=float),
                "skew": s,
            }
            for i, j, s in _monotone_matches(front_order, views[0]["order"], skew_of)
        ]
        rec["consistency"] = "two_view_rank"
    rec["triangulated"] = [
        {
            "front": b["front"],
            "wrist": b["wrist"],
            "wrist2": b.get("wrist2"),
            "skew_m": round(b["skew"], 3),
            "world": b["world"].round(3).tolist(),
        }
        for b in tri_bars
    ]
    rec["ordinal"] = word
    axes = []
    for b in tri_bars:
        if b.get("wrist2") is None:
            continue
        w1, w2 = (W1[b["wrist"]], W2[b["wrist2"]])
        ends = []
        for side in (0, 1):
            tri = triangulate_rays(
                w1["origin"], w1["end_rays"][side], w2["origin"], w2["end_rays"][side]
            )
            if tri is None or tri[1] > 0.06 or tri[2] <= 0.05 or (tri[3] <= 0.05):
                ends = []
                break
            ends.append(tri[0][:2])
        if len(ends) == 2 and 0.06 <= float(np.linalg.norm(ends[1] - ends[0])) <= 0.4:
            axes.append((ends[1] - ends[0]) / float(np.linalg.norm(ends[1] - ends[0])))
    if axes:
        ax = np.mean(
            [a if float(np.dot(a, axes[0])) >= 0 else -a for a in axes], axis=0
        )
        ax = _snap(ax)
        n_meas = robot_facing(env, np.array([ax[1], -ax[0]]), on_ray[:2])
        rec["bar_axis_measured"] = ax.tolist()
        rec["out_measured"] = n_meas.tolist()
        record["out_measured"] = n_meas.tolist()
    bars = W1
    hit = next((b for b in tri_bars if b["front"] == chosen_idx), None)
    if hit is not None:
        rec["source"] = "triangulated"
        return (obs, np.asarray(hit["world"], dtype=float))
    if tri_bars:
        sib = min(
            tri_bars,
            key=lambda b: abs(
                front_order.index(b["front"]) - front_order.index(chosen_idx)
            ),
        )
        p = _ray_vertical_line(
            front_rays[chosen_idx]["origin"],
            front_rays[chosen_idx]["direction"],
            sib["world"][:2],
        )
        if p is not None and support_z - 0.05 <= float(p[2]) <= support_z + 0.45:
            rec["source"] = "sibling_vertical"
            rec["sibling"] = sib["front"]
            return (obs, np.asarray(p, dtype=float))
    if bars:
        plane = []
        for j, wb in enumerate(bars):
            w = _ray_plane(wb["origin"], wb["direction"], on_ray, np.r_[out, 0.0])
            if w is not None and support_z - 0.05 <= float(w[2]) <= support_z + 0.45:
                plane.append(
                    {"wrist": j, "world": np.asarray(w, dtype=float), "box": wb["box"]}
                )
        if plane:
            chosen, word = _pick_by_ordinal(
                target,
                plane,
                lambda items: max(
                    items,
                    key=lambda b: (b["box"][2] - b["box"][0])
                    * (b["box"][3] - b["box"][1]),
                ),
            )
            rec["source"] = "wrist_plane"
            rec["ordinal"] = word
            return (obs, np.asarray(chosen["world"], dtype=float))
    rec["source"] = None
    return (obs, None)


MIN_GRIP_GAP_M = 0.006
GRASP_OVERSHOOT_M = 0.0
MAX_GRIP_GAP_M = 0.045
ABORT_STANDOFF_RESIDUAL_M = 0.05
GRASP_ESCALATION_M = 0.025
PUSH_OVERSHOOT_M = 0.05
OPEN_DRAWER_SLACK_M = 0.2


def push_from_standoff(
    env, obs, handle, out, into, rot, base_rot, events, mode, record
):
    """Prismatic close: closed fingertips push the drawer FACE beside the bar (past the bar's end).  A flat face
    does not deflect the fingers the way the bar's cylinder did, and the fingers stay clear of the bar of the
    drawer below (straddling the bar with open fingers needs the height right to ~1.5 cm).
    """
    axis = np.array([out[1], -out[0]])
    lateral = record.get("bar_half_length_m", 0.065) + 0.03
    robot = np.asarray(env.home_xy, dtype=float)
    side = 1.0 if float(np.dot(axis, robot - handle[:2])) >= 0 else -1.0
    spot = np.r_[handle[:2] + axis * side * lateral, handle[2]]
    record["push_spot"] = spot.round(3).tolist()
    obs = hold(env, obs, 1.0, 6, events, f"handle_close_fingers:{mode}")
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, ss_reached, ss_res, _, _ = pose_servo(
        env,
        obs,
        np.r_[spot[:2] + out * float(np.dot(cur[:2] - handle[:2], out)), spot[2]],
        rot,
        1.0,
        events,
        f"handle_side_step:{mode}",
        max_steps=30,
        stop_on_no_progress=True,
    )
    if not ss_reached and ss_res > 0.05:
        record["advance"] = {
            "reached": False,
            "residual_m": round(ss_res, 3),
            "stage": "side_step",
        }
        record["travel_m"] = 0.0
        return obs
    before = np.asarray(obs["robot0_eef_pos"], dtype=float)[:2].copy()
    obs, adv_reached, adv_res, _, _ = pose_servo(
        env,
        obs,
        np.r_[spot[:2] + into * PUSH_OVERSHOOT_M, spot[2]],
        rot,
        1.0,
        events,
        f"handle_advance:{mode}",
        max_steps=70,
        stop_on_stall=True,
    )
    advanced = float(
        np.dot(np.asarray(obs["robot0_eef_pos"], dtype=float)[:2] - before, into)
    )
    record["advance"] = {
        "reached": adv_reached,
        "residual_m": round(adv_res, 3),
        "advanced_m": round(advanced, 3),
    }
    if not adv_reached and advanced < 0.03:
        record["travel_m"] = 0.0
        return obs
    obs = push_segments(
        env,
        obs,
        into,
        float(spot[2]),
        rot,
        1.0,
        events,
        mode,
        record,
        budget=TRAVEL_M["prismatic"] + OPEN_DRAWER_SLACK_M,
    )
    return _retreat(env, obs, out, rot, base_rot, events, mode)


STANDOFF_M = 0.12


def _attempt_pitched_family(
    env,
    obs,
    kind,
    mode,
    target,
    handle,
    out,
    into,
    pitch,
    bar_vertical,
    base_rot,
    support_z,
    events,
    fam,
):
    """One approach family with the fingers pointing along `into`, pitched down by `pitch`: standoff -> (close:
    push the face beside the bar | open: grasp the bar, pull along the mechanism, and for a door continue from
    inside once the pull stalls).  Returns (obs, ok); on failure the hand has retreated.
    """
    from depthrobotic.skills.articulate.door import finish_door_open

    into3 = np.r_[into * np.cos(pitch), -np.sin(pitch)] if pitch > 0 else into
    rot = approach_rotation(base_rot, into3, bar_vertical)
    fam["pitch_deg"] = round(float(np.degrees(pitch)), 1)
    standoff_range = STANDOFF_M + (
        OPEN_DRAWER_SLACK_M if mode == "close" and kind == "prismatic" else 0.0
    )
    standoff = np.r_[
        handle[:2] + out * standoff_range, handle[2] + standoff_range * np.tan(pitch)
    ]
    obs, reached, res_m, res_rad, _ = pose_servo(
        env,
        obs,
        standoff,
        rot,
        -1.0,
        events,
        f"handle_standoff:{mode}",
        max_steps=45,
        stop_on_no_progress=True,
    )
    fam["standoff"] = {
        "reached": reached,
        "residual_m": round(res_m, 3),
        "residual_rad": round(res_rad, 3),
    }
    if not reached and 0.05 < res_m <= 0.15:
        home = np.r_[
            np.asarray(env.home_xy, dtype=float), support_z + SURVEY_EEF_Z_M + 0.05
        ]
        obs, *_ = pose_servo(
            env,
            obs,
            home,
            base_rot,
            -1.0,
            events,
            f"handle_home_reset:{mode}",
            max_steps=40,
        )
        obs, reached, res_m, res_rad, _ = pose_servo(
            env,
            obs,
            standoff,
            rot,
            -1.0,
            events,
            f"handle_standoff_retry:{mode}",
            max_steps=60,
            stop_on_no_progress=True,
        )
        fam["standoff_retry"] = {
            "reached": reached,
            "residual_m": round(res_m, 3),
            "residual_rad": round(res_rad, 3),
        }
    if not reached and (res_m > ABORT_STANDOFF_RESIDUAL_M or res_rad > 0.2):
        fam["abort"] = "standoff_unreachable"
        cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
        obs, *_ = pose_servo(
            env,
            obs,
            np.r_[cur[:2], max(cur[2], support_z + SURVEY_EEF_Z_M) + 0.05],
            rot,
            -1.0,
            events,
            f"handle_lift_next:{mode}",
            max_steps=10,
        )
        return (obs, False)
    if mode == "close" and kind == "prismatic":
        obs = push_from_standoff(
            env, obs, handle, out, into, rot, base_rot, events, mode, fam
        )
        adv = fam.get("advance", {})
        ok = (
            bool(adv.get("reached"))
            or float(adv.get("advanced_m") or 0.0) >= 0.03
            or float(fam.get("travel_m") or 0.0) > 0.02
        )
        if not ok:
            fam["abort"] = "no_drawer_front_met"
        return (obs, ok)
    ok = False
    try:
        direction = out.copy()
        target_travel = TRAVEL_M[kind]
        grip_point = np.r_[
            handle[:2] + into * GRASP_OVERSHOOT_M,
            handle[2] - GRASP_OVERSHOOT_M * np.tan(pitch),
        ]
        attempts, grip, blocked = ([], "empty", 0)
        for attempt in range(3):
            tgt = np.r_[
                grip_point[:2] + into * (GRASP_ESCALATION_M * attempt),
                grip_point[2] - GRASP_ESCALATION_M * attempt * np.tan(pitch),
            ]
            obs, adv_reached, adv_res, adv_rad, _ = pose_servo(
                env,
                obs,
                tgt,
                rot,
                -1.0,
                events,
                f"handle_advance:{mode}:{attempt}",
                max_steps=50,
                stop_on_stall=True,
            )
            if adv_rad > 0.06:
                cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
                obs, *_ = pose_servo(
                    env,
                    obs,
                    cur,
                    rot,
                    -1.0,
                    events,
                    f"handle_settle:{mode}:{attempt}",
                    max_steps=12,
                )
            obs = hold(env, obs, 1.0, 12, events, f"handle_grip:{mode}:{attempt}")
            gq = obs["robot0_gripper_qpos"]
            gap = abs(float(gq[0]) - float(gq[1]))
            grip = (
                "bar"
                if MIN_GRIP_GAP_M <= gap <= MAX_GRIP_GAP_M
                else "panel" if gap > MAX_GRIP_GAP_M else "empty"
            )
            attempts.append(
                {
                    "advance": {
                        "reached": adv_reached,
                        "residual_m": round(adv_res, 3),
                    },
                    "grip_gap_m": round(gap, 4),
                    "grip": grip,
                }
            )
            if grip != "empty":
                break
            if not adv_reached and adv_res > 0.04:
                blocked += 1
            obs = hold(
                env, obs, -1.0, 6, events, f"handle_regrip_open:{mode}:{attempt}"
            )
        fam.update(
            {
                "advance": attempts[-1]["advance"],
                "grip_gap_m": attempts[-1]["grip_gap_m"],
                "grip": grip,
                "grip_attempts": len(attempts),
                "attempts": attempts,
            }
        )
        if grip == "empty":
            fam["abort"] = "advance_blocked" if blocked >= 2 else "grip_empty"
            return (obs, False)
        start = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
        last, stalls, travel, path, pts = (start.copy(), 0, 0.0, [], [start[:2].copy()])
        while travel < target_travel - 0.01 and stalls < 2:
            cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
            goal = np.r_[cur[:2] + direction * SEGMENT_M, grip_point[2]]
            obs, _, _, seg_rad, _ = pose_servo(
                env,
                obs,
                goal,
                rot,
                1.0,
                events,
                f"handle_{mode}:{int(travel * 100)}",
                max_steps=25 if kind == "revolute" else 30,
                stop_on_stall=True,
            )
            now = np.asarray(obs["robot0_eef_pos"], dtype=float)
            moved_seg = now[:2] - last[:2]
            path.append(moved_seg.round(3).tolist())
            gq_ = obs["robot0_gripper_qpos"]
            if abs(float(gq_[0]) - float(gq_[1])) < MIN_GRIP_GAP_M and (
                not fam.get("regripped")
            ):
                fam["regripped"] = {"at_travel_m": round(travel, 3)}
                obs = hold(env, obs, -1.0, 6, events, f"handle_regrip_open:{mode}:pull")
                back_to = np.r_[now[:2] + into * 0.03, grip_point[2]]
                obs, *_ = pose_servo(
                    env,
                    obs,
                    back_to,
                    rot,
                    -1.0,
                    events,
                    f"handle_regrip_advance:{mode}",
                    max_steps=25,
                    stop_on_stall=True,
                )
                obs = hold(env, obs, 1.0, 12, events, f"handle_regrip:{mode}")
                gq_ = obs["robot0_gripper_qpos"]
                if abs(float(gq_[0]) - float(gq_[1])) < MIN_GRIP_GAP_M:
                    stalls = 2
                    last = now
                    continue
            if seg_rad < 0.1:
                pts.append(now[:2].copy())
            if kind == "revolute" and float(np.linalg.norm(moved_seg)) > 0.02:
                direction = moved_seg / float(np.linalg.norm(moved_seg))
            progress = float(np.linalg.norm((now - start)[:2]))
            if progress - travel < (0.025 if kind == "revolute" else 0.01) or (
                kind == "revolute" and seg_rad > 0.15
            ):
                stalls += 1
            travel = progress
            last = now
        swing = float(
            np.arctan2(
                out[0] * direction[1] - out[1] * direction[0],
                float(np.dot(out, direction)),
            )
        )
        fam.update(
            {
                "travel_m": round(travel, 3),
                "stalled": stalls >= 2,
                "path_segments": path,
                "swing_deg": round(float(np.degrees(swing)), 1),
            }
        )
        ok = travel >= 0.03
        obs = hold(env, obs, -1.0, 8, events, f"handle_release:{mode}")
        if kind == "revolute" and len(pts) >= 3:
            cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
            obs, *_ = pose_servo(
                env,
                obs,
                np.r_[cur[:2] + out * 0.05, cur[2] + 0.15],
                rot,
                -1.0,
                events,
                f"handle_retreat:{mode}",
                max_steps=20,
            )
            obs = finish_door_open(
                env,
                obs,
                pts,
                start[:2],
                out,
                float(handle[2]),
                base_rot,
                events,
                mode,
                fam,
            )
            return (obs, True)
    finally:
        pass
    obs = hold(env, obs, -1.0, 6, events, f"handle_release2:{mode}")
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2] + out * 0.1, cur[2]],
        rot,
        -1.0,
        events,
        f"handle_retreat:{mode}",
        max_steps=30,
    )
    cur = np.asarray(obs["robot0_eef_pos"], dtype=float)
    obs, *_ = pose_servo(
        env,
        obs,
        np.r_[cur[:2], cur[2] + 0.1],
        base_rot,
        -1.0,
        events,
        f"handle_restore:{mode}",
        max_steps=45,
    )
    return (obs, ok)


def execute_handle_articulation(
    env,
    obs,
    grounder,
    instruction,
    step,
    size,
    support_z,
    events,
    perception_log,
    *,
    context,
):
    from depthrobotic.skills.articulate.door import close_door_by_arc
    from depthrobotic.skills.articulate.door import measure_door_front
    from depthrobotic.skills.articulate.door import robot_facing

    target, mode = (step["target"], step["mode"])
    kind = (
        "revolute"
        if any((w in target.lower() for w in REVOLUTE_WORDS))
        else "prismatic"
    )
    record: dict[str, Any] = {
        "action": "articulate_primitive",
        "kind": kind,
        "target": target,
        "mode": mode,
    }
    perception_log.append(record)
    g = ground_handle(
        env, obs, grounder, target, instruction, size, support_z, context=context
    )
    record.update(g["rec"])
    handle = np.asarray(g["world"], dtype=float)
    record["handle_front_world"] = handle.round(3).tolist()
    if g.get("out") is not None:
        out = _snap(g["out"])
        record["out_source"] = "depth_normal"
    else:
        out = _snap(np.asarray(env.home_xy, dtype=float) - handle[:2])
        record["out_source"] = "home_direction"
    out = robot_facing(env, out, handle[:2])
    bar_vertical = kind == "revolute"
    into = -out
    record.update({"out": out.tolist(), "bar_vertical": bar_vertical})
    base_rot = Rotation.from_quat(obs["robot0_eef_quat"])
    rot = approach_rotation(base_rot, into, bar_vertical)
    if kind == "revolute" and mode == "close":
        geom, drec = measure_door_front(
            env,
            obs,
            grounder,
            record.get("furniture", target),
            instruction,
            size,
            support_z,
            furniture_box=record.get("furniture_box"),
            context=context,
        )
        record["door"] = drec
        if geom is None:
            record["abort"] = "no_door_geometry"
            return obs
        lift = np.r_[
            np.asarray(obs["robot0_eef_pos"], dtype=float)[:2],
            max(float(obs["robot0_eef_pos"][2]), geom["z_top"] + 0.15),
        ]
        obs, *_ = pose_servo(
            env, obs, lift, base_rot, -1.0, events, f"handle_lift:{mode}", max_steps=25
        )
        return close_door_by_arc(
            env,
            obs,
            geom["free"],
            geom["hinge"],
            geom["n_out"],
            base_rot,
            events,
            mode,
            record,
            geom=geom,
        )
    lift = np.r_[
        np.asarray(obs["robot0_eef_pos"], dtype=float)[:2],
        max(float(obs["robot0_eef_pos"][2]), support_z + SURVEY_EEF_Z_M + 0.05),
    ]
    obs, *_ = pose_servo(
        env, obs, lift, base_rot, -1.0, events, f"handle_lift:{mode}", max_steps=25
    )
    surveyed = None
    if g.get("front_rays"):
        obs, surveyed = survey_triangulate(
            env,
            obs,
            grounder,
            target,
            record.get("furniture", target),
            size,
            support_z,
            g["front_rays"],
            g.get("chosen", 0),
            out,
            rot,
            events,
            mode,
            record,
            context=context,
        )
    if surveyed is not None:
        handle = np.asarray(surveyed, dtype=float)
        record["handle_survey_world"] = handle.round(3).tolist()
        record["handle_survey_source"] = record["survey"].get("source")
    if record.get("out_measured") is not None and (
        not np.allclose(record["out_measured"], out)
    ):
        out = np.asarray(record["out_measured"], dtype=float)
        into = -out
        rot = approach_rotation(base_rot, into, bar_vertical)
        record["out_replaced_by_measurement"] = out.tolist()
    record["handle_refined_source"] = (
        record["survey"].get("source") if surveyed is not None else "front_depth"
    )
    if kind == "revolute":
        families = [("level", 0.0)]
    else:
        families = [("level", 0.0), ("pitched", np.radians(PITCH_DOWN_DEG))] + (
            [("topdown", None)] if mode == "close" else []
        )
    record["families"] = []
    for name, pitch in families:
        fam: dict[str, Any] = {"family": name}
        record["families"].append(fam)
        if name == "topdown":
            obs = push_topdown(
                env, obs, handle, out, into, base_rot, events, mode, fam, support_z
            )
            ok = (
                bool(fam.get("advance", {}).get("reached")) and fam.get("abort") is None
            )
        else:
            obs, ok = _attempt_pitched_family(
                env,
                obs,
                kind,
                mode,
                target,
                handle,
                out,
                into,
                pitch,
                bar_vertical,
                base_rot,
                support_z,
                events,
                fam,
            )
        fam["ok"] = ok
        if ok:
            break
    last = record["families"][-1]
    for k in (
        "standoff",
        "advance",
        "grip_gap_m",
        "grip",
        "grip_attempts",
        "travel_m",
        "stalled",
        "path_segments",
        "stages",
        "push_spot",
        "push_variant",
        "descend_attempts",
        "push_phases",
        "abort",
        "finish",
    ):
        if k in last:
            record[k] = last[k]
    if not last.get("ok"):
        record["abort"] = last.get("abort") or "all_families_failed"
    return obs


PRISMATIC_WORDS = ("drawer",)


def execute_articulate(
    env,
    obs,
    grounder,
    instruction,
    step,
    size,
    support_z,
    events,
    perception_log,
    *,
    context,
):
    t = str(step.get("target", "")).lower()
    if any((w in t for w in PRISMATIC_WORDS + REVOLUTE_WORDS)):
        return execute_handle_articulation(
            env,
            obs,
            grounder,
            instruction,
            step,
            size,
            support_z,
            events,
            perception_log,
            context=context,
        )
    return execute_contact_articulation(
        env,
        obs,
        grounder,
        instruction,
        step,
        size,
        support_z,
        events,
        perception_log,
        context=context,
    )
