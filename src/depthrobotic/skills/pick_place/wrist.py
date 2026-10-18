"""skills / pick_place / wrist: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import move_to
from depthrobotic.geometry.camera import (
    camera_geometry,
    deproject,
    intersect_horizontal_plane,
    project_world_to_raw,
)
from depthrobotic.geometry.depth import depth_scale_to_workplane, sample_point_depth
from depthrobotic.perception.images import (
    crop_from_normalized,
    crop_point_to_global,
    observation_image,
)
from depthrobotic.perception.parsing import (
    normalize_box,
    normalize_point,
    safe_ask_structured,
)
from depthrobotic.prompts import wrist_center_prompt
from depthrobotic.skills.pick_place.locate import (
    FINGER_AXIS,
    WRIST_SERVO_ITERS,
    ambiguous_source_description,
)
import numpy as np


def measure_object_box(grounder, obs, phrase, instruction, *, context):
    image = observation_image(obs, "robot0_eye_in_hand_image", context=context)
    prompt = f'Robot instruction: {instruction}\nThis is the wrist camera looking down at the source object {phrase}. Return the tight 2D bounding box of the whole visible object (its full outer extent; ignore the gripper fingers and other objects). JSON only, normalized 0..1000: {{"bbox_2d":[x1,y1,x2,y2]}}.'

    def plausible(b):
        return b is not None and 20 < b[2] - b[0] < 700 and (20 < b[3] - b[1] < 700)

    response, value = safe_ask_structured(grounder, image, prompt, 96)
    box = value.get("bbox_2d") or value.get("bbox") or value.get("box")
    box = normalize_box(box) if box is not None else None
    if not plausible(box):
        strict = (
            prompt
            + " The object occupies only part of the image: the box must be much smaller than the image and must not touch the image borders."
        )
        response, value = safe_ask_structured(grounder, image, strict, 96)
        box = value.get("bbox_2d") or value.get("bbox") or value.get("box")
        box = normalize_box(box) if box is not None else None
    return (box if plausible(box) else None, response)


def wrist_servo(
    env,
    obs,
    grounder,
    phrase,
    instruction,
    role,
    anchor_world,
    plane_z,
    travel_z,
    gripper,
    size,
    events,
    log,
    iterations=WRIST_SERVO_ITERS,
    max_anchor_deviation_override=None,
    force_full_image=False,
    *,
    context,
):
    identity_anchor = np.asarray(anchor_world, dtype=float).copy()
    estimate_world = identity_anchor.copy()
    ambiguous_source = role == "source" and ambiguous_source_description(
        phrase, instruction
    )
    max_anchor_deviation = (
        max_anchor_deviation_override
        if max_anchor_deviation_override is not None
        else 0.07 if ambiguous_source else 0.18 if role == "source" else 0.04
    )
    for iteration in range(iterations):
        full_image = observation_image(obs, "robot0_eye_in_hand_image", context=context)
        expected_point = project_world_to_raw(
            env, "robot0_eye_in_hand", estimate_world, size, context=context
        )
        use_crop = not force_full_image and bool(
            np.all(
                (np.asarray(expected_point) >= 0) & (np.asarray(expected_point) <= 1000)
            )
        )
        if use_crop:
            image, crop_bounds = crop_from_normalized(full_image, expected_point, 0.55)
        else:
            image, crop_bounds = (full_image, None)
        response, value = safe_ask_structured(
            grounder, image, wrist_center_prompt(phrase, instruction, role), 96
        )
        local_point = normalize_point(value.get("point"))
        point = local_point
        if local_point is not None and crop_bounds is not None:
            point = crop_point_to_global(local_point, crop_bounds, full_image.size)
        entry = {
            "iteration": iteration,
            "role": role,
            "response": response,
            "expected_point": expected_point,
            "crop_bounds": None if crop_bounds is None else list(crop_bounds),
            "point": point,
        }
        log.append(entry)
        if point is None:
            break
        camera_to_world, focal_px, fovy = camera_geometry(
            env, "robot0_eye_in_hand", size
        )
        world = intersect_horizontal_plane(
            point, plane_z, camera_to_world, focal_px, size, context=context
        )
        entry.update({"world": world.tolist(), "fovy": fovy})
        current_xy = np.asarray(obs["robot0_eef_pos"], dtype=float)[:2]
        correction = float(np.linalg.norm(world[:2] - current_xy))
        anchor_deviation = float(np.linalg.norm(world[:2] - identity_anchor[:2]))
        entry["correction_m"] = correction
        entry["anchor_deviation_m"] = anchor_deviation
        if anchor_deviation > max_anchor_deviation:
            entry["rejected_identity_jump"] = True
            break
        estimate_world = world
        if correction < 0.006:
            break
        obs = move_to(
            env,
            obs,
            np.r_[world[:2], travel_z],
            gripper,
            events,
            f"wrist_servo:{role}:{iteration}",
            max_steps=40,
        )
    final_point = project_world_to_raw(
        env, "robot0_eye_in_hand", estimate_world, size, context=context
    )
    log.append({"role": role, "final_reprojected_point": final_point})
    return (obs, final_point)


def box_metrics(env, size, box, plane_z, *, context):
    """Metric extent of a wrist-image box on the horizontal plane plane_z.
    Returns (width_along_finger_axis, width_across, center_world_xy)."""
    cam2world, focal, _ = camera_geometry(env, "robot0_eye_in_hand", size)
    x1, y1, x2, y2 = box
    xc, yc = ((x1 + x2) / 2, (y1 + y2) / 2)
    p1 = intersect_horizontal_plane(
        [x1, yc], plane_z, cam2world, focal, size, context=context
    )
    p2 = intersect_horizontal_plane(
        [x2, yc], plane_z, cam2world, focal, size, context=context
    )
    q1 = intersect_horizontal_plane(
        [xc, y1], plane_z, cam2world, focal, size, context=context
    )
    q2 = intersect_horizontal_plane(
        [xc, y2], plane_z, cam2world, focal, size, context=context
    )
    width_finger = abs(float(np.dot((p2 - p1)[:2], FINGER_AXIS)))
    width_across = float(np.linalg.norm((q2 - q1)[:2]))
    center = intersect_horizontal_plane(
        [xc, yc], plane_z, cam2world, focal, size, context=context
    )[:2]
    return (width_finger, width_across, center)


def confirm_grasp(grounder, obs, source, instruction, *, context):
    gap = float(np.mean(np.abs(obs["robot0_gripper_qpos"])))
    image = observation_image(obs, "robot0_eye_in_hand_image", context=context)
    response, value = safe_ask_structured(
        grounder,
        image,
        f'Robot instruction: {instruction}\nThis is a close wrist-camera view after lifting. Is the exact object {source} visibly held between the gripper fingers? Return JSON only: {{"held":true}}.',
        64,
    )
    visible = value.get("held") is True
    minimum_gap = 0.002
    confirmed = gap > minimum_gap and visible
    return (
        confirmed,
        {
            "grip_gap": gap,
            "minimum_gap": minimum_gap,
            "visible_held": visible,
            "confirmation_camera": "robot0_eye_in_hand_image",
            "response": response,
        },
    )


def estimate_held_offset_3d(
    env, obs, grounder, source, instruction, size, support_z, *, context
):
    """Held-object center relative to the eef, from the model's point in the wrist image and DepthVLM's
    metric wrist depth (scale-calibrated on the visible table). Returns (offset_xyz or None, record).
    """
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    image = observation_image(obs, "robot0_eye_in_hand_image", context=context)
    prompt = f'Robot instruction: {instruction}\nThis is the wrist camera after lifting. Point to the visual center of the held object {source}: the object between or hanging below the gripper fingers, not the fingers themselves and not the background. JSON only, normalized 0..1000: {{"point":[x,y]}}.'
    response, value = safe_ask_structured(grounder, image, prompt, 96)
    point = normalize_point(value.get("point"))
    record = {"held_point": point, "held_response": response[:120]}
    if point is None:
        return (None, record)
    cam2world, focal, _ = camera_geometry(env, "robot0_eye_in_hand", size)
    eef_px = project_world_to_raw(env, "robot0_eye_in_hand", eef, size, context=context)
    record["eef_px"] = eef_px
    if float(np.linalg.norm(np.asarray(point) - np.asarray(eef_px))) > 450:
        record["held_point_rejected"] = "too far from the gripper in the image"
        return (None, record)
    try:
        depth = grounder.predict_depth(image, focal)
        scale = depth_scale_to_workplane(
            depth, cam2world, focal, size, support_z, context=context
        )
        z_cam = sample_point_depth(depth, point, radius=6) * scale
        obj = deproject(point, z_cam, cam2world, focal, size, context=context)
        offset = obj - eef
        record.update(
            {"depth_scale": scale, "z_cam": z_cam, "held_offset_xyz": offset.tolist()}
        )
        drop = float(-offset[2])
        if float(np.linalg.norm(offset[:2])) > 0.08 or not -0.02 <= drop <= 0.16:
            record["held_offset_rejected"] = "implausible"
            return (None, record)
        return (offset, record)
    except Exception as e:
        record["depth_error"] = f"{type(e).__name__}: {e}"[:120]
        return (None, record)


def current_observation(env):
    candidate = env
    for _ in range(4):
        getter = getattr(candidate, "_get_observations", None)
        if getter is not None:
            return getter()
        nested = getattr(candidate, "env", None)
        if nested is None or nested is candidate:
            break
        candidate = nested
    raise RuntimeError("Environment cannot return a current observation")
