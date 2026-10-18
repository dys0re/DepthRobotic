"""skills / simple: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import hold, move_contact, move_to, rotate_wrist
from depthrobotic.geometry.camera import (
    deproject,
    image_motion_world,
    intersect_horizontal_plane,
)
from depthrobotic.geometry.depth import frame_geometry, point_world, sample_point_depth
from depthrobotic.perception.images import raw_image
from depthrobotic.perception.parsing import (
    normalize_point,
    safe_ask_structured,
    valid_vector,
)
from depthrobotic.prompts import articulation_prompt, push_prompt, toggle_prompt
from depthrobotic.skills.pick_place.wrist import current_observation
import numpy as np


def execute_push_object(
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
    source, destination = (step["source"], step["destination"])
    image = raw_image(obs, context=context)
    response, value = safe_ask_structured(
        grounder, image, push_prompt(source, destination, instruction), 128
    )
    contact = normalize_point(value.get("contact"))
    destination_point = normalize_point(value.get("destination"))
    if contact is None or destination_point is None:
        raise RuntimeError(f"Invalid push grounding: {response}")
    _, camera_to_world, focal_px, _, depth_scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    push_z = support_z + 0.04
    contact_world = intersect_horizontal_plane(
        contact, push_z, camera_to_world, focal_px, size, context=context
    )
    semantic_destination = intersect_horizontal_plane(
        destination_point, push_z, camera_to_world, focal_px, size, context=context
    )
    push_delta = semantic_destination[:2] - contact_world[:2]
    push_norm = float(np.linalg.norm(push_delta))
    if push_norm < 1e-05:
        raise RuntimeError("Degenerate push direction")
    push_direction = push_delta / push_norm
    contact_start = contact_world.copy()
    contact_start[:2] -= push_direction * 0.045
    endpoint = contact_world.copy()
    endpoint[:2] += push_direction * 0.12
    perception_log.append(
        {
            "action": "push_object_closed_loop",
            "source": source,
            "destination": destination,
            "response": response,
            "contact": contact,
            "destination_point": destination_point,
            "contact_world": contact_world.tolist(),
            "rear_edge_contact_world": contact_start.tolist(),
            "semantic_destination_world": semantic_destination.tolist(),
            "limited_endpoint_world": endpoint.tolist(),
            "push_z": push_z,
            "depth_scale": depth_scale,
        }
    )
    obs = move_to(
        env,
        obs,
        np.r_[contact_start[:2], push_z + 0.14],
        1.0,
        events,
        "prepush",
        max_steps=55,
    )
    try:
        obs = move_contact(
            env, obs, contact_start, 1.0, events, "push_contact", max_steps=55
        )
    except RuntimeError as error:
        obs = current_observation(env)
        xy_residual = float(
            np.linalg.norm(contact_start[:2] - np.asarray(obs["robot0_eef_pos"])[:2])
        )
        if xy_residual > 0.03:
            raise
        perception_log[-1]["accepted_tool_height_contact_error"] = str(error)
    endpoint_eef = endpoint.copy()
    endpoint_eef[2] = float(np.asarray(obs["robot0_eef_pos"])[2])
    obs = move_contact(env, obs, endpoint_eef, 1.0, events, "push", max_steps=65)
    try:
        obs = move_to(
            env,
            obs,
            np.r_[endpoint[:2], push_z + 0.14],
            1.0,
            events,
            "push_retreat",
            max_steps=35,
        )
    except RuntimeError:
        obs = current_observation(env)
    return obs


def toggle_contact_motion(
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
    target_name, mode = (step["target"], step["mode"])
    image = raw_image(obs, context=context)
    response, value = safe_ask_structured(
        grounder, image, toggle_prompt(target_name, mode, instruction), 128
    )
    point = normalize_point(value.get("point"))
    if point is None:
        raise RuntimeError(f"Invalid toggle grounding: {response}")
    depth, camera_to_world, focal_px, _, depth_scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    depth_world = point_world(
        depth, point, depth_scale, camera_to_world, focal_px, size, context=context
    )
    contact_z = float(np.clip(depth_world[2], support_z - 0.04, support_z + 0.18))
    contact = intersect_horizontal_plane(
        point, contact_z, camera_to_world, focal_px, size, context=context
    )
    perception_log.append(
        {
            "action": "toggle",
            "target": target_name,
            "mode": mode,
            "response": response,
            "point": point,
            "contact_world": contact.tolist(),
            "depth_scale": depth_scale,
        }
    )
    obs = move_to(
        env, obs, np.r_[contact[:2], contact_z + 0.1], -1.0, events, "pretoggle"
    )
    obs = move_contact(env, obs, contact, -1.0, events, "toggle_contact")
    obs = hold(env, obs, 1.0, 12, events, "toggle_grip")
    obs = rotate_wrist(
        env, obs, 1.0 if mode == "on" else -1.0, 35, events, "toggle_rotate"
    )
    obs = hold(env, obs, -1.0, 8, events, "toggle_release")
    obs = move_to(
        env, obs, np.r_[contact[:2], contact_z + 0.1], -1.0, events, "toggle_retreat"
    )
    return obs


def execute_toggle(*args, context, **kwargs):
    env = args[0]
    perception_log = args[8]
    try:
        return toggle_contact_motion(*args, **kwargs, context=context)
    except RuntimeError as error:
        if "toggle_retreat" not in str(error):
            raise
        if perception_log:
            perception_log[-1]["nonfatal_retreat_error"] = str(error)
        return current_observation(env)


def execute_contact_articulation(
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
    target_name, mode = (step["target"], step["mode"])
    image = raw_image(obs, context=context)
    response, value = safe_ask_structured(
        grounder, image, articulation_prompt(target_name, mode, instruction), 128
    )
    contact = normalize_point(value.get("contact"))
    motion = value.get("motion")
    if contact is None:
        raise RuntimeError(f"Invalid articulation grounding: {response}")
    depth, camera_to_world, focal_px, _, depth_scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    z_camera = sample_point_depth(depth, contact, radius=5) * depth_scale
    depth_world = deproject(
        contact, z_camera, camera_to_world, focal_px, size, context=context
    )
    contact_z = float(np.clip(depth_world[2], support_z - 0.08, support_z + 0.34))
    world = intersect_horizontal_plane(
        contact, contact_z, camera_to_world, focal_px, size, context=context
    )
    home_direction = np.asarray(env.home_xy, dtype=float) - world[:2]
    home_norm = float(np.linalg.norm(home_direction))
    if home_norm < 1e-05:
        raise RuntimeError("Cannot infer front-side articulation approach")
    home_direction /= home_norm
    if valid_vector(motion):
        delta = image_motion_world(
            contact,
            motion,
            z_camera,
            camera_to_world,
            focal_px,
            size,
            distance=0.18,
            context=context,
        )
        motion_source = "depthvlm_image_direction"
    else:
        signed = home_direction if mode == "open" else -home_direction
        delta = np.r_[signed * 0.18, 0.0]
        motion_source = "generic_toward_or_away_from_robot"
    endpoint = world + delta
    approach = world.copy()
    approach[:2] += home_direction * 0.1
    approach[2] += 0.02
    record = {
        "action": "articulate_closed_loop",
        "target": target_name,
        "mode": mode,
        "response": response,
        "contact": contact,
        "motion": motion,
        "motion_source": motion_source,
        "contact_world": world.tolist(),
        "approach_world": approach.tolist(),
        "endpoint_world": endpoint.tolist(),
        "depth_scale": depth_scale,
    }
    perception_log.append(record)
    try:
        obs = move_to(
            env, obs, approach, -1.0, events, f"precontact:{mode}", max_steps=60
        )
    except RuntimeError as error:
        obs = current_observation(env)
        remaining = approach - np.asarray(obs["robot0_eef_pos"], dtype=float)
        if float(np.linalg.norm(remaining)) > 0.05:
            raise
        record["accepted_approach_residual_m"] = float(np.linalg.norm(remaining))
        record["nonfatal_approach_error"] = str(error)
    events.append(
        {
            "phase": f"contact:{mode}",
            "steps": 0,
            "reached": True,
            "tool_offset_contact": True,
            "eef": obs["robot0_eef_pos"].tolist(),
        }
    )
    obs = hold(env, obs, 1.0, 14, events, f"grip:{mode}")
    endpoint_eef = approach + delta
    obs = move_to(
        env, obs, endpoint_eef, 1.0, events, f"articulate:{mode}", max_steps=55
    )
    obs = hold(env, obs, -1.0, 8, events, f"release:{mode}")
    try:
        obs = move_to(
            env,
            obs,
            np.r_[endpoint_eef[:2] + home_direction * 0.08, endpoint_eef[2] + 0.04],
            -1.0,
            events,
            f"retreat:{mode}",
            max_steps=35,
        )
    except RuntimeError as error:
        record["nonfatal_retreat_error"] = str(error)
        obs = current_observation(env)
    return obs
