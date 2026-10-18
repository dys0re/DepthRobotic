"""geometry / camera: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from math import pi
from math import tan
import numpy as np


def camera_geometry(env, camera, size):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

    camera_id = env.sim.model.camera_name2id(camera)
    fovy = float(env.sim.model.cam_fovy[camera_id])
    focal_px = size / (2 * tan(fovy * pi / 360))
    extrinsic = get_camera_extrinsic_matrix(env.sim, camera)
    return (extrinsic, focal_px, fovy)


def raw_ray(point, focal_px, size, *, context):
    col = float(point[0]) / 1000 * (size - 1)
    raw_row = float(point[1]) / 1000 * (size - 1)
    row = raw_row if context.settings.orient_fix else size - 1 - raw_row
    return np.array([(col - size / 2) / focal_px, (row - size / 2) / focal_px, 1.0])


def intersect_horizontal_plane(
    point, plane_z, camera_to_world, focal_px, size, *, context
):
    origin = camera_to_world[:3, 3]
    direction = camera_to_world[:3, :3] @ raw_ray(
        point, focal_px, size, context=context
    )
    if abs(float(direction[2])) < 1e-08:
        raise RuntimeError("Grounding ray is parallel to the work plane")
    distance = (plane_z - origin[2]) / direction[2]
    if distance <= 0:
        raise RuntimeError(
            "Grounding ray does not intersect the work plane in front of the camera"
        )
    return origin + direction * distance


def project_world_to_raw(env, camera, world, size, *, context):
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    transform = get_camera_transform_matrix(env.sim, camera, size, size)
    homogeneous = transform @ np.r_[np.asarray(world, dtype=float)[:3], 1.0]
    if abs(float(homogeneous[2])) < 1e-08:
        raise RuntimeError("World anchor projects at zero camera depth")
    col, row = homogeneous[:2] / homogeneous[2]
    return [
        float(col / (size - 1) * 1000.0),
        float(
            (row if context.settings.orient_fix else size - 1 - row)
            / (size - 1)
            * 1000.0
        ),
    ]


def deproject(point, z_camera, camera_to_world, focal_px, size, *, context):
    camera_point = np.r_[
        raw_ray(point, focal_px, size, context=context) * z_camera, 1.0
    ]
    return (camera_to_world @ camera_point)[:3]


def image_motion_world(
    contact, motion, z_camera, camera_to_world, focal_px, size, distance, *, context
):
    start = deproject(
        contact, z_camera, camera_to_world, focal_px, size, context=context
    )
    pixel_end = [
        float(np.clip(contact[0] + 180.0 * motion[0], 0.0, 1000.0)),
        float(np.clip(contact[1] + 180.0 * motion[1], 0.0, 1000.0)),
    ]
    end = deproject(
        pixel_end, z_camera, camera_to_world, focal_px, size, context=context
    )
    delta = end - start
    delta[2] = 0.0
    norm = float(np.linalg.norm(delta))
    if norm < 1e-05:
        raise RuntimeError(f"Degenerate image motion {motion}")
    return delta / norm * distance
