"""geometry / depth: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.geometry.camera import deproject, intersect_horizontal_plane
from math import pi
from math import tan
import numpy as np


def depth_scale_to_workplane(
    depth, camera_to_world, focal_px, size, support_z, *, context
):
    """Robustly align one predicted depth frame to a fixed robot work plane.

    The lower/central image is mostly tabletop in standard tabletop views. We
    compare its predicted camera-z values with ray/work-plane intersections and
    take a robust low quantile so objects above the table do not dominate.
    """
    height, width = depth.shape
    ratios = []
    world_to_camera = np.linalg.inv(camera_to_world)
    for y in np.linspace(420, 900, 13):
        for x in np.linspace(100, 900, 17):
            row = int(np.clip(round(y / 1000 * (height - 1)), 0, height - 1))
            col = int(np.clip(round(x / 1000 * (width - 1)), 0, width - 1))
            predicted = float(depth[row, col])
            if not np.isfinite(predicted) or predicted <= 0:
                continue
            try:
                world = intersect_horizontal_plane(
                    [x, y], support_z, camera_to_world, focal_px, size, context=context
                )
            except RuntimeError:
                continue
            expected = float((world_to_camera @ np.r_[world, 1.0])[2])
            if expected > 0:
                ratios.append(expected / predicted)
    if len(ratios) < 20:
        raise RuntimeError("Too few valid depth/work-plane calibration rays")
    scale = float(np.percentile(ratios, 35))
    if not np.isfinite(scale) or not 0.1 <= scale <= 5.0:
        raise RuntimeError(f"Implausible depth scale {scale}")
    return scale


def frame_geometry(env, grounder, image, size, support_z, *, context):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

    camera_id = env.sim.model.camera_name2id("agentview")
    fovy = float(env.sim.model.cam_fovy[camera_id])
    focal_px = size / (2 * tan(fovy * pi / 360))
    depth = grounder.predict_depth(image, focal_px)
    camera_to_world = get_camera_extrinsic_matrix(env.sim, "agentview")
    scale = depth_scale_to_workplane(
        depth, camera_to_world, focal_px, size, support_z, context=context
    )
    return (depth, camera_to_world, focal_px, support_z, scale)


def sample_point_depth(depth, point, radius=5):
    height, width = depth.shape
    x = int(round(float(point[0]) / 1000 * (width - 1)))
    y = int(round(float(point[1]) / 1000 * (height - 1)))
    values = depth[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        raise RuntimeError(f"No valid depth around point {point}")
    return float(np.median(values))


def point_world(depth, point, scale, camera_to_world, focal_px, size, *, context):
    z_camera = sample_point_depth(depth, point, radius=4) * scale
    world = deproject(point, z_camera, camera_to_world, focal_px, size, context=context)
    if not np.isfinite(world).all():
        raise RuntimeError(f"Non-finite 3D point for {point}")
    return world


def receiving_plane(surface_z, support_z, is_container):
    return surface_z if is_container or surface_z > support_z + 0.06 else support_z
