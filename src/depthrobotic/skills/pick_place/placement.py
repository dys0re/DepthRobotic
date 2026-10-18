"""skills / pick_place / placement: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import (
    PRECISE_TOL,
    WORKSPACE_X,
    WORKSPACE_Y,
    descend_gentle_contact,
    descend_hold_xy,
    hold,
    in_workspace,
    move_to,
    move_to_precise,
    probe_top_height,
)
from depthrobotic.geometry.camera import (
    camera_geometry,
    intersect_horizontal_plane,
    project_world_to_raw,
)
from depthrobotic.geometry.depth import frame_geometry, point_world, receiving_plane
from depthrobotic.perception.images import (
    crop_from_normalized,
    crop_point_to_global,
    observation_image,
    upright,
)
from depthrobotic.perception.matching import consensus_match, flip_point
from depthrobotic.perception.parsing import (
    normalize_box,
    normalize_point,
    safe_ask_structured,
)
from depthrobotic.prompts import wrist_center_prompt
from depthrobotic.signals import StepBudgetExceeded, TaskCompleted
from depthrobotic.skills.pick_place.locate import (
    DEST_RELATION_DIRS,
    DEST_RELATION_MARGIN,
    FINGER_AXIS,
    RELEASE_GAP_ATTACHED,
    RELEASE_MIN_SEP_PX,
    RELEASE_PINCH_EMPTY,
    RELEASE_RETREAT,
    STRATEGY_DROP,
    locate_destination,
    locate_relational_source,
    parse_dest_relation,
)
from depthrobotic.skills.pick_place.state import region_descriptor
from depthrobotic.skills.pick_place.wrist import (
    box_metrics,
    measure_object_box,
    wrist_servo,
)
import numpy as np
import torch


class Placement:

    def locate_destination(
        self, env, obs, grounder, phrase, instruction, size, support_z, is_container
    ):
        result = self.locate_destination_correspondence(
            env, obs, grounder, phrase, instruction, size, support_z, is_container
        )
        if not is_container and self.memory is not None:
            point = self.match_destination()
            if point is not None:
                z = self.memory["z"]
                plane = z if z > support_z + 0.06 else support_z
                cam, focal, _ = camera_geometry(env, "agentview", size)
                world = intersect_horizontal_plane(
                    point, plane, cam, focal, size, context=self.context
                )
                if in_workspace(world[:2]):
                    result = (point, world, z)
                    self.dest_pre_point = point
                    self.memory_dest = (world, z)
                    self.record["tool_calls"].append(
                        {
                            "v30_surface_destination": {
                                "phrase": phrase,
                                "world": world.tolist(),
                                "z": z,
                            }
                        }
                    )
        point, world, z = result
        self.state.target.observe(
            self.env.steps,
            point=flip_point(point, not self.context.settings.orient_fix),
            world=world,
            surface_z=z,
        )
        self._save()
        return result

    def _container_realign(self, dest_desc, dest_world, plane_z, held_xy):
        self.state.contact("container_contact", self.env.steps)
        point = self.match_destination() if self.memory is not None else None
        if point is not None:
            cam, focal, _ = camera_geometry(self.env, "agentview", self.size)
            now = intersect_horizontal_plane(
                point, plane_z, cam, focal, self.size, context=self.context
            )
            if (
                in_workspace(now[:2])
                and np.linalg.norm(now[:2] - dest_world[:2]) < 0.12
            ):
                dest_world = now
                self.last_dest_world = np.asarray(now)
                self.state.target.observe(
                    self.env.steps,
                    point=flip_point(point, not self.context.settings.orient_fix),
                    world=now,
                )
        self._save()
        return self._container_realign_contact(dest_desc, dest_world, plane_z, held_xy)

    def _release_and_verify(self, held_xy, place_travel_z):
        self.state.contact("release_contact", self.env.steps)
        rec = self._release_and_verify_contact(held_xy, place_travel_z)
        self.state.held = rec.get("final_state", "UNKNOWN")
        self.state.held_offset = (
            None if self.state.held == "RELEASED" else self.state.held_offset
        )
        self._refresh_target()
        self._save()
        return rec

    def _refresh_target(self):
        """New visual/depth measurement after contact; old appearance only anchors identity."""
        ref = self._reference(self.target_label, "receiving region after contact")
        if ref is None or self.memory is None:
            return
        a = region_descriptor(self.memory["features"], self.memory["box"])
        b = region_descriptor(ref["features"], ref["box"])
        similarity = float(torch.dot(a, b)) if a is not None and b is not None else None
        dz = abs(ref["z"] - self.memory["z"])
        consistent = similarity is not None and similarity >= 0.65 and (dz <= 0.08)
        self.state.record(
            "receiving_reobserve",
            self.env.steps,
            appearance_similarity=similarity,
            height_change=dz,
            accepted=consistent,
        )
        if not consistent:
            return
        point = flip_point(ref["point"], not self.context.settings.orient_fix)
        cam, focal, _ = camera_geometry(self.env, "agentview", self.size)
        world = intersect_horizontal_plane(
            point, ref["z"], cam, focal, self.size, context=self.context
        )
        if not in_workspace(world[:2]):
            return
        self.state.target.observe(
            self.env.steps,
            point=ref["point"],
            box=ref["box"],
            world=world,
            surface_z=ref["z"],
        )
        self.memory = ref
        self.dest_pre_point = point
        self.last_dest_world = world

    def _measure_released_object(self, exp_obj):
        xy = self._measure_released_object_contact(exp_obj)
        descriptor = self._wrist_descriptor() if xy is not None else None
        score = (
            float(torch.dot(self.release_descriptor, descriptor))
            if self.release_descriptor is not None and descriptor is not None
            else None
        )
        self.release_association = (
            "MATCHED" if score is not None and score >= 0.65 else "UNKNOWN"
        )
        self.state.source.association = self.release_association
        self.state.record(
            "released_object",
            self.env.steps,
            appearance_similarity=score,
            measured_xy=None if xy is None else np.asarray(xy).tolist(),
        )
        if score is not None and score < 0.4:
            return None
        return xy

    def _check_inside_container(
        self, dest_desc, dest_world, plane_z, release_eef, held_xy
    ):
        if self.state.target.geometry_valid:
            dest_world = np.asarray(self.state.target.world)
            plane_z = self.state.target.surface_z
        rec = self._check_inside_container_contact(
            dest_desc, dest_world, plane_z, release_eef, held_xy
        )
        relation = self.state.set_relation(rec.get("inside"))
        if rec.get("inside") and relation != "OBSERVED_SATISFIED":
            rec["geometric_inside"] = rec.pop("inside")
            rec["skipped"] = (
                "identity or post-contact receiving geometry remains unverified"
            )
        self._save()
        return rec

    def _look_and_nudge(self, dest_desc, dest_world, plane_z, release_eef, held_xy):
        if self.state.target.geometry_valid and self.dest_anchor is None:
            dest_world = np.asarray(self.state.target.world)
            plane_z = self.state.target.surface_z
        rec = self._look_and_nudge_contact(
            dest_desc, dest_world, plane_z, release_eef, held_xy
        )
        if rec:
            error = rec.get("final_error_cm", rec.get("error_cm"))
            self.state.set_relation(None if error is None else error <= 1.5)
        self._save()
        return rec

    def _place_object(self, args):
        text, image = self._place_object_transport(args)
        if self.placed_once and (not self.held) and (self.state.relation == "UNKNOWN"):
            text += " Completion of the specified object relation is unverified. Use a fresh observation or retry_place when feasible."
        self._save()
        return (text, image)

    def match_destination(self):
        if self.memory is None:
            return None
        current = self.context.models.features.features(
            upright(
                observation_image(self.obs, "agentview_image", context=self.context),
                context=self.context,
            )
        )
        point, rec = consensus_match(
            self.memory["features"],
            current,
            self.memory["box"],
            self.memory["point"],
            self.context.models.features.resolution // 16,
        )
        rec["mode"] = "eupe"
        self.record["tool_calls"].append({"eupe_match": rec})
        return (
            flip_point(point, not self.context.settings.orient_fix)
            if point is not None
            else None
        )

    def locate_destination_correspondence(
        self, env, obs, grounder, phrase, instruction, size, support_z, is_container
    ):
        legacy = self.locate_destination_geometry(
            env, obs, grounder, phrase, instruction, size, support_z, is_container
        )
        self.memory_dest = None
        if not is_container or self.memory is None:
            return legacy
        point = self.match_destination()
        if point is None or "eupe" != "eupe":
            return legacy
        cam, focal, _ = camera_geometry(env, "agentview", size)
        z = self.memory["z"]
        world = intersect_horizontal_plane(
            point, z, cam, focal, size, context=self.context
        )
        valid = in_workspace(world[:2])
        self.record["tool_calls"].append(
            {
                "eupe_destination": {
                    "accepted": bool(valid),
                    "point": point,
                    "z": z,
                    "world": world.tolist(),
                    "legacy_world": np.asarray(legacy[1]).tolist(),
                    "legacy_z": float(legacy[2]),
                }
            }
        )
        if not valid:
            return legacy
        self.dest_pre_point = point
        self.memory_dest = (world, z)
        return (point, world, z)

    def _servo_destination(
        self, dest_desc, plane_z, iterations, phase, expected_world=None, max_jump=0.08
    ):
        if "eupe" != "eupe" or self.memory_dest is None:
            return self._servo_destination_contact(
                dest_desc, plane_z, iterations, phase, expected_world, max_jump
            )
        for it in range(iterations):
            point = self.match_destination()
            if point is None:
                self.record["tool_calls"].append(
                    {"eupe_servo": {"reason": "unmatched_keep_coarse"}}
                )
                break
            cam, focal, _ = camera_geometry(self.env, "agentview", self.size)
            world = intersect_horizontal_plane(
                point, plane_z, cam, focal, self.size, context=self.context
            )
            eef = np.asarray(self.obs["robot0_eef_pos"])
            delta = world[:2] - (eef[:2] + self._held_xy())
            allowed = float(np.linalg.norm(delta)) <= max_jump and in_workspace(
                world[:2] - self._held_xy()
            )
            self.record["tool_calls"].append(
                {
                    "eupe_servo": {
                        "iteration": it,
                        "delta": delta.tolist(),
                        "allowed": bool(allowed),
                    }
                }
            )
            if not allowed or np.linalg.norm(delta) < PRECISE_TOL:
                break
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[eef[:2] + delta, eef[2]],
                1.0,
                self.events,
                f"{phase}:eupe:{it}",
                max_steps=30,
            )

    def destination_plane(self, surface_z, is_container):
        return receiving_plane(surface_z, self.support_z, is_container)

    def locate_destination_geometry(
        self, env, obs, grounder, phrase, instruction, size, support_z, is_container
    ):
        point, world, z = locate_destination(
            env,
            obs,
            grounder,
            phrase,
            instruction,
            size,
            support_z,
            is_container,
            context=self.context,
        )
        plane = receiving_plane(z, support_z, is_container)
        old_world = np.asarray(world).copy()
        if not is_container and plane != support_z:
            camera, focal, _ = camera_geometry(env, "agentview", size)
            world = intersect_horizontal_plane(
                point, plane, camera, focal, size, context=self.context
            )
        recovery = None
        if not in_workspace(world[:2]):
            image = observation_image(obs, "agentview_image", context=self.context)
            raw_box, value = safe_ask_structured(
                grounder,
                image,
                f'Instruction: {instruction}\nFind the receiving object/region {phrase}. Return its visible bounding box, not the held source object. JSON {{"bbox_2d":[x1,y1,x2,y2]}}, 0..1000, or bbox_2d:null.',
                100,
            )
            raw = value.get("bbox_2d")
            box = (
                normalize_box(raw) if isinstance(raw, list) and len(raw) == 4 else None
            )
            recovery = {"bbox_response": raw_box, "accepted": False}
            if (
                box is not None
                and 0 <= box[0] < box[2] <= 1000
                and (0 <= box[1] < box[3] <= 1000)
            ):
                center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                fraction = min(1.0, max(box[2] - box[0], box[3] - box[1]) / 1000 * 1.2)
                crop, bounds = crop_from_normalized(image, center, fraction)
                raw_point, value = safe_ask_structured(
                    grounder,
                    crop,
                    f'Locate the usable receiving surface for {phrase}. For shelves choose the horizontal support, for containers the interior. Preserve the instruction: {instruction}. JSON {{"point":[x,y]}} in this crop, 0..1000, or point:null.',
                    96,
                )
                recovery["point_response"] = raw_point
                local = normalize_point(value.get("point"))
                if local is not None:
                    candidate = crop_point_to_global(local, bounds, image.size)
                    depth, c2w, focal, _, scale = frame_geometry(
                        env, grounder, image, size, support_z, context=self.context
                    )
                    predicted = point_world(
                        depth, candidate, scale, c2w, focal, size, context=self.context
                    )
                    new_z = float(
                        np.clip(predicted[2], support_z - 0.08, support_z + 0.3)
                    )
                    if new_z >= support_z + 0.28:
                        new_z = support_z + (0.05 if is_container else 0.0)
                    new_plane = receiving_plane(new_z, support_z, is_container)
                    new_world = intersect_horizontal_plane(
                        candidate, new_plane, c2w, focal, size, context=self.context
                    )
                    recovery.update(
                        candidate=candidate, world=np.asarray(new_world).tolist()
                    )
                    if in_workspace(new_world[:2]):
                        point, world, z, plane = (
                            candidate,
                            new_world,
                            new_z,
                            new_plane,
                        )
                        self.dest_pre_point = None
                        recovery["accepted"] = True
        self.record["tool_calls"].append(
            {
                "destination_geometry": {
                    "description": phrase,
                    "pixel": point,
                    "world": np.asarray(world).tolist(),
                    "legacy_world": old_world.tolist(),
                    "surface_z": float(z),
                    "projection_plane_z": float(plane),
                    "is_container": bool(is_container),
                    "relocalization": recovery,
                    "corrected_elevated_surface": bool(
                        not is_container and plane != support_z
                    ),
                }
            }
        )
        return (point, world, z)

    def _place_object_transport(self, args):
        if not self.held:
            return ("Error: no object held. Grasp first.", None)
        dest_desc = args.get("destination_description", self.destination)
        is_container = bool(args.get("is_container", False))
        self.last_dest_desc = dest_desc
        self.last_is_container = is_container
        self.dest_anchor = None
        self.place_blocked = None
        rel = parse_dest_relation(dest_desc) or parse_dest_relation(self.destination)
        if rel is not None and rel["relation"] in DEST_RELATION_DIRS:
            is_container = False
            direction = DEST_RELATION_DIRS[rel["relation"]]
            _, anchor_world, _ = self.locate_destination(
                self.env,
                self.obs,
                self.grounder,
                rel["anchor"],
                self.instruction,
                self.size,
                self.support_z,
                False,
            )
            half = self._anchor_half_extent_agentview(
                rel["anchor"], anchor_world, direction
            )
            obj_half = float(self.width_finger) / 2 if self.width_finger else 0.03
            offset = direction * float(
                np.clip(
                    (half if half is not None else 0.08)
                    + obj_half
                    + DEST_RELATION_MARGIN,
                    0.06,
                    0.16,
                )
            )
            dest_world = np.r_[
                np.asarray(anchor_world[:2], dtype=float) + offset, self.support_z
            ]
            dest_z = self.support_z
            self.dest_anchor = {
                "desc": rel["anchor"],
                "world": np.asarray(anchor_world, dtype=float),
                "dir": direction,
                "offset": offset,
                "half": half,
                "obj_half": obj_half,
            }
            self.record["tool_calls"].append(
                {
                    "relational_destination": {
                        "relation": rel["relation"],
                        "anchor": rel["anchor"],
                        "anchor_world": np.asarray(anchor_world[:2]).round(4).tolist(),
                        "anchor_half_m": None if half is None else round(half, 3),
                        "obj_half_m": round(obj_half, 3),
                        "offset_m": offset.round(4).tolist(),
                        "dest_world": dest_world[:2].round(4).tolist(),
                    }
                }
            )
        else:
            _, dest_world, dest_z = self.locate_destination(
                self.env,
                self.obs,
                self.grounder,
                dest_desc,
                self.instruction,
                self.size,
                self.support_z,
                is_container,
            )
        if self.dest_pre_point is not None and self.dest_anchor is None:
            cam2world, focal, _ = camera_geometry(self.env, "agentview", self.size)
            proj_z = self.destination_plane(dest_z, is_container)
            pre_world = intersect_horizontal_plane(
                self.dest_pre_point,
                proj_z,
                cam2world,
                focal,
                self.size,
                context=self.context,
            )
            gap = float(np.linalg.norm(pre_world[:2] - dest_world[:2]))
            self.record["dest_pre_vs_now_m"] = gap
            if gap > 0.08:
                dest_world = pre_world
                self.record["dest_used_pregrasp_grounding"] = True
            elif gap > 0.02:
                dest_world = np.r_[(pre_world[:2] + dest_world[:2]) / 2, dest_world[2]]
        elevated = dest_z > self.support_z + 0.06
        if is_container or elevated:
            place_travel_z = min(
                max(self.travel_z + 0.08, dest_z + 0.22), self.support_z + 0.34
            )
        else:
            place_travel_z = max(self.travel_z - 0.04, dest_z + 0.1)
        self.last_place_travel_z = place_travel_z
        self.last_dest_world = np.asarray(dest_world, dtype=float)
        held_xy = self._held_xy()
        drop = float(-self.held_offset3[2]) if self.held_offset3 is not None else None
        plane_z = dest_z if is_container else max(dest_z, self.support_z)
        if is_container or elevated:
            cur_xy = np.asarray(self.obs["robot0_eef_pos"], dtype=float)[:2]
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[cur_xy, place_travel_z],
                1.0,
                self.events,
                "raise_for_dest",
                max_steps=35,
                tol=0.02,
            )
        coarse_xy = np.array(
            [
                np.clip(dest_world[0] - held_xy[0], *WORKSPACE_X),
                np.clip(dest_world[1] - held_xy[1], *WORKSPACE_Y),
            ]
        )
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[coarse_xy, place_travel_z],
            1.0,
            self.events,
            "coarse_dest",
            max_steps=40,
            tol=0.012,
        )
        if is_container:
            self._servo_destination(
                dest_desc,
                plane_z,
                iterations=2,
                phase="dest_servo_high",
                expected_world=dest_world,
                max_jump=0.05,
            )
            if self.width_finger is not None:
                eef_now = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
                ext = self._measure_destination_extent(
                    dest_desc, np.r_[eef_now[:2] + held_xy, plane_z], plane_z
                )
                fit = {"object_width_m": round(float(self.width_finger), 3)}
                if ext is not None:
                    _, hx, hy = ext
                    interior = 2 * float(
                        np.dot(np.array([hx, hy]), np.abs(FINGER_AXIS))
                    )
                    fit.update(
                        {
                            "interior_width_m": round(interior, 3),
                            "interior_across_m": round(
                                2
                                * float(
                                    np.dot(np.array([hx, hy]), 1 - np.abs(FINGER_AXIS))
                                ),
                                3,
                            ),
                        }
                    )
                    if (
                        0.04 < interior < 0.4
                        and 0.04 < fit["interior_across_m"] < 0.4
                        and (float(self.width_finger) > interior + 0.03)
                    ):
                        fit["verdict"] = "does_not_fit"
                        self.record["tool_calls"].append({"fit_check": fit})
                        return (
                            f"Cannot place '{self.source}' into '{dest_desc}': the object measures {self.width_finger * 100:.0f} cm along the gripper axis but the visible interior is only {interior * 100:.0f} cm wide, so it will not fit this way. The object is still held. Call done.",
                            None,
                        )
                self.record["tool_calls"].append({"fit_check": fit})
        else:
            held_xy = self._align_before_release(
                dest_desc, dest_world, plane_z, held_xy
            )
        release_z = plane_z + 0.005
        cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        expected = self._expected_stall_above()
        fast_z = release_z + max(expected - 0.02, 0.0)
        self.record["tool_calls"].append(
            {
                "place_descent": {
                    "release_z": round(release_z, 4),
                    "expected_stall_above_m": round(expected, 3),
                    "fast_z": round(fast_z, 4),
                }
            }
        )
        contact = False
        reached = False
        if cur[2] > fast_z + 0.01:
            self.obs, reached, contact = descend_hold_xy(
                self.env,
                self.obs,
                np.r_[cur[:2], fast_z],
                1.0,
                self.events,
                "place_fast",
                max_steps=40,
            )
        if not contact:
            cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            self.obs, reached, contact = descend_gentle_contact(
                self.env,
                self.obs,
                np.r_[cur[:2], release_z],
                self.events,
                "place",
                max_steps=16,
            )
        self.place_stall_above_m = None
        if is_container and contact:
            self._container_realign(dest_desc, dest_world, plane_z, held_xy)
            if self.place_blocked is not None:
                cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
                self.obs = move_to_precise(
                    self.env,
                    self.obs,
                    np.r_[cur[:2], place_travel_z],
                    1.0,
                    self.events,
                    "blocked_lift",
                    max_steps=25,
                    tol=0.02,
                )
                return (
                    f"Could not place '{self.source}' into '{dest_desc}': the descent is blocked {self.place_blocked['stall_above_m'] * 100:.0f} cm above the interior floor although the object is centred over the opening. The interior cannot be reached from above with this placement mode (unsupported). The object is still held. Call done.",
                    None,
                )
        if is_container:
            self.place_stall_above_m = (
                float(np.asarray(self.obs["robot0_eef_pos"])[2]) - release_z
            )
        release_eef = np.asarray(self.obs["robot0_eef_pos"], dtype=float).copy()
        release_rec = self._release_and_verify(held_xy, place_travel_z)
        state = release_rec.get("final_state", "UNKNOWN")
        if state == "ATTACHED" and self.context.settings.release_mode == "act":
            self.release_attached_count += 1
            more = (
                "Call place_object again to set it down once more (a different release motion will be used), or call done."
                if self.release_attached_count < 2
                else "Call done."
            )
            return (
                f"Released the gripper at '{dest_desc}' but the '{self.source}' is STILL ATTACHED to the fingers (gripper gap {release_rec.get('gap_after_open', 0) * 100:.1f} cm; the object moved with the hand when it rose). It was lifted again and is still held. {more}",
                None,
            )
        verify_note = (
            ""
            if state == "RELEASED"
            else " (release not verified: the object's outline could not be tracked)"
        )
        self.held = False
        self.held_offset3 = None
        self.placed_once = True
        outcome = None
        try:
            if not is_container:
                outcome = self._look_and_nudge(
                    dest_desc, dest_world, plane_z, release_eef, held_xy
                )
            else:
                outcome = self._check_inside_container(
                    dest_desc, dest_world, plane_z, release_eef, held_xy
                )
        except (TaskCompleted, StepBudgetExceeded):
            raise
        except Exception as e:
            self.record["tool_calls"].append(
                {"nudge_error": f"{type(e).__name__}: {e}"[:120]}
            )
        if outcome and outcome.get("final_error_cm") is not None:
            err_cm = outcome["final_error_cm"]
            how = "after a correction"
        elif (
            outcome
            and outcome.get("touched")
            and (outcome.get("final_error_cm") is None)
        ):
            err_cm = None
            how = "outcome unknown: a correction attempt touched the object and it could not be re-measured"
        elif outcome and outcome.get("error_cm") is not None:
            err_cm = outcome["error_cm"]
            how = (
                "no correction applied"
                if outcome.get("skipped")
                else "before correction"
            )
        else:
            err_cm = None
            how = (
                outcome.get("skipped", "not measured")
                if outcome
                else "not measured (container)"
            )
        if outcome and outcome.get("inside") is not None:
            if outcome["inside"]:
                return (
                    f"Placed '{self.source}' at '{dest_desc}'. The object is seen inside the receiving region ({outcome.get('offset_cm')} cm from its centre).{verify_note} Call done.",
                    None,
                )
            return (
                f"Placed '{self.source}' at '{dest_desc}', but it is probably NOT inside the receiving region ({outcome.get('note') or str(outcome.get('offset_cm')) + ' cm from its centre'}). Call retry_place if step budget allows, otherwise done.",
                None,
            )
        if err_cm is None:
            return (
                f"Placed '{self.source}' at '{dest_desc}'. Placement could not be measured ({how}).{verify_note} Call done, or retry_place if you believe it is wrong.",
                None,
            )
        if err_cm <= 3.0:
            return (
                f"Placed '{self.source}' at '{dest_desc}'. Measured offset of the object from the destination center: {err_cm:.1f} cm ({how}), within the ~3 cm tolerance.{verify_note} Call done.",
                None,
            )
        return (
            f"Placed '{self.source}' at '{dest_desc}', but the measured offset from the destination center is {err_cm:.1f} cm ({how}), outside the ~3 cm tolerance. Call retry_place to pick it up and place it again if step budget allows, otherwise call done.",
            None,
        )

    def _measure_destination_extent(self, dest_desc, expected_world, plane_z):
        """Wrist-view destination centre and metric half extents from its bbox. Returns (xy, half_x, half_y) or None."""
        full = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        try:
            exp_px = project_world_to_raw(
                self.env,
                "robot0_eye_in_hand",
                np.r_[expected_world[:2], plane_z],
                self.size,
                context=self.context,
            )
        except RuntimeError:
            return None
        if not (0 <= exp_px[0] <= 1000 and 0 <= exp_px[1] <= 1000):
            return None
        crop, bounds = crop_from_normalized(full, exp_px, 0.7)
        prompt = f'Robot instruction: {self.instruction}\nThis wrist-camera crop looks down at the destination {dest_desc}. Return the bounding box of the WHOLE {dest_desc} outline (its outer edge), including any part covered by an object. JSON only, normalized 0..1000 within this crop: {{"bbox_2d":[x1,y1,x2,y2]}}.'
        _, value = safe_ask_structured(self.grounder, crop, prompt, 96)
        box = value.get("bbox_2d") or value.get("bbox") or value.get("box")
        box = normalize_box(box) if box is not None else None
        if box is None or not (
            60 < box[2] - box[0] < 950 and 60 < box[3] - box[1] < 950
        ):
            return None
        corners = [
            crop_point_to_global([box[0], box[1]], bounds, full.size),
            crop_point_to_global([box[2], box[3]], bounds, full.size),
        ]
        cam2world, focal, _ = camera_geometry(self.env, "robot0_eye_in_hand", self.size)
        try:
            w1 = intersect_horizontal_plane(
                corners[0], plane_z, cam2world, focal, self.size, context=self.context
            )[:2]
            w2 = intersect_horizontal_plane(
                corners[1], plane_z, cam2world, focal, self.size, context=self.context
            )[:2]
        except RuntimeError:
            return None
        return ((w1 + w2) / 2, abs(w2[0] - w1[0]) / 2, abs(w2[1] - w1[1]) / 2)

    def _container_realign_contact(self, dest_desc, dest_world, plane_z, held_xy):
        """Place descent stalled above the container floor: lift a little, re-centre the held object on the
        interior seen in the wrist image, descend again (once)."""
        rec = {"phase": "container_realign"}
        self.record["tool_calls"].append({"realign": rec})
        cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[cur[:2], cur[2] + 0.05],
            1.0,
            self.events,
            "realign_lift",
            max_steps=15,
            tol=0.01,
        )
        ext = self._measure_destination_extent(
            dest_desc, np.r_[cur[:2] + held_xy, plane_z], plane_z
        )
        if ext is None:
            self.obs, _, _ = descend_hold_xy(
                self.env,
                self.obs,
                np.r_[cur[:2], cur[2] - 0.002],
                1.0,
                self.events,
                "realign_return",
                max_steps=15,
            )
            rec["skipped"] = "no interior measurement"
            return
        center_xy, hx, hy = ext
        delta = np.asarray(center_xy) - (cur[:2] + held_xy)
        rec.update(
            {
                "delta_m": delta.round(4).tolist(),
                "half_extent": [round(hx, 3), round(hy, 3)],
            }
        )
        if float(np.linalg.norm(delta)) > 0.1:
            rec["skipped"] = "implausible interior centre"
            return
        stall_above = float(cur[2]) - (plane_z + 0.005)
        expected = self._expected_stall_above()
        rec.update(
            {
                "stall_above_m": round(stall_above, 3),
                "expected_stall_above_m": round(expected, 3),
            }
        )
        if float(np.linalg.norm(delta)) < 0.03 and stall_above > expected + 0.06:
            self.place_blocked = {
                "stall_above_m": round(stall_above, 3),
                "expected_m": round(expected, 3),
                "delta_m": delta.round(4).tolist(),
            }
            rec["blocked_from_above"] = True
            return
        goal = np.array(
            [
                np.clip(cur[0] + delta[0], *WORKSPACE_X),
                np.clip(cur[1] + delta[1], *WORKSPACE_Y),
            ]
        )
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[goal, cur[2] + 0.05],
            1.0,
            self.events,
            "realign_shift",
            max_steps=25,
            tol=0.008,
        )
        fast_z = plane_z + 0.005 + max(expected - 0.02, 0.0)
        contact = False
        reached = False
        if float(np.asarray(self.obs["robot0_eef_pos"])[2]) > fast_z + 0.01:
            self.obs, reached, contact = descend_hold_xy(
                self.env,
                self.obs,
                np.r_[goal, fast_z],
                1.0,
                self.events,
                "realign_place_fast",
                max_steps=40,
            )
        if not contact:
            self.obs, reached, contact = descend_gentle_contact(
                self.env,
                self.obs,
                np.r_[goal, plane_z + 0.005],
                self.events,
                "realign_place",
                max_steps=16,
            )
        rec.update(
            {
                "reached": reached,
                "contact": contact,
                "eef_z": float(np.asarray(self.obs["robot0_eef_pos"])[2]),
            }
        )

    def _expected_stall_above(self):
        """How far above the container floor the eef legitimately stops when the held object's bottom touches it."""
        height_est = (
            self.top_z - self.support_z + 0.03 if self.top_z is not None else 0.05
        )
        return max(height_est - STRATEGY_DROP.get(self.current_strategy, 0.02), 0.0)

    def _anchor_half_extent_agentview(self, anchor_desc, anchor_world, direction):
        """Half extent of the anchor along `direction` (world xy) from its agentview bounding box on the table plane."""
        full = observation_image(self.obs, "agentview_image", context=self.context)
        cam2world, focal, _ = camera_geometry(self.env, "agentview", self.size)
        try:
            px = project_world_to_raw(
                self.env,
                "agentview",
                np.r_[np.asarray(anchor_world[:2]), self.support_z],
                self.size,
                context=self.context,
            )
        except RuntimeError:
            return None
        if not (0 <= px[0] <= 1000 and 0 <= px[1] <= 1000):
            return None
        crop, bounds = crop_from_normalized(full, px, 0.45)
        prompt = f'Robot instruction: {self.instruction}\nThis crop of the front camera shows the {anchor_desc}. Return the bounding box of the WHOLE {anchor_desc} (its full outer extent). JSON only, normalized 0..1000 within this crop: {{"bbox_2d":[x1,y1,x2,y2]}}.'
        _, value = safe_ask_structured(self.grounder, crop, prompt, 96)
        box = value.get("bbox_2d") or value.get("bbox") or value.get("box")
        box = normalize_box(box) if box is not None else None
        if box is None or not (
            40 < box[2] - box[0] < 950 and 40 < box[3] - box[1] < 950
        ):
            return None
        c1 = crop_point_to_global([box[0], (box[1] + box[3]) / 2], bounds, full.size)
        c2 = crop_point_to_global([box[2], (box[1] + box[3]) / 2], bounds, full.size)
        r1 = crop_point_to_global([(box[0] + box[2]) / 2, box[1]], bounds, full.size)
        r2 = crop_point_to_global([(box[0] + box[2]) / 2, box[3]], bounds, full.size)
        try:
            w1 = intersect_horizontal_plane(
                c1, self.support_z, cam2world, focal, self.size, context=self.context
            )[:2]
            w2 = intersect_horizontal_plane(
                c2, self.support_z, cam2world, focal, self.size, context=self.context
            )[:2]
            v1 = intersect_horizontal_plane(
                r1, self.support_z, cam2world, focal, self.size, context=self.context
            )[:2]
            v2 = intersect_horizontal_plane(
                r2, self.support_z, cam2world, focal, self.size, context=self.context
            )[:2]
        except RuntimeError:
            return None
        d = np.asarray(direction, dtype=float)
        half = max(abs(float(np.dot(w2 - w1, d))), abs(float(np.dot(v2 - v1, d)))) / 2
        return half if 0.03 < half < 0.2 else None

    def _check_inside_container_contact(
        self, dest_desc, dest_world, plane_z, release_eef, held_xy
    ):
        """After releasing into a container: is the object seen inside the receiving region? One image, no motion."""
        rec = {"phase": "inside_check"}
        self.record["tool_calls"].append({"nudge": rec})
        obj_xy = self._measure_released_object(
            np.r_[release_eef[:2] + held_xy, plane_z + 0.03]
        )
        ext = self._measure_destination_extent(dest_desc, dest_world, plane_z)
        if obj_xy is None or ext is None:
            rec["skipped"] = "object or container not measured"
            return rec
        center_xy, hx, hy = ext
        off = np.asarray(obj_xy) - np.asarray(center_xy)
        inside = bool(abs(off[0]) <= hx * 0.9 and abs(off[1]) <= hy * 0.9)
        stall = getattr(self, "place_stall_above_m", None)
        rec.update(
            {
                "inside": inside,
                "offset_cm": round(float(np.linalg.norm(off)) * 100, 1),
                "container_half_extent": [round(hx, 3), round(hy, 3)],
                "place_stall_above_m": stall,
            }
        )
        height_est = (
            self.top_z - self.support_z + 0.03 if self.top_z is not None else 0.05
        )
        expected_stall = max(
            height_est - STRATEGY_DROP.get(self.current_strategy, 0.02), 0.0
        )
        rec["expected_stall_above_m"] = round(expected_stall, 3)
        if stall is not None and stall > expected_stall + 0.04:
            rec["inside"] = False
            rec["note"] = (
                f"descent stopped {stall * 100:.0f}cm above the floor estimate, about {(stall - expected_stall) * 100:.0f}cm higher than the object's own height allows (object likely resting on the rim)"
            )
        return rec

    def _release_tracking_check(
        self, obj_world, gap, phase, rise=RELEASE_RETREAT, pre=None
    ):
        """Rise straight up by `rise` and decide whether the object stayed behind. A released object's outline moves
        to the pixel where its set-down world position projects from the new camera pose; an attached object keeps
        its old pixel (camera and object move together). Returns (state, record)."""
        track = {"rise_m": rise, "gap": round(float(gap), 4)}
        obj_world = np.asarray(obj_world, dtype=float)

        def _proj():
            try:
                return np.asarray(
                    project_world_to_raw(
                        self.env,
                        "robot0_eye_in_hand",
                        obj_world,
                        self.size,
                        context=self.context,
                    )
                )
            except RuntimeError:
                return None

        if pre is not None:
            px0, box0, pt0, exp0 = pre
            gray0 = None
        else:
            exp0 = _proj()
            px0, box0 = self._object_outline_px()
            pt0 = self._object_point_px() if px0 is None else None
            gray0 = self._wrist_gray()
        if rise > 0:
            cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[cur[:2], cur[2] + rise],
                -1.0,
                self.events,
                phase,
                max_steps=12,
                tol=0.008,
            )
        exp1 = _proj()
        px1, box1 = self._object_outline_px()
        pt1 = self._object_point_px() if px1 is None or px0 is None else None
        gray1 = self._wrist_gray()
        track.update(
            {
                "px_before": px0,
                "px_after": px1,
                "box_before": box0,
                "box_after": box1,
                "pt_before": pt0,
                "pt_after": pt1,
                "exp_before": None if exp0 is None else exp0.round(1).tolist(),
                "exp_after": None if exp1 is None else exp1.round(1).tolist(),
                "patch": (
                    self._patch_similarity(gray0, gray1, box0)
                    if gray0 is not None
                    else None
                ),
            }
        )
        votes = {"attached": 0, "released": 0}
        sep = None
        if exp0 is not None and exp1 is not None:
            motion = exp1 - exp0
            sep = float(np.linalg.norm(motion))
            pairs = []
            if px0 is not None and px1 is not None:
                b0, b1 = (np.asarray(box0), np.asarray(box1))
                pairs = [
                    ("centre", np.asarray(px0), np.asarray(px1)),
                    ("tl", b0[:2], b1[:2]),
                    ("br", b0[2:], b1[2:]),
                ]
                track["tracker"] = "outline"
            elif pt0 is not None and pt1 is not None:
                pairs = [("point", np.asarray(pt0), np.asarray(pt1))]
                track["tracker"] = "point"
            det = []
            for name, p0, p1 in pairs:
                d_att = float(np.linalg.norm(p1 - p0))
                d_stat = float(np.linalg.norm(p1 - (p0 + motion)))
                v = (
                    "attached"
                    if d_att < d_stat and d_att < 0.4 * sep
                    else "released" if d_stat < d_att and d_stat < 0.6 * sep else "none"
                )
                if name == "point" and v == "attached":
                    v = "none"
                det.append(
                    {
                        "pt": name,
                        "d_attached_px": round(d_att, 1),
                        "d_stationary_px": round(d_stat, 1),
                        "vote": v,
                    }
                )
                if v != "none":
                    votes[v] += 1
            track.update({"sep_px": round(sep, 1), "votes": det})
        if sep is None or not track.get("tracker"):
            track["tracking_verdict"] = "no_tracker"
        elif sep < RELEASE_MIN_SEP_PX:
            track["tracking_verdict"] = "not_discriminable"
        else:
            need = 2 if track["tracker"] == "outline" else 1
            if votes["attached"] >= need and votes["attached"] > votes["released"]:
                track["tracking_verdict"] = "ATTACHED"
            elif votes["released"] >= need and votes["released"] > votes["attached"]:
                track["tracking_verdict"] = "RELEASED"
            else:
                track["tracking_verdict"] = "ambiguous"
        if gap < RELEASE_GAP_ATTACHED:
            track["by"] = "gap"
            return ("ATTACHED", track)
        self.obs = hold(self.env, self.obs, 1.0, 14, self.events, f"{phase}:pinch_test")
        pinch = self._gripper_gap()
        track["pinch_gap"] = round(float(pinch), 4)
        if pinch >= RELEASE_PINCH_EMPTY:
            track["by"] = "pinch_test"
            return ("ATTACHED", track)
        track["by"] = "pinch_test"
        return ("RELEASED", track)

    def _unhook_shift_direction(self):
        """Sign along the finger axis that moves the hand toward the destination centre (push the object inward)."""
        dest = getattr(self, "last_dest_world", None)
        if dest is None:
            return 1.0
        d = float(
            np.dot(
                np.asarray(dest[:2], dtype=float)
                - np.asarray(self.obs["robot0_eef_pos"], dtype=float)[:2],
                FINGER_AXIS,
            )
        )
        return 1.0 if d >= 0 else -1.0

    def _unhook(self, setdown_eef, held_xy, attempt, gap_after_open):
        """Three ways an 'open' gripper still carries the object, each with its own remedy (smoke evidence, §13):
        blocked opening (gap well below the 7.6cm free-air value; the resting object leans on a pad and the weak
        finger actuator cannot push it) -> (1) press the object flat (1.5cm below the set-down height) and open there;
        (2) pinch again, lift 4cm clear of the surface and open in the air (freed the spatial:1 bowl by accident).
        Full gap but tracked as moving with the hand (a tall object leaning between the open pads wedges by its height:
        w*cos(t)+h*sin(t) > opening at ~15 deg) -> (3) push it with ONE pad only: shift 3cm along the finger axis at the
        set-down height, then rise (one pad cannot hold it against gravity). Every variant ends with a full 12-step
        re-open (free air needs 12 steps from closed to reach 7.6cm) and the outline tracking check.
        """
        rec = {"attempt": attempt}
        setdown = np.asarray(setdown_eef, dtype=float)
        held = np.asarray(held_xy, dtype=float)
        blocked = gap_after_open < 0.07
        if blocked and attempt == 1:
            variant = "regrasp_lift_open"
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[setdown[:2], setdown[2] + 0.004],
                -1.0,
                self.events,
                f"unhook_lower:{attempt}",
                max_steps=14,
                tol=0.008,
            )
            self.obs = hold(
                self.env, self.obs, 1.0, 12, self.events, f"unhook_close:{attempt}"
            )
            grip = 1.0
        elif blocked:
            variant = "press_down_relief"
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[setdown[:2], setdown[2] - 0.015],
                -1.0,
                self.events,
                f"unhook_press:{attempt}",
                max_steps=12,
                tol=0.006,
            )
            self.obs = hold(
                self.env, self.obs, -1.0, 12, self.events, f"unhook_open:{attempt}"
            )
            rec["gap_after_press_open"] = round(self._gripper_gap(), 4)
            grip = -1.0
        else:
            variant = "finger_axis_shift"
            base = np.r_[setdown[:2], setdown[2] + 0.004]
            self.obs = move_to_precise(
                self.env,
                self.obs,
                base,
                -1.0,
                self.events,
                f"unhook_lower:{attempt}",
                max_steps=14,
                tol=0.008,
            )
            self.obs = hold(
                self.env, self.obs, -1.0, 8, self.events, f"unhook_open:{attempt}"
            )
            sgn = self._unhook_shift_direction() * (1.0 if attempt == 1 else -1.0)
            goal = np.r_[base[:2] + 0.03 * sgn * FINGER_AXIS, base[2]]
            self.obs = move_to_precise(
                self.env,
                self.obs,
                goal,
                -1.0,
                self.events,
                f"unhook_shift:{attempt}",
                max_steps=10,
                tol=0.005,
            )
            rec["shift_sign"] = sgn
            grip = -1.0
        rec["variant"] = variant
        eef = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        obj_world = np.r_[
            eef[:2] + held, setdown[2] + STRATEGY_DROP.get(self.current_strategy, 0.02)
        ]
        px0, box0 = self._object_outline_px()
        pt0 = self._object_point_px() if px0 is None else None
        exp0 = None
        try:
            exp0 = np.asarray(
                project_world_to_raw(
                    self.env,
                    "robot0_eye_in_hand",
                    obj_world,
                    self.size,
                    context=self.context,
                )
            )
        except RuntimeError:
            pass
        z0 = float(eef[2])
        for i in range(5):
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[eef[:2], z0 + 0.008 * (i + 1)],
                grip,
                self.events,
                f"unhook_rise:{attempt}:{i}",
                max_steps=3,
                tol=0.004,
            )
        self.obs = hold(
            self.env, self.obs, -1.0, 12, self.events, f"unhook_reopen:{attempt}"
        )
        gap = self._gripper_gap()
        rec["gap_after_reopen"] = round(float(gap), 4)
        state, track = self._release_tracking_check(
            obj_world,
            gap,
            f"unhook_check:{attempt}",
            rise=0.0,
            pre=(px0, box0, pt0, exp0),
        )
        rec.update({"track": track, "state": state})
        return rec

    def _release_and_verify_contact(self, held_xy, place_travel_z):
        """Release state machine (v12). Opening the gripper is not the same as having put the object down: v10/v11
        spatial:1 hooked a bowl rim on a finger (gap 5.5cm), v11 object:7 carried a carton away with the gripper fully
        open. Signals: the gripper gap after opening (primary) and outline tracking during a 4cm straight retreat.
        States RELEASED / ATTACHED / UNKNOWN; v11's outline-width test (4 false positives, 2 misses) is gone.
        Unhook only on ATTACHED (RELEASE_MODE == "act")."""
        rec = {"phase": "release_verify", "mode": self.context.settings.release_mode}
        self.record["tool_calls"].append({"release": rec})
        eef0 = np.asarray(self.obs["robot0_eef_pos"], dtype=float).copy()
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[eef0[:2], eef0[2] + 0.008],
            1.0,
            self.events,
            "release_unload",
            max_steps=4,
            tol=0.004,
        )
        self.obs = hold(self.env, self.obs, -1.0, 12, self.events, "release")
        gap = self._gripper_gap()
        tries = 0
        while gap < 0.07 and tries < (2 if gap < 0.05 else 1):
            cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[cur[:2], cur[2] + 0.012],
                -1.0,
                self.events,
                f"release_relieve:{tries}",
                max_steps=8,
                tol=0.006,
            )
            self.obs = hold(
                self.env, self.obs, -1.0, 8, self.events, f"release_open:{tries}"
            )
            gap = self._gripper_gap()
            tries += 1
        rec["gap_after_open"] = round(float(gap), 4)
        held = np.asarray(held_xy, dtype=float)
        drop = STRATEGY_DROP.get(self.current_strategy, 0.02)
        obj_world = np.r_[eef0[:2] + held, eef0[2] + drop]
        state, track = self._release_tracking_check(
            obj_world, gap, "retreat_check", rise=max(RELEASE_RETREAT, drop + 0.025)
        )
        rec.update({"track": track, "state": state})
        attempts = 0
        while (
            state == "ATTACHED"
            and self.context.settings.release_mode == "act"
            and (attempts < 2)
            and (int(self.env.budget - self.env.steps) > 60)
        ):
            attempts += 1
            un = self._unhook(eef0, held, attempts, gap)
            rec.setdefault("unhook", []).append(un)
            state = un["state"]
        rec["final_state"] = state
        try:
            retreat_z = max(
                place_travel_z, float(np.asarray(self.obs["robot0_eef_pos"])[2]) + 0.06
            )
            self.obs = move_to(
                self.env,
                self.obs,
                np.r_[
                    np.asarray(self.obs["robot0_eef_pos"], dtype=float)[:2], retreat_z
                ],
                -1.0,
                self.events,
                "retreat",
                max_steps=30,
            )
        except RuntimeError:
            pass
        return rec

    def _align_before_release(self, dest_desc, dest_world, plane_z, held_xy):
        """Before releasing on a surface: (1) the hanging object's outline centre, projected on its own top plane
        (rim at fingertips + drop), gives the real hang offset; (2) the destination's outline centre (outer edge
        visible around the object) gives the target. Move so the object, not the eef, is over the target.
        """
        rec = {"phase": "align_before_release", "iters": []}
        self.record["tool_calls"].append({"align": rec})
        held = np.asarray(held_xy, dtype=float).copy()
        ref_w = getattr(self, "width_finger", None)
        for it in range(1):
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            obj_plane = float(eef[2]) + STRATEGY_DROP.get(self.current_strategy, 0.02)
            entry = {"it": it}
            box, _ = measure_object_box(
                self.grounder,
                self.obs,
                self.source,
                self.instruction,
                context=self.context,
            )
            if box is not None:
                try:
                    wf, wa, bc = box_metrics(
                        self.env, self.size, box, obj_plane, context=self.context
                    )
                    entry.update(
                        {
                            "obj_width_m": round(wf, 3),
                            "hang_meas": (bc - eef[:2]).round(4).tolist(),
                        }
                    )
                    plausible = (
                        ref_w is None or 0.6 < wf / max(ref_w, 0.001) < 1.4
                    ) and float(np.linalg.norm(bc - eef[:2])) < 0.1
                    if plausible:
                        held = bc - eef[:2]
                        self.measured_hang_xy = held.copy()
                except RuntimeError:
                    pass
            dest_xy = np.asarray(dest_world[:2], dtype=float)
            if self.dest_anchor is not None:
                anc = self.dest_anchor
                ext = self._measure_destination_extent(
                    anc["desc"], np.r_[np.asarray(anc["world"][:2]), plane_z], plane_z
                )
                if ext is not None:
                    cxy, hx, hy = ext
                    entry["anchor_outline"] = np.asarray(cxy).round(4).tolist()
                    entry["anchor_half"] = [round(hx, 3), round(hy, 3)]
                    if (
                        float(
                            np.linalg.norm(
                                np.asarray(cxy) - np.asarray(anc["world"][:2])
                            )
                        )
                        < 0.06
                    ):
                        axis_half = abs(
                            float(np.dot(np.array([hx, hy]), np.abs(anc["dir"])))
                        )
                        offset = anc["offset"]
                        if 0.03 < axis_half < 0.2:
                            offset = anc["dir"] * float(
                                np.clip(
                                    axis_half + anc["obj_half"] + DEST_RELATION_MARGIN,
                                    0.06,
                                    0.16,
                                )
                            )
                        dest_xy = (dest_xy + np.asarray(cxy) + offset) / 2
                        entry["dest_from_anchor"] = dest_xy.round(4).tolist()
            else:
                ext = self._measure_destination_extent(
                    dest_desc, np.r_[eef[:2] + held, plane_z], plane_z
                )
                if ext is not None:
                    cxy, hx, hy = ext
                    entry["dest_outline"] = np.asarray(cxy).round(4).tolist()
                    if float(np.linalg.norm(np.asarray(cxy) - dest_xy)) < 0.06:
                        dest_xy = (
                            (dest_xy + np.asarray(cxy)) / 2
                            if it == 0
                            else np.asarray(cxy)
                        )
            delta = dest_xy - (eef[:2] + held)
            entry["delta_m"] = delta.round(4).tolist()
            rec["iters"].append(entry)
            if (
                float(np.linalg.norm(delta)) < 0.006
                or float(np.linalg.norm(delta)) > 0.08
            ):
                break
            goal = np.array(
                [
                    np.clip(eef[0] + delta[0], *WORKSPACE_X),
                    np.clip(eef[1] + delta[1], *WORKSPACE_Y),
                ]
            )
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[goal, eef[2]],
                1.0,
                self.events,
                f"align_release:{it}",
                max_steps=15,
                tol=0.008,
            )
        return held

    def _held_xy(self):
        if self.held_offset3 is not None:
            return np.asarray(self.held_offset3[:2], dtype=float)
        return np.asarray(getattr(self, "held_offset", np.zeros(2)), dtype=float)

    def _servo_destination_contact(
        self, dest_desc, plane_z, iterations, phase, expected_world=None, max_jump=0.08
    ):
        """Wrist-image servo: move until the destination's receiving point lies under the HELD OBJECT
        (its vertical projection onto plane_z), not under the end-effector."""
        held_xy = self._held_xy()
        for it in range(iterations):
            eef = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            try:
                target_px = project_world_to_raw(
                    self.env,
                    "robot0_eye_in_hand",
                    np.r_[eef[:2] + held_xy, plane_z],
                    self.size,
                    context=self.context,
                )
            except RuntimeError:
                break
            full = observation_image(
                self.obs, "robot0_eye_in_hand_image", context=self.context
            )
            crop_bounds = None
            image = full
            if expected_world is not None:
                try:
                    expected_px = project_world_to_raw(
                        self.env,
                        "robot0_eye_in_hand",
                        np.r_[expected_world[:2], plane_z],
                        self.size,
                        context=self.context,
                    )
                    if 0 <= expected_px[0] <= 1000 and 0 <= expected_px[1] <= 1000:
                        image, crop_bounds = crop_from_normalized(
                            full, expected_px, 0.6
                        )
                except RuntimeError:
                    pass
            _, value = safe_ask_structured(
                self.grounder,
                image,
                wrist_center_prompt(dest_desc, self.instruction, "destination"),
                96,
            )
            dest_px = normalize_point(value.get("point"))
            if dest_px is not None and crop_bounds is not None:
                dest_px = crop_point_to_global(dest_px, crop_bounds, full.size)
            if (
                dest_px is not None
                and crop_bounds is None
                and (abs(dest_px[0] - 500) < 3)
                and (abs(dest_px[1] - 500) < 3)
            ):
                dest_px = None
            entry = {
                "phase": phase,
                "iteration": it,
                "target_px": target_px,
                "dest_px": dest_px,
                "cropped": crop_bounds is not None,
            }
            if dest_px is None:
                self.record["tool_calls"].append({"dest_servo": entry})
                break
            cam2world, focal, _ = camera_geometry(
                self.env, "robot0_eye_in_hand", self.size
            )
            try:
                d_world = intersect_horizontal_plane(
                    dest_px, plane_z, cam2world, focal, self.size, context=self.context
                )
                t_world = intersect_horizontal_plane(
                    target_px,
                    plane_z,
                    cam2world,
                    focal,
                    self.size,
                    context=self.context,
                )
            except RuntimeError:
                self.record["tool_calls"].append({"dest_servo": entry})
                break
            delta = (d_world - t_world)[:2]
            entry["delta_m"] = delta.tolist()
            self.record["tool_calls"].append({"dest_servo": entry})
            if float(np.linalg.norm(delta)) > max_jump:
                entry["rejected_jump"] = True
                break
            if float(np.linalg.norm(delta)) < PRECISE_TOL:
                break
            goal_xy = np.array(
                [
                    np.clip(eef[0] + delta[0], *WORKSPACE_X),
                    np.clip(eef[1] + delta[1], *WORKSPACE_Y),
                ]
            )
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[goal_xy, eef[2]],
                1.0,
                self.events,
                f"{phase}:{it}",
                max_steps=30,
            )
            moved = np.asarray(self.obs["robot0_eef_pos"], dtype=float)[:2]
            expected_world = (
                np.r_[moved + held_xy, plane_z] if expected_world is not None else None
            )
            if (
                float(np.linalg.norm(goal_xy - moved)) > 0.02
                or float(np.linalg.norm(goal_xy - (eef[:2] + delta))) > 0.005
            ):
                entry["move_limited"] = True
                break

    def _measure_destination_box(self, dest_desc, expected_world, plane_z):
        """Wrist-view destination center from its full outline (bbox), tolerant to partial occlusion by the
        placed object. Returns world xy or None."""
        full = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        try:
            exp_px = project_world_to_raw(
                self.env,
                "robot0_eye_in_hand",
                np.r_[expected_world[:2], plane_z],
                self.size,
                context=self.context,
            )
        except RuntimeError:
            return None
        if not (0 <= exp_px[0] <= 1000 and 0 <= exp_px[1] <= 1000):
            return None
        crop, bounds = crop_from_normalized(full, exp_px, 0.7)
        prompt = f'Robot instruction: {self.instruction}\nThis wrist-camera crop looks down at the destination {dest_desc}. Return the bounding box of the WHOLE {dest_desc} outline, including any part covered by the object on it. JSON only, normalized 0..1000 within this crop: {{"bbox_2d":[x1,y1,x2,y2]}}.'
        _, value = safe_ask_structured(self.grounder, crop, prompt, 96)
        box = value.get("bbox_2d") or value.get("bbox") or value.get("box")
        box = normalize_box(box) if box is not None else None
        if box is None or not (
            60 < box[2] - box[0] < 950 and 60 < box[3] - box[1] < 950
        ):
            return None
        center_local = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        center_px = crop_point_to_global(center_local, bounds, full.size)
        cam2world, focal, _ = camera_geometry(self.env, "robot0_eye_in_hand", self.size)
        try:
            return intersect_horizontal_plane(
                center_px, plane_z, cam2world, focal, self.size, context=self.context
            )[:2]
        except RuntimeError:
            return None

    def _measure_released_object_contact(self, exp_obj):
        """Wrist-view position of the released object on its known top plane. Returns xy or None."""
        full = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        try:
            exp_px = project_world_to_raw(
                self.env, "robot0_eye_in_hand", exp_obj, self.size, context=self.context
            )
        except RuntimeError:
            return None
        if not (0 <= exp_px[0] <= 1000 and 0 <= exp_px[1] <= 1000):
            return None
        crop, bounds = crop_from_normalized(full, exp_px, 0.6)
        prompt = f'Robot instruction: {self.instruction}\nThis wrist-camera crop looks down at the released object {self.source}. Return the tight bounding box of the whole object. JSON only, normalized 0..1000 within this crop: {{"bbox_2d":[x1,y1,x2,y2]}}.'
        _, value = safe_ask_structured(self.grounder, crop, prompt, 96)
        box = value.get("bbox_2d") or value.get("bbox") or value.get("box")
        box = normalize_box(box) if box is not None else None
        cam2world, focal, _ = camera_geometry(self.env, "robot0_eye_in_hand", self.size)
        cands = []
        if (
            box is not None
            and 40 < box[2] - box[0] < 950
            and (40 < box[3] - box[1] < 950)
        ):
            c1 = crop_point_to_global([box[0], box[1]], bounds, full.size)
            c2 = crop_point_to_global([box[2], box[3]], bounds, full.size)
            try:
                w1 = intersect_horizontal_plane(
                    c1,
                    float(exp_obj[2]),
                    cam2world,
                    focal,
                    self.size,
                    context=self.context,
                )[:2]
                w2 = intersect_horizontal_plane(
                    c2,
                    float(exp_obj[2]),
                    cam2world,
                    focal,
                    self.size,
                    context=self.context,
                )[:2]
                ow = abs(float(w2[1] - w1[1]))
                ref_w = getattr(self, "width_finger", None)
                if ref_w is None or 0.7 < ow / max(ref_w, 0.001) < 1.3:
                    cands.append((w1 + w2) / 2)
            except RuntimeError:
                pass
        prompt2 = f'Robot instruction: {self.instruction}\nThis wrist-camera crop looks down at the released object {self.source}. Locate its visual center (for a container, the center of its opening). JSON only, normalized 0..1000 within this crop: {{"point":[x,y]}}.'
        _, value2 = safe_ask_structured(self.grounder, crop, prompt2, 96)
        lp = normalize_point(value2.get("point"))
        if lp is not None:
            try:
                cands.append(
                    intersect_horizontal_plane(
                        crop_point_to_global(lp, bounds, full.size),
                        float(exp_obj[2]),
                        cam2world,
                        focal,
                        self.size,
                        context=self.context,
                    )[:2]
                )
            except RuntimeError:
                pass
        cands = [c for c in cands if float(np.linalg.norm(c - exp_obj[:2])) <= 0.12]
        self.last_measure_agreement = (
            float(np.linalg.norm(cands[0] - cands[1])) if len(cands) == 2 else None
        )
        if not cands:
            return None
        if len(cands) == 2 and float(np.linalg.norm(cands[0] - cands[1])) > 0.03:
            return cands[1]
        return np.mean(np.asarray(cands), axis=0)

    def _look_and_nudge_contact(
        self, dest_desc, dest_world, plane_z, release_eef, held_xy
    ):
        """After release: look down, measure where the object actually is (unoccluded, exact plane), compare
        with the destination; small error -> re-grasp at the same relative point and shift; large error (with
        budget) -> full re-place. Returns the record incl. the final measured error."""
        budget_left = int(self.env.budget - self.env.steps)
        rec = {"phase": "look_and_nudge", "budget_left": budget_left}
        self.record["tool_calls"].append({"nudge": rec})
        self.last_place_record = rec
        if budget_left < 110:
            rec["skipped"] = "budget"
            return rec
        if (
            int(self.env.budget) >= 500
            and len(self.perception_log) == 1
            and (budget_left < 330)
        ):
            rec["skipped"] = "first object of a two-object task: budget reserved"
            return rec
        exp_xy = release_eef[:2] + held_xy
        probe_xy = (
            exp_xy
            - np.asarray(getattr(self, "planned_offset", np.zeros(2)), dtype=float)
            if self.current_strategy == "rim"
            else exp_xy
        )
        self.obs, top_contact = probe_top_height(
            self.env,
            self.obs,
            probe_xy,
            float(release_eef[2]) + 0.03,
            self.support_z,
            self.events,
            "nudge_probe",
            action_z=-0.6,
        )
        rec["probe_top_z"] = top_contact
        if top_contact is None or top_contact < plane_z + 0.008:
            rec["skipped"] = "probe found no object where it was released"
            return rec
        obj_top_z = float(top_contact)
        full = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        exp_obj = np.r_[exp_xy, obj_top_z]
        try:
            exp_px = project_world_to_raw(
                self.env, "robot0_eye_in_hand", exp_obj, self.size, context=self.context
            )
        except RuntimeError:
            rec["skipped"] = "object projection"
            return
        obj_xy = self._measure_released_object(exp_obj)
        if obj_xy is None:
            rec["skipped"] = "no trusted object measurement"
            return rec
        rec["obj_xy"] = obj_xy.round(4).tolist()
        rec["expected_obj_xy"] = exp_obj[:2].round(4).tolist()
        rec["measurement_agreement_m"] = getattr(self, "last_measure_agreement", None)
        if float(np.linalg.norm(obj_xy - exp_obj[:2])) > 0.12:
            rec["skipped"] = "object estimate implausible"
            return rec
        dest_xy = np.asarray(dest_world[:2], dtype=float)
        if self.dest_anchor is not None:
            anc = self.dest_anchor
            box_xy = self._measure_destination_box(
                anc["desc"], np.r_[np.asarray(anc["world"][:2]), plane_z], plane_z
            )
            rec["anchor_box_xy"] = (
                None if box_xy is None else np.asarray(box_xy).round(4).tolist()
            )
            if (
                box_xy is not None
                and float(
                    np.linalg.norm(np.asarray(box_xy) - np.asarray(anc["world"][:2]))
                )
                < 0.04
            ):
                dest_xy = (dest_xy + np.asarray(box_xy) + anc["offset"]) / 2
        else:
            box_xy = self._measure_destination_box(dest_desc, dest_world, plane_z)
            rec["dest_box_xy"] = (
                None if box_xy is None else np.asarray(box_xy).round(4).tolist()
            )
            if (
                box_xy is not None
                and float(np.linalg.norm(np.asarray(box_xy) - dest_xy)) < 0.04
            ):
                dest_xy = (dest_xy + np.asarray(box_xy)) / 2
        err = dest_xy - obj_xy
        rec["error_xy"] = err.round(4).tolist()
        rec["error_cm"] = round(float(np.linalg.norm(err)) * 100, 1)
        norm = float(np.linalg.norm(err))
        if norm < 0.015:
            rec["skipped"] = "within 1.5cm (below measurement noise)"
            return rec
        if (
            rec.get("measurement_agreement_m") is not None
            and rec["measurement_agreement_m"] > 0.025
        ):
            rec["skipped"] = (
                "outline and point estimates disagree; position not trusted"
            )
            return rec
        if norm > 0.3:
            rec["skipped"] = "error implausible"
            return rec
        big = norm > 0.1
        if big and budget_left < 150:
            rec["skipped"] = "large error, not enough budget to re-place"
            return rec
        grasp_xy = obj_xy - np.asarray(
            getattr(self, "planned_offset", np.zeros(2)), dtype=float
        )
        grasp_z = float(obj_top_z) - (
            STRATEGY_DROP["rim"]
            if self.current_strategy == "rim"
            else STRATEGY_DROP.get(self.current_strategy, 0.02)
        )
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[grasp_xy, grasp_z + 0.07],
            -1.0,
            self.events,
            "nudge_pregrasp",
            max_steps=25,
            tol=0.012,
        )
        self.obs, reached, contact = descend_hold_xy(
            self.env,
            self.obs,
            np.r_[grasp_xy, grasp_z],
            -1.0,
            self.events,
            "nudge_descend",
        )
        self.obs = hold(self.env, self.obs, 1.0, 12, self.events, "nudge_close")
        gap = abs(
            self.obs["robot0_gripper_qpos"][0] - self.obs["robot0_gripper_qpos"][1]
        )
        rec["regrasp_gap"] = float(gap)
        cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        lift_z = grasp_z + (0.12 if big else 0.035)
        if gap >= 0.003:
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[cur[:2], grasp_z + 0.035],
                1.0,
                self.events,
                "nudge_lift",
                max_steps=15,
                tol=0.01,
            )
            self.obs = hold(self.env, self.obs, 1.0, 3, self.events, "nudge_hold")
            gap2 = abs(
                self.obs["robot0_gripper_qpos"][0] - self.obs["robot0_gripper_qpos"][1]
            )
            rec["regrasp_gap_after_lift"] = float(gap2)
            if gap2 < 0.003 or gap2 < 0.5 * gap:
                gap = 0.0
        if gap < 0.003:
            self.obs = hold(self.env, self.obs, -1.0, 6, self.events, "nudge_reopen")
            rec["skipped"] = "regrasp empty or dropped"
            rec["touched"] = True
            rec["error_cm"] = None
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[grasp_xy, grasp_z + 0.1],
                -1.0,
                self.events,
                "nudge_retreat",
                max_steps=20,
                tol=0.02,
            )
            new_xy = self._measure_released_object(np.r_[obj_xy, obj_top_z])
            if new_xy is not None:
                rec["final_error_cm"] = round(
                    float(np.linalg.norm(dest_xy - new_xy)) * 100, 1
                )
            return rec
        if big:
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[cur[:2], lift_z],
                1.0,
                self.events,
                "replace_lift",
                max_steps=20,
                tol=0.015,
            )
        goal_xy = np.array(
            [
                np.clip(cur[0] + err[0], *WORKSPACE_X),
                np.clip(cur[1] + err[1], *WORKSPACE_Y),
            ]
        )
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[goal_xy, lift_z],
            1.0,
            self.events,
            "nudge_shift",
            max_steps=30 if big else 25,
            tol=0.006,
        )
        self.obs, _, _ = descend_hold_xy(
            self.env,
            self.obs,
            np.r_[goal_xy, grasp_z + 0.004],
            1.0,
            self.events,
            "nudge_lower",
            max_steps=30 if big else 15,
        )
        self.obs = hold(self.env, self.obs, -1.0, 10, self.events, "nudge_release")
        self.obs = move_to_precise(
            self.env,
            self.obs,
            np.r_[goal_xy, grasp_z + 0.1],
            -1.0,
            self.events,
            "nudge_retreat",
            max_steps=20,
            tol=0.02,
        )
        rec["nudged"] = True
        rec["replaced"] = big
        new_xy = self._measure_released_object(np.r_[goal_xy + held_xy, obj_top_z])
        if new_xy is not None:
            rec["final_error_cm"] = round(
                float(np.linalg.norm(dest_xy - new_xy)) * 100, 1
            )
        return rec

    def _retry_place(self, args):
        if self.held:
            return self._place_object(
                {
                    "destination_description": self.last_dest_desc,
                    "is_container": self.last_is_container,
                }
            )
        self.top_z = None
        self.rim_z = None
        self.servo_plane_done = None
        self.probe_done = False
        return self._retry_place_tool(args)

    def _retry_place_tool(self, args):
        """Re-grasp the just-released object and re-place it."""
        if self.held:
            return ("Error: object is still held. Call place_object instead.", None)
        self.place_retries = getattr(self, "place_retries", 0) + 1
        if self.place_retries > 1:
            return ("Max place retries reached. Call done.", None)
        if int(self.env.budget - self.env.steps) < 130:
            return (
                "Not enough step budget left to re-grasp and re-place. Call done.",
                None,
            )
        recovery_desc = f"the {self.source} just released"
        try:
            _, source_world, surface_z = locate_relational_source(
                self.env,
                self.obs,
                self.grounder,
                recovery_desc,
                self.instruction,
                self.size,
                self.support_z,
                context=self.context,
            )
            self.source_world = source_world
            self.source_surface_z = surface_z
            self.obs = move_to(
                self.env,
                self.obs,
                np.r_[source_world[:2], self.travel_z],
                -1.0,
                self.events,
                "retry_approach",
                max_steps=45,
            )
            self.obs, self.wrist_center = wrist_servo(
                self.env,
                self.obs,
                self.grounder,
                self.source,
                self.instruction,
                "source",
                source_world,
                self.support_z,
                self.travel_z,
                -1.0,
                self.size,
                self.events,
                [],
                context=self.context,
            )
            self.grasp_attempt = 0
            result_text, result_image = self._execute_grasp(self.current_strategy)
            if not self.held:
                return (f"Re-grasp failed: {result_text}. Call done.", None)
            return self._place_object(
                {
                    "destination_description": self.last_dest_desc,
                    "is_container": self.last_is_container,
                }
            )
        except RuntimeError as e:
            return (f"Retry failed: {e}. Call done.", None)
