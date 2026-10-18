"""skills / pick_place / grasp: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import (
    descend_hold_xy,
    hold,
    in_workspace,
    move_to,
    move_to_precise,
    probe_top_height,
)
from depthrobotic.geometry.camera import (
    camera_geometry,
    deproject,
    intersect_horizontal_plane,
    project_world_to_raw,
)
from depthrobotic.geometry.depth import depth_scale_to_workplane, sample_point_depth
from depthrobotic.perception.images import (
    annotate_candidates,
    crop_from_normalized,
    crop_point_to_global,
    observation_image,
    upright,
)
from depthrobotic.perception.matching import flip_point
from depthrobotic.perception.parsing import normalize_point, safe_ask_structured
from depthrobotic.skills.pick_place.locate import (
    ADJUSTMENT_OFFSETS,
    FINGER_AXIS,
    GRIPPER_OPENING,
    MAX_FEASIBLE_WIDTH,
    RETRY_HEIGHT,
    RIM_INSET,
    STRATEGY_DROP,
)
from depthrobotic.skills.pick_place.state import region_descriptor
from depthrobotic.skills.pick_place.wrist import (
    box_metrics,
    confirm_grasp,
    estimate_held_offset_3d,
    measure_object_box,
    wrist_servo,
)
import numpy as np
import torch


class Grasping:

    def _wrist_descriptor(self):
        box, _ = measure_object_box(
            self.grounder, self.obs, self.source, self.instruction, context=self.context
        )
        if box is None:
            return None
        image = upright(
            observation_image(
                self.obs, "robot0_eye_in_hand_image", context=self.context
            ),
            context=self.context,
        )
        if not self.context.settings.orient_fix:
            box = [box[0], 1000 - box[3], box[2], 1000 - box[1]]
        return region_descriptor(self.context.models.features.features(image), box)

    def _execute_grasp(self, strategy, height_adjust=0.0):
        before = self._wrist_descriptor() if self.wrist_center is not None else None
        text, image = self._execute_grasp_contact(strategy, height_adjust)
        if self.held:
            after = self._wrist_descriptor()
            score = (
                float(torch.dot(before, after))
                if before is not None and after is not None
                else None
            )
            self.state.held = (
                "HELD_SUPPORTED"
                if score is not None and score >= 0.65
                else "HELD_UNCONFIRMED"
            )
            self.held_descriptor = after
            self.release_descriptor = after
            offset = self._held_xy()
            self.state.held_offset = np.asarray(offset).tolist()
            self.state.source.association = (
                "MATCHED" if score is not None and score >= 0.65 else "UNKNOWN"
            )
            self.state.record(
                "lift",
                self.env.steps,
                appearance_similarity=score,
                gripper_gap=self._gripper_gap(),
                held_offset=self.state.held_offset,
            )
            if self.state.held == "HELD_UNCONFIRMED":
                text += " Visual identity through the lift is uncertain; retain the named source reference during placement."
        else:
            self.state.held = "NOT_HELD"
            self.state.source.invalidate("grasp_failed")
        self._save()
        return (text, image)

    def _retry_grasp(self, args):
        kind = (getattr(self, "last_failure", None) or {}).get("kind")
        if kind in {"empty", "stall_high", "motion"}:
            self.state.source.invalidate(kind)
            self._source_match()
        start = len(self.record["tool_calls"])
        text, image = self._retry_grasp_contact(args)
        applied = [
            c["recovery"] for c in self.record["tool_calls"][start:] if "recovery" in c
        ]
        if applied:
            a = applied[-1]
            text += f" Executed recovery: {a.get('executor_recovery') or 'requested adjustment'}, strategy={a.get('applied_strategy')}, dz={a.get('dz', 0):.3f}m."
        self.state.record(
            "recovery", self.env.steps, requested=dict(args), applied=applied
        )
        self._save()
        return (text, image)

    def _repoint_source(self, plane_z):
        world = self._source_match()
        if world is not None:
            px = flip_point(
                self.state.source.point, not self.context.settings.orient_fix
            )
            cam, focal, _ = camera_geometry(self.env, "agentview", self.size)
            xy = intersect_horizontal_plane(
                px, plane_z, cam, focal, self.size, context=self.context
            )[:2]
            if (
                in_workspace(xy)
                and np.linalg.norm(xy - np.asarray(self.obs["robot0_eef_pos"])[:2])
                < 0.12
            ):
                return xy
        return self._repoint_source_contact(plane_z)

    def _gripper_gap(self):
        gq = self.obs["robot0_gripper_qpos"]
        return abs(float(gq[0]) - float(gq[1]))

    def _retry_grasp_contact(self, args):
        adj = args.get("adjustment", "higher")
        strategy = args.get("strategy", self.current_strategy)
        fail = getattr(self, "last_failure", None) or {}
        kind = fail.get("kind")
        hist = getattr(self, "failure_history", [])
        n_same = sum((1 for h in hist if h.get("kind") == kind))
        rec = None
        dx, dy, dz = ADJUSTMENT_OFFSETS.get(adj, (0, 0, 0))
        model_changed = args.get("strategy") is not None and args.get(
            "strategy"
        ) != fail.get("strategy")
        reobserve = kind in ("empty", "stall_high", "motion") and n_same >= 2
        if kind in ("empty", "stall_high", "slipped") and n_same >= 2 and model_changed:
            rec = "model_strategy_change_after_executor_failure"
            dx = dy = 0.0
        elif kind == "empty":
            if n_same >= 3:
                rec = (
                    "switch_rim_side"
                    if fail.get("strategy") == "rim"
                    else "switch_to_rim_edge_pinch"
                )
                strategy = "rim"
            elif n_same == 2:
                rec = "re_observe"
            else:
                rec = (
                    "switch_rim_side"
                    if fail.get("strategy") == "rim"
                    else "recenter_on_fresh_measurement"
                )
            dx = dy = 0.0
        elif kind == "stall_high":
            wide = self.width_finger is not None and self.width_finger >= 0.06
            if n_same >= 3:
                rec = (
                    "switch_rim_side"
                    if fail.get("strategy") == "rim"
                    else "switch_to_rim_edge_pinch"
                )
                strategy = "rim"
            elif n_same == 2:
                rec = "re_observe"
                strategy = fail.get("strategy") or strategy
            elif fail.get("strategy") != "rim" and wide:
                rec = "switch_to_rim_edge_pinch"
                strategy = "rim"
            elif fail.get("strategy") != "rim":
                rec = "reopen_and_recenter"
                strategy = fail.get("strategy")
                self.servo_plane_done = None
            else:
                rec = "switch_rim_side"
            dx = dy = 0.0
        elif kind == "slipped":
            if fail.get("strategy") == "rim":
                rec = "regrasp_higher"
                dz = 0.01 * n_same
            else:
                rec = "regrasp_lower_on_body"
                dz = -0.035 * n_same
            dx = dy = 0.0
        if rec == "switch_rim_side":
            self.rim_side_pref = (
                "W" if getattr(self, "last_rim_label", "E") == "E" else "E"
            )
        if rec == "re_observe" or reobserve:
            plane_z = (
                self.top_z
                if strategy != "rim" and self.top_z is not None
                else max(
                    (
                        float(self.source_surface_z)
                        if self.source_surface_z is not None
                        else self.support_z
                    ),
                    self.support_z + 0.01,
                )
            )
            new_xy = self._repoint_source(plane_z)
            note = {"re_observe": None if new_xy is None else new_xy.round(4).tolist()}
            if (
                kind == "stall_high"
                and fail.get("stall_xy") is not None
                and (self.width_finger is not None)
            ):
                wx = float(getattr(self, "width_across", None) or self.width_finger)
                tact = np.asarray(fail["stall_xy"], dtype=float) + np.array(
                    [-0.5 * wx, 0.0]
                )
                note["tactile_center"] = tact.round(4).tolist()
                if new_xy is None or float(np.linalg.norm(new_xy - tact)) > 0.04:
                    new_xy = tact if new_xy is None else (new_xy + tact) / 2
                    note["used"] = (
                        "tactile" if note.get("re_observe") is None else "blend"
                    )
            if new_xy is not None:
                cam2world, focal, _ = camera_geometry(
                    self.env, "robot0_eye_in_hand", self.size
                )
                try:
                    self.wrist_center = project_world_to_raw(
                        self.env,
                        "robot0_eye_in_hand",
                        np.r_[new_xy, plane_z],
                        self.size,
                        context=self.context,
                    )
                except RuntimeError:
                    pass
            self.record["tool_calls"].append(note)
            self.servo_plane_done = None
        self.record["tool_calls"].append(
            {
                "recovery": {
                    "failure": kind,
                    "n_same": n_same,
                    "model_adjustment": adj,
                    "model_strategy": args.get("strategy"),
                    "executor_recovery": rec,
                    "applied_strategy": strategy,
                    "dz": dz,
                }
            }
        )
        self.current_strategy = strategy
        self._pending_xy_shift = np.array([dx, dy])
        return self._execute_grasp(strategy, height_adjust=dz)

    def _measure_plane(self, plane_z):
        """Re-servo at the object's own height plane (removes the table-plane parallax bias for tall
        objects) and measure the wrist bounding box there."""
        center_world = intersect_horizontal_plane(
            self.wrist_center,
            plane_z,
            *camera_geometry(self.env, "robot0_eye_in_hand", self.size)[:2],
            self.size,
            context=self.context,
        )
        if plane_z - self.support_z > 0.03 and self.servo_plane_done != round(
            plane_z, 3
        ):
            try:
                self.obs, refreshed = wrist_servo(
                    self.env,
                    self.obs,
                    self.grounder,
                    self.source,
                    self.instruction,
                    "source",
                    np.r_[center_world[:2], plane_z],
                    plane_z,
                    self.travel_z,
                    -1.0,
                    self.size,
                    self.events,
                    [],
                    iterations=2,
                    max_anchor_deviation_override=0.05,
                    context=self.context,
                )
                if refreshed is not None:
                    self.wrist_center = refreshed
                    center_world = intersect_horizontal_plane(
                        self.wrist_center,
                        plane_z,
                        *camera_geometry(self.env, "robot0_eye_in_hand", self.size)[:2],
                        self.size,
                        context=self.context,
                    )
            except RuntimeError as e:
                self.record["tool_calls"].append({"plane_servo_error": str(e)[:100]})
            self.servo_plane_done = round(plane_z, 3)
        box, resp = measure_object_box(
            self.grounder, self.obs, self.source, self.instruction, context=self.context
        )
        self.box = box
        entry = {"plane_z": plane_z, "box": box, "box_response": resp[:100]}
        if box is not None:
            wf, wa, box_center = box_metrics(
                self.env, self.size, box, plane_z, context=self.context
            )
            self.width_finger = wf
            self.width_across = wa
            entry.update(
                {
                    "width_finger_axis_m": wf,
                    "width_across_m": wa,
                    "box_center_vs_servo_m": float(
                        np.linalg.norm(box_center - center_world[:2])
                    ),
                }
            )
            dc = float(np.linalg.norm(box_center - center_world[:2]))
            if dc < 0.03:
                center_world = np.r_[(box_center + center_world[:2]) / 2, plane_z]
            elif dc < 0.18:
                center_world = np.r_[box_center, plane_z]
                self.servo_plane_done = None
                entry["object_moved_trust_box"] = True
                entry["box_touches_border"] = bool(
                    min(box[0], box[1]) <= 2 or max(box[2], box[3]) >= 998
                )
            if self.top_z is not None and self.top_z - self.support_z > 0.04:
                tf = self._depth_top_face_center(plane_z, box)
                if (
                    tf is not None
                    and float(np.linalg.norm(tf - center_world[:2])) < 0.06
                ):
                    entry["top_face_shift_m"] = (
                        (tf - center_world[:2]).round(4).tolist()
                    )
                    center_world = np.r_[tf, plane_z]
        else:
            self.width_finger = None
        self.record["tool_calls"].append({"measure": entry})
        return center_world

    def _choose_rim_side(self, center_world, r, plane_z):
        """Two candidates on the finger axis; the model picks the one with free space."""
        cands_world = {
            "E": np.r_[center_world[:2] - r * FINGER_AXIS, plane_z],
            "W": np.r_[center_world[:2] + r * FINGER_AXIS, plane_z],
        }
        cands_px = {}
        for label, w in cands_world.items():
            try:
                px = project_world_to_raw(
                    self.env, "robot0_eye_in_hand", w, self.size, context=self.context
                )
                cands_px[label] = [
                    float(np.clip(px[0], 15, 985)),
                    float(np.clip(px[1], 15, 985)),
                ]
            except RuntimeError:
                pass
        label = "E"
        pref = getattr(self, "rim_side_pref", None)
        if pref in cands_world:
            label = pref
            self.rim_side_pref = None
            self.record["tool_calls"].append(
                {
                    "rim_candidates_px": cands_px,
                    "rim_label": label,
                    "rim_radius_m": r,
                    "forced_by_recovery": True,
                }
            )
            self.last_rim_label = label
            return cands_world[label]
        if len(cands_px) == 2:
            marked = annotate_candidates(
                observation_image(
                    self.obs, "robot0_eye_in_hand_image", context=self.context
                ),
                cands_px,
            )
            prompt = f'Robot instruction: {self.instruction}\nThe two red labeled points E and W mark where the gripper fingers can pinch the wall/edge of {self.source}. Pick the one with free space around it (no other object, no obstacle, not blocked by the gripper). Return JSON only: {{"label":"E"}} or {{"label":"W"}}.'
            _, value = safe_ask_structured(self.grounder, marked, prompt, 48)
            chosen = str(value.get("label", "")).upper().strip()
            if chosen in cands_world:
                label = chosen
        self.record["tool_calls"].append(
            {"rim_candidates_px": cands_px, "rim_label": label, "rim_radius_m": r}
        )
        self.last_rim_label = label
        return cands_world[label]

    def _execute_grasp_contact(self, strategy, height_adjust=0.0):
        if self.grasp_attempt >= self.context.settings.max_grasp_attempts:
            raise RuntimeError(
                f"Max grasp attempts ({self.context.settings.max_grasp_attempts}) exceeded for '{self.source}'"
            )
        if self.wrist_center is None:
            raise RuntimeError("No wrist center -- call approach_source first")
        if self.grasp_attempt > 0 and int(self.env.budget - self.env.steps) < 140:
            return (
                "Not enough step budget left for another grasp attempt plus placement. Call done.",
                None,
            )
        strategy = strategy if strategy in STRATEGY_DROP else "center"
        self.current_strategy = strategy
        cam2world, focal, _ = camera_geometry(self.env, "robot0_eye_in_hand", self.size)
        center_table = intersect_horizontal_plane(
            self.wrist_center,
            self.support_z,
            cam2world,
            focal,
            self.size,
            context=self.context,
        )
        surface_est = (
            float(self.source_surface_z)
            if self.source_surface_z is not None
            else self.support_z
        )
        budget_left = int(self.env.budget - self.env.steps)
        if (
            strategy != "rim"
            and self.top_z is None
            and (not self.probe_done)
            and (budget_left > 90)
        ):
            self.obs, contact = probe_top_height(
                self.env,
                self.obs,
                center_table[:2],
                self.travel_z,
                self.support_z,
                self.events,
                f"probe:{self.grasp_attempt}",
                close_first=True,
                open_after=strategy != "rim",
            )
            self.fingers_closed = False
            self.top_z = contact
            self.probe_done = True
            self.record["tool_calls"].append(
                {
                    "probe_center_top_z": contact,
                    "top_above_support_m": (
                        None if contact is None else contact - self.support_z
                    ),
                }
            )
        plane_z = (
            self.top_z
            if strategy != "rim" and self.top_z is not None
            else max(surface_est, self.support_z + 0.01)
        )
        if strategy == "rim" and self.top_z is not None:
            plane_z = max(plane_z, self.top_z + 0.02)
        center_world = self._measure_plane(plane_z)
        if (
            strategy != "rim"
            and self.width_finger is not None
            and (self.width_finger > MAX_FEASIBLE_WIDTH)
        ):
            self.infeasible_count += 1
            self.last_failure = {"kind": "infeasible", "strategy": strategy}
            if self.infeasible_count <= 1:
                return (
                    f"The object measures {self.width_finger * 100:.0f} cm along the gripper's closing axis, wider than the {GRIPPER_OPENING * 100:.0f} cm opening, so a '{strategy}' pinch cannot enclose it. Call grasp again with strategy 'rim' to pinch its wall/edge instead.",
                    None,
                )
            strategy = "rim"
            self.current_strategy = "rim"
            plane_z = max(
                surface_est, self.top_z or self.support_z, self.support_z + 0.01
            )
        pending = getattr(self, "_pending_xy_shift", np.zeros(2))
        self._pending_xy_shift = np.zeros(2)
        attempt_idx = min(self.grasp_attempt, 2)
        if strategy == "rim":
            r = self.width_finger / 2 - RIM_INSET if self.width_finger else 0.04
            r = float(np.clip(r, 0.02, 0.09))
            grasp_world = self._choose_rim_side(center_world, r, plane_z)
            grasp_world[:2] += pending
            self.planned_offset = center_world[:2] - grasp_world[:2]
            if self.rim_z is None and budget_left > 70:
                start_z = max(
                    (self.top_z or self.support_z) + 0.08,
                    surface_est + 0.05,
                    self.support_z + 0.08,
                )
                for shrink in (0.0,):
                    probe_xy = grasp_world[:2] + shrink * (
                        center_world[:2] - grasp_world[:2]
                    ) / max(r, 1e-06)
                    self.obs, rim_contact = probe_top_height(
                        self.env,
                        self.obs,
                        probe_xy,
                        start_z,
                        self.support_z,
                        self.events,
                        f"probe_rim:{self.grasp_attempt}",
                        close_first=not self.fingers_closed,
                        open_after=False,
                        action_z=-0.5,
                    )
                    self.fingers_closed = True
                    valid = (
                        rim_contact is not None
                        and rim_contact > self.support_z - 0.04
                        and (self.top_z is None or rim_contact > self.top_z + 0.012)
                    )
                    self.record["tool_calls"].append(
                        {
                            "probe_rim_z": rim_contact,
                            "rim_probe_valid": valid,
                            "shrink": shrink,
                        }
                    )
                    if valid:
                        self.rim_z = rim_contact
                        if shrink:
                            grasp_world[:2] = probe_xy
                            self.planned_offset = center_world[:2] - grasp_world[:2]
                        break
            if self.fingers_closed:
                cur = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
                self.obs = move_to_precise(
                    self.env,
                    self.obs,
                    np.r_[cur[:2], cur[2] + 0.06],
                    -1.0,
                    self.events,
                    f"probe_rim:{self.grasp_attempt}:retreat",
                    max_steps=20,
                    tol=0.015,
                )
                self.obs = hold(
                    self.env,
                    self.obs,
                    -1.0,
                    4,
                    self.events,
                    f"probe_rim:{self.grasp_attempt}:open",
                )
                self.fingers_closed = False
            top_ref = (
                self.rim_z
                if self.rim_z is not None
                else max(surface_est, self.support_z + 0.03)
            )
            base_z = top_ref - STRATEGY_DROP["rim"]
        else:
            grasp_world = np.r_[center_world[:2] + pending, plane_z]
            self.planned_offset = np.zeros(2)
            if self.top_z is not None:
                base_z = self.top_z - STRATEGY_DROP[strategy]
            else:
                depth_height = float(
                    np.clip(
                        surface_est - self.support_z,
                        -0.025 if strategy == "top" else 0.0,
                        0.16,
                    )
                )
                base_z = (
                    self.support_z + depth_height + (0.0, -0.035, 0.035)[attempt_idx]
                )
        self.grasp_world = grasp_world
        z = base_z + RETRY_HEIGHT[strategy][attempt_idx] + height_adjust
        floor = self.support_z - 0.05
        z = max(z, floor)
        if self.top_z is not None and strategy != "rim":
            z = min(z, self.top_z - 0.005)
        self.grasp_eef_z = float(z)
        self.record["tool_calls"].append(
            {
                "grasp_plan": {
                    "strategy": strategy,
                    "xy": grasp_world[:2].tolist(),
                    "z": self.grasp_eef_z,
                    "z_above_support": self.grasp_eef_z - self.support_z,
                    "planned_offset": self.planned_offset.tolist(),
                }
            }
        )
        try:
            for _ in range(3):
                gq = self.obs["robot0_gripper_qpos"]
                if abs(float(gq[0]) - float(gq[1])) > 0.07:
                    break
                self.obs = hold(
                    self.env,
                    self.obs,
                    -1.0,
                    5,
                    self.events,
                    f"ensure_open:{self.grasp_attempt}",
                )
            self.fingers_closed = False
            if self.fingers_closed:
                self.obs = hold(
                    self.env,
                    self.obs,
                    -1.0,
                    6,
                    self.events,
                    f"open_before_pregrasp:{self.grasp_attempt}",
                )
                self.fingers_closed = False
            self.obs = move_to(
                self.env,
                self.obs,
                np.r_[grasp_world[:2], self.grasp_eef_z + 0.1],
                -1.0,
                self.events,
                f"pregrasp:{self.grasp_attempt}",
                max_steps=35,
            )
        except RuntimeError as e:
            return (f"Motion failed during grasp approach: {e}", None)
        self.obs, reached, contact = descend_hold_xy(
            self.env,
            self.obs,
            np.r_[grasp_world[:2], self.grasp_eef_z],
            -1.0,
            self.events,
            f"descend:{self.grasp_attempt}",
        )
        remaining_z = self.grasp_eef_z - float(
            np.asarray(self.obs["robot0_eef_pos"], dtype=float)[2]
        )
        if not reached and (not contact):
            try:
                self.obs = move_to(
                    self.env,
                    self.obs,
                    np.r_[grasp_world[:2], self.travel_z],
                    -1.0,
                    self.events,
                    f"recover:{self.grasp_attempt}",
                    max_steps=35,
                )
            except RuntimeError:
                pass
            self.last_failure = {"kind": "motion", "strategy": strategy}
            self.failure_history.append(
                {
                    "kind": "motion",
                    "strategy": strategy,
                    "note": f"stopped {abs(remaining_z) * 100:.1f}cm above target",
                }
            )
            return (
                f"Motion failed during grasp approach: descent stopped {abs(remaining_z) * 100:.1f} cm above the grasp height without contact."
                + self._history_text(),
                None,
            )
        if contact and (
            remaining_z < -0.02 or (strategy == "rim" and remaining_z < -0.012)
        ):
            try:
                self.obs = move_to(
                    self.env,
                    self.obs,
                    np.r_[grasp_world[:2], self.travel_z],
                    -1.0,
                    self.events,
                    f"recover:{self.grasp_attempt}",
                    max_steps=35,
                )
            except RuntimeError:
                pass
            self.grasp_attempt += 1
            self.last_failure = {
                "kind": "stall_high",
                "strategy": strategy,
                "stall_z": float(np.asarray(self.obs["robot0_eef_pos"])[2]),
                "stall_xy": np.asarray(self.obs["robot0_eef_pos"], dtype=float)[
                    :2
                ].tolist(),
            }
            self.failure_history.append(
                {
                    "kind": "stall_high",
                    "strategy": strategy,
                    "note": f"fingers stalled {abs(remaining_z) * 100:.1f}cm above target",
                }
            )
            return (
                f"Grasp FAILED with strategy '{strategy}': the open fingers landed on top of something {abs(remaining_z) * 100:.1f} cm above the planned height (misaligned, wider than expected, or blocked)."
                + self._history_text()
                + " Call retry_grasp.",
                None,
            )
        self.grasp_attempt += 1
        self.obs = hold(
            self.env, self.obs, 1.0, 15, self.events, f"close:{self.grasp_attempt}"
        )
        self.obs = move_to(
            self.env,
            self.obs,
            np.r_[grasp_world[:2], self.travel_z],
            1.0,
            self.events,
            f"lift:{self.grasp_attempt}",
            max_steps=35,
        )
        self.obs = hold(
            self.env, self.obs, 1.0, 4, self.events, f"hold:{self.grasp_attempt}"
        )
        self.held, confirmation = confirm_grasp(
            self.grounder, self.obs, self.source, self.instruction, context=self.context
        )
        gap = confirmation.get("grip_gap", 0)
        close_gap = None
        for e in reversed(self.events):
            if e.get("phase") == f"close:{self.grasp_attempt}" and "gripper_qpos" in e:
                close_gap = abs(e["gripper_qpos"][0] - e["gripper_qpos"][1])
                break
        hold_gap = 2 * gap
        if (
            self.held
            and close_gap is not None
            and (close_gap > 0.006)
            and (hold_gap < 0.55 * close_gap)
            and (hold_gap < 0.005)
        ):
            self.record["tool_calls"].append(
                {"slip_detected": {"close_gap": close_gap, "hold_gap": hold_gap}}
            )
            self.last_failure = {"kind": "slipped", "strategy": strategy}
            self.failure_history.append(
                {
                    "kind": "slipped",
                    "strategy": strategy,
                    "note": f"pinched gap {close_gap:.3f} then slipped",
                }
            )
            self.held = False
            self.obs = hold(
                self.env, self.obs, -1.0, 8, self.events, f"reopen:{self.grasp_attempt}"
            )
            remaining = self.context.settings.max_grasp_attempts - self.grasp_attempt
            return (
                f"Grasp FAILED with strategy '{strategy}': the object was pinched (gap {close_gap:.3f}) but slipped out during lift (gap {hold_gap:.3f}). {remaining} retries remaining; a shallower/higher grip or a different point may hold better. Call retry_grasp.",
                None,
            )
        if self.held:
            self.last_failure = None
            self.held_offset = np.array(self.planned_offset, dtype=float).copy()
            offset3, rec = estimate_held_offset_3d(
                self.env,
                self.obs,
                self.grounder,
                self.source,
                self.instruction,
                self.size,
                self.support_z,
                context=self.context,
            )
            self.held_offset3 = offset3
            self.record["tool_calls"].append({"held_offset_measure": rec})
            return (
                f"Grasp SUCCESS with strategy '{strategy}'. Gripper gap: {gap:.4f}. Object is held. Now call place_object to transport and place it.",
                None,
            )
        self.obs = hold(
            self.env, self.obs, -1.0, 8, self.events, f"reopen:{self.grasp_attempt}"
        )
        remaining = self.context.settings.max_grasp_attempts - self.grasp_attempt
        self.last_failure = {"kind": "empty", "strategy": strategy}
        self.failure_history.append(
            {"kind": "empty", "strategy": strategy, "note": "closed on nothing"}
        )
        return (
            f"Grasp FAILED with strategy '{strategy}'. Gripper gap: {gap:.4f} (empty grasp). {remaining} retries remaining."
            + self._history_text()
            + " Call retry_grasp with an adjustment.",
            None,
        )

    def _repoint_source_contact(self, plane_z):
        """Second same-kind failure: get a fresh observation instead of re-using the biased centre. The model points
        at the source in the current wrist image (crop around the expected position); returns world xy or None.
        """
        full = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        eef = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        try:
            exp_px = project_world_to_raw(
                self.env,
                "robot0_eye_in_hand",
                np.r_[eef[:2], plane_z],
                self.size,
                context=self.context,
            )
        except RuntimeError:
            return None
        if not (0 <= exp_px[0] <= 1000 and 0 <= exp_px[1] <= 1000):
            exp_px = [500.0, 400.0]
        crop, bounds = crop_from_normalized(full, exp_px, 0.7)
        prompt = f'Robot instruction: {self.instruction}\nThe previous grasp of {self.source} missed. In this wrist-camera crop, locate the visual centre of the TOP FACE of {self.source} (the surface nearest the camera, not its side). JSON only, normalized 0..1000 within this crop: {{"point":[x,y]}}.'
        _, value = safe_ask_structured(self.grounder, crop, prompt, 96)
        lp = normalize_point(value.get("point"))
        if lp is None:
            return None
        px = crop_point_to_global(lp, bounds, full.size)
        cam2world, focal, _ = camera_geometry(self.env, "robot0_eye_in_hand", self.size)
        try:
            xy = intersect_horizontal_plane(
                px, plane_z, cam2world, focal, self.size, context=self.context
            )[:2]
        except RuntimeError:
            return None
        self.wrist_center = px
        return xy

    def _depth_top_face_center(self, plane_z, box):
        """Centre of the object's top face from DepthVLM's wrist depth: sample the bbox, keep points whose world z is
        within 2.5cm of the probed top, return their centroid. Removes the +x bias of 2D centres on tall objects.
        """
        image = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        cam2world, focal, _ = camera_geometry(self.env, "robot0_eye_in_hand", self.size)
        try:
            depth = self.grounder.predict_depth(image, focal)
            scale = depth_scale_to_workplane(
                depth, cam2world, focal, self.size, self.support_z, context=self.context
            )
        except Exception as e:
            self.record["tool_calls"].append(
                {"top_face_depth_error": f"{type(e).__name__}: {e}"[:100]}
            )
            return None
        x1, y1, x2, y2 = box
        pts = []
        for u in np.linspace(x1 + 5, x2 - 5, 22):
            for v in np.linspace(y1 + 5, y2 - 5, 22):
                try:
                    z_cam = sample_point_depth(depth, [u, v], radius=1) * scale
                    w = deproject(
                        [u, v], z_cam, cam2world, focal, self.size, context=self.context
                    )
                except Exception:
                    continue
                if abs(float(w[2]) - plane_z) < 0.025:
                    pts.append(w[:2])
        if len(pts) < 15:
            self.record["tool_calls"].append(
                {"top_face_depth": {"points": len(pts), "used": False}}
            )
            return None
        c = np.mean(np.asarray(pts), axis=0)
        self.record["tool_calls"].append(
            {
                "top_face_depth": {
                    "points": len(pts),
                    "center": c.round(4).tolist(),
                    "used": True,
                }
            }
        )
        return c

    def _history_text(self):
        hist = getattr(self, "failure_history", [])
        if not hist:
            return ""
        parts = [
            f"{i + 1}) {h.get('strategy')}: {h.get('kind')}"
            + (f" ({h.get('note')})" if h.get("note") else "")
            for i, h in enumerate(hist)
        ]
        return " Attempts so far: " + "; ".join(parts) + "."

    def _object_outline_px(self):
        """Centre (normalized px) and box of the source object's outline in the current wrist image, or (None, None)."""
        box, _ = measure_object_box(
            self.grounder, self.obs, self.source, self.instruction, context=self.context
        )
        if box is None:
            return (None, None)
        return (
            [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
            [float(b) for b in box],
        )

    def _object_point_px(self):
        """Model point on the released object (fallback tracker when no outline comes back), normalized px or None."""
        image = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        prompt = f'Robot instruction: {self.instruction}\nThe wrist camera looks down just after the gripper opened. Point to the visual centre of the {self.source} (the object that was just released), ignoring the gripper fingers. JSON only, normalized 0..1000: {{"point":[x,y]}}.'
        _, value = safe_ask_structured(self.grounder, image, prompt, 64)
        pt = normalize_point(value.get("point"))
        if pt is None or (abs(pt[0] - 500) < 3 and abs(pt[1] - 500) < 3):
            return None
        return [float(pt[0]), float(pt[1])]

    def _wrist_gray(self):
        img = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        ).convert("L")
        return np.asarray(img, dtype=np.float32)

    @staticmethod
    def _ncc(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return None if d < 1e-06 else round(float((a * b).sum() / d), 3)

    def _patch_similarity(self, gray0, gray1, box):
        """VLM-free auxiliary cue (logged only): correlation of the wrist image inside the object's old outline and
        outside it. An attached object keeps its pixels (inside high, outside low); a released one changes both.
        """
        if gray0 is None or gray1 is None:
            return None
        h, w = gray0.shape
        if box is None:
            x1, y1, x2, y2 = (0.25 * w, 0.15 * h, 0.75 * w, 0.55 * h)
        else:
            x1, y1, x2, y2 = (
                box[0] / 1000 * (w - 1),
                box[1] / 1000 * (h - 1),
                box[2] / 1000 * (w - 1),
                box[3] / 1000 * (h - 1),
            )
        x1, x2 = (
            int(np.clip(min(x1, x2), 0, w - 2)),
            int(np.clip(max(x1, x2), 1, w - 1)),
        )
        y1, y2 = (
            int(np.clip(min(y1, y2), 0, h - 2)),
            int(np.clip(max(y1, y2), 1, h - 1)),
        )
        mask = np.zeros_like(gray0, dtype=bool)
        mask[y1:y2, x1:x2] = True
        inside = self._ncc(gray0[y1:y2, x1:x2], gray1[y1:y2, x1:x2])
        outside = self._ncc(gray0[~mask], gray1[~mask]) if (~mask).sum() > 100 else None
        return {"inside": inside, "outside": outside, "region_default": box is None}

    def _grasp(self, args):
        strategy = args.get("strategy", "center")
        self.current_strategy = strategy
        return self._execute_grasp(strategy)
