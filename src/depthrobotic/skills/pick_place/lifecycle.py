"""skills / pick_place / lifecycle: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import move_to_precise
from depthrobotic.geometry.depth import frame_geometry, point_world
from depthrobotic.perception.images import observation_image, upright
from depthrobotic.perception.matching import flip_point
from depthrobotic.perception.parsing import (
    normalize_box,
    normalize_point,
    safe_ask_structured,
)
from depthrobotic.signals import StepBudgetExceeded, TaskCompleted
from depthrobotic.skills.pick_place.state import valid_region
import numpy as np


class ExecutorLifecycle:

    def _save(self):
        self.record["v30_state"] = self.state.snapshot()

    def _reference(self, label, role):
        raw_image = observation_image(self.obs, "agentview_image", context=self.context)
        image = upright(raw_image, context=self.context)
        response, value = safe_ask_structured(
            self.grounder,
            image,
            f'Instruction: {self.instruction}\nLocate only the {role}: {label}. Resolve any spatial qualifiers using the whole scene. Return its tight bbox and a point on the visible top/interior surface. JSON {{"bbox_2d":[x1,y1,x2,y2],"point":[x,y]}}, 0..1000; use null when not visible.',
            150,
        )
        box, point = (value.get("bbox_2d"), value.get("point"))
        rec = {"role": role, "label": label, "response": response, "valid": False}
        self.record["tool_calls"].append({"v30_reference": rec})
        if not valid_region(box, point):
            return None
        depth, cam, focal, _, scale = frame_geometry(
            self.env,
            self.grounder,
            raw_image,
            self.size,
            self.support_z,
            context=self.context,
        )
        pred = point_world(
            depth,
            flip_point(point, not self.context.settings.orient_fix),
            scale,
            cam,
            focal,
            self.size,
            context=self.context,
        )
        z = float(np.clip(pred[2], self.support_z - 0.08, self.support_z + 0.3))
        if z >= self.support_z + 0.28:
            z = self.support_z + 0.05
        features = self.context.models.features.features(image)
        rec.update(valid=True, box=box, point=point, z=z)
        return {"features": features, "box": box, "point": point, "z": z}

    def _adopt_target(self, ref):
        self.state.target.description = self.target_label
        self.state.target.observe(
            self.env.steps,
            point=ref["point"],
            box=ref["box"],
            surface_z=ref["z"],
            association="REFERENCE",
            identity="GROUNDED",
        )

    def _done(self, args):
        if (
            self.placed_once
            and self.state.relation in {"OUTSIDE", "UNKNOWN"}
            and (self.recovery_observations == 0)
            and (self.env.budget - self.env.steps > 60)
        ):
            self.recovery_observations += 1
            eef = np.asarray(self.obs["robot0_eef_pos"])
            self.obs = move_to_precise(
                self.env,
                self.obs,
                np.r_[eef[:2], min(eef[2] + 0.05, self.support_z + 0.4)],
                -1.0,
                self.events,
                "v30_relation_observe",
                max_steps=12,
            )
            self.state.record(
                "active_observation", self.env.steps, reason=self.state.relation
            )
            self._save()
            return (
                "A fresh view is available. The specified source-target relation is not yet verified; use approach_source/retry_place if correction is feasible, otherwise done.",
                observation_image(
                    self.obs, "robot0_eye_in_hand_image", context=self.context
                ),
            )
        return self._done_tool(args)

    def _capture_receiving_reference(self):
        self.record["action"] = "pick_place_v29_eupe"
        self.memory = None
        self.memory_dest = None
        image = observation_image(self.obs, "agentview_image", context=self.context)
        view = upright(image, context=self.context)
        raw, value = safe_ask_structured(
            self.grounder,
            view,
            f'Instruction: {self.instruction}\nLocate the receiving region: {self.destination}. Return the tight visible bounding box of this receiving region and a point on its usable interior/support surface. Exclude the source object and robot. JSON {{"bbox_2d":[x1,y1,x2,y2],"point":[x,y]}}; coordinates 0..1000. If not visible use null.',
            150,
        )
        box = (
            normalize_box(value.get("bbox_2d"))
            if isinstance(value.get("bbox_2d"), list)
            else None
        )
        point = normalize_point(value.get("point"))
        rec = {
            "mode": "eupe",
            "response": raw,
            "box": box,
            "point": point,
            "valid": False,
        }
        self.record["tool_calls"].append({"eupe_reference": rec})
        if box is None or point is None:
            return
        b, p = (np.asarray(box), np.asarray(point))
        if not (
            np.isfinite(b).all()
            and (b[:2] < b[2:]).all()
            and (b >= 0).all()
            and (b <= 1000).all()
            and (p >= b[:2]).all()
            and (p <= b[2:]).all()
        ):
            return
        raw_point = flip_point(point, not self.context.settings.orient_fix)
        depth, c2w, focal, _, scale = frame_geometry(
            self.env,
            self.grounder,
            image,
            self.size,
            self.support_z,
            context=self.context,
        )
        pred = point_world(
            depth, raw_point, scale, c2w, focal, self.size, context=self.context
        )
        z = float(np.clip(pred[2], self.support_z - 0.08, self.support_z + 0.3))
        if z >= self.support_z + 0.28:
            z = self.support_z + 0.05
        features = self.context.models.features.features(view)
        self.memory = {"features": features, "box": box, "point": point, "z": z}
        rec.update(valid=True, surface_z=z, raw_point=raw_point)

    def _dispatch_tool(self, name, args):
        """Execute a tool call, return (text_result, optional_image)."""
        self.record["tool_calls"].append({"tool": name, "args": args})
        handler = {
            "approach_source": self._approach_source,
            "grasp": self._grasp,
            "retry_grasp": self._retry_grasp,
            "place_object": self._place_object,
            "retry_place": self._retry_place,
            "done": self._done,
        }.get(name)
        if handler is None:
            return (f"Unknown tool: {name}", None)
        try:
            return handler(args)
        except (TaskCompleted, StepBudgetExceeded):
            raise
        except (NameError, AttributeError, ImportError, OSError):
            raise
        except Exception as e:
            return (f"Error: {type(e).__name__}: {e}", None)

    def _done_tool(self, args):
        if self.held and (not self.placed_once) and (self._done_refusals < 2):
            self._done_refusals += 1
            return (
                "You are still holding the object and have not placed it. Call place_object with the destination first.",
                None,
            )
        self.done = True
        return ("Step complete.", None)

    def dispatch(self, name, args):
        start = len(self.record["tool_calls"])
        try:
            text, image = self._dispatch_tool(name, args)
            self.record["tool_calls"].append(
                {"tool": name, "args": args, "result": text}
            )
            calls = self.record["tool_calls"][start:]
            outcome = {
                "tool": name,
                "held": bool(self.held),
                "placed_once": bool(self.placed_once),
                "steps": self.env.steps,
                "status": "UNKNOWN",
            }
            if text.startswith(("Error:", "Motion failed", "Grasp FAILED")):
                outcome["status"] = "FAILED"
            elif name in ("grasp", "retry_grasp") and self.held:
                outcome["status"] = "GRASP_CONFIRMED"
            for call in calls:
                if "release" in call:
                    outcome["release_state"] = call["release"].get(
                        "final_state", "UNKNOWN"
                    )
                if "nudge" in call:
                    outcome["placement_measurement"] = call["nudge"]
            if name == "done":
                outcome["status"] = "CONTROLLER_STOPPED"
            self.record["execution_states"].append(outcome)
            return (text, image)
        finally:
            self._save()
