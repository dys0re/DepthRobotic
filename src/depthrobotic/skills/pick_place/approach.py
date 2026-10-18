"""skills / pick_place / approach: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.control.motion import in_workspace, move_to
from depthrobotic.geometry.camera import (
    camera_geometry,
    intersect_horizontal_plane,
    project_world_to_raw,
)
from depthrobotic.geometry.depth import frame_geometry, point_world
from depthrobotic.perception.candidates import (
    ELEVATED_MIN_DZ,
    GDINO_MIN_BOXES,
    MARK_STYLE,
    MAX_ALTERNATES,
    MAX_TEXT_SCAN,
    MIN_BUDGET_FOR_LAST_RESORT,
    RELATION_RADIUS,
    RecordingProposer,
    _relation_from_record,
    _target_and_relation,
    _tokens,
    clean_model_text,
    content_tokens,
    generic_variants,
)
from depthrobotic.perception.images import (
    annotate_candidates,
    crop_from_normalized,
    observation_image,
    upright,
)
from depthrobotic.perception.marks import MAX_MARKS, choose_mark, filter_boxes
from depthrobotic.perception.matching import consensus_match, flip_point
from depthrobotic.perception.parsing import safe_ask_structured
from depthrobotic.skills.pick_place.locate import (
    STRONG_IDENTITY,
    ambiguous_source_description,
    list_instances,
    locate_relational_source,
    locate_source,
    name_tokens,
    parse_relation,
)
from depthrobotic.skills.pick_place.state import semantic_evidence
from depthrobotic.skills.pick_place.wrist import (
    box_metrics,
    measure_object_box,
    wrist_servo,
)
from typing import Any
import numpy as np
import re
import torch


class SourceApproach:

    def _source_match(self):
        if self.source_reference is None:
            return None
        ref = self.source_reference
        features = self.context.models.features.features(
            upright(
                observation_image(self.obs, "agentview_image", context=self.context),
                context=self.context,
            )
        )
        point, rec = consensus_match(
            ref["features"],
            features,
            ref["box"],
            ref["point"],
            self.context.models.features.resolution // 16,
        )
        self.record["tool_calls"].append({"v30_source_match": rec})
        if point is None:
            self.state.source.association = "UNKNOWN"
            return None
        shift = np.asarray(point) - np.asarray(ref["point"])
        box = np.asarray(ref["box"]) + np.tile(shift, 2)
        elevated = ref["z"] > self.support_z + 0.06
        plane = ref["z"] - 0.02 if elevated else self.support_z
        px = point if elevated else [(box[0] + box[2]) / 2, box[3]]
        cam, focal, _ = camera_geometry(self.env, "agentview", self.size)
        world = intersect_horizontal_plane(
            flip_point(px, not self.context.settings.orient_fix),
            plane,
            cam,
            focal,
            self.size,
            context=self.context,
        )
        if not in_workspace(world[:2]):
            return None
        fresh_height = ref["z"] if self.state.source.dirty_reason is None else None
        self.state.source.observe(
            self.env.steps, point=point, box=box, world=world, surface_z=fresh_height
        )
        return world

    def _verify_identity_at_wrist(self, desc):
        ok, rec = self._verify_identity_at_wrist_contact(desc)
        evidence = semantic_evidence(desc, rec.get("object", ""), rec.get("text", ""))
        self.state.source.identity = evidence
        self.last_identity = rec
        if evidence == "LABEL_CONFLICT":
            ok = False
            rec["verdict"] = "rejected_by_label"
        candidate = self._source_match()
        if candidate is not None and evidence == "UNKNOWN" and (ok is True):
            gap = float(
                np.linalg.norm(
                    candidate[:2] - np.asarray(self.obs["robot0_eef_pos"])[:2]
                )
            )
            rec["v30_reference_gap_m"] = gap
            visual = self._wrist_descriptor() if gap > 0.06 else None
            score = (
                float(torch.dot(self.source_descriptor, visual))
                if self.source_descriptor is not None and visual is not None
                else None
            )
            rec["v30_reference_appearance"] = score
            if (
                gap > 0.06
                and score is not None
                and (score < 0.65)
                and (not self.source_candidate_used)
            ):
                rec["verdict"] = "v30_instance_disagreement"
                ok = False
        self.state.record(
            "identity",
            self.env.steps,
            evidence=evidence,
            legacy_verdict=rec.get("verdict"),
        )
        self._save()
        return (ok, rec)

    def _next_source_candidate(self, desc, alternatives):
        if not self.source_candidate_used:
            world = self._source_match()
            self.source_candidate_used = True
            if (
                world is not None
                and np.linalg.norm(world[:2] - np.asarray(self.source_world)[:2]) > 0.04
            ):
                self.source_surface_z = self.source_reference["z"]
                self.state.record(
                    "alternate_source", self.env.steps, world=world.tolist()
                )
                return world
        return self._next_source_candidate_elevated(desc, alternatives)

    def _approach_source(self, args):
        proposer = self.context.models.detector
        if isinstance(proposer, RecordingProposer):
            proposer.boxes = []
        return self._approach_source_relation(args)

    def _bbox_for(self, entry):
        proposer = self.context.models.detector
        if not isinstance(proposer, RecordingProposer):
            return None
        want = np.asarray(entry["norm_px"], dtype=float)
        for bx in proposer.boxes:
            px = bx.base
            norm = np.asarray(
                [
                    px[0] / max(1, self.size - 1) * 1000.0,
                    px[1] / max(1, self.size - 1) * 1000.0,
                ]
            )
            if float(np.abs(norm - want).max()) < 0.5:
                return bx
        return None

    def _world_from_box(self, bx, frame, cam2world, focal):
        """v17 projected every box bottom onto the table plane; an object resting on another object then lands
        far beyond itself, fails the eef-distance/workspace filters and never enters the pool (spatial:3 bowl on
        cookie box).  Use DepthVLM depth at the box centre, as v12's own alternate path does, when it says the
        object is elevated."""
        base = bx.base
        norm = [
            base[0] / max(1, self.size - 1) * 1000.0,
            base[1] / max(1, self.size - 1) * 1000.0,
        ]
        world = intersect_horizontal_plane(
            norm, self.support_z, cam2world, focal, self.size, context=self.context
        )
        surface_z, elevated, dz = (self.support_z, False, None)
        if frame is not None:
            x1, y1, x2, y2 = bx.bbox
            pt = [
                (x1 + x2) / 2 / max(1, self.size - 1) * 1000.0,
                (y1 + y2) / 2 / max(1, self.size - 1) * 1000.0,
            ]
            try:
                depth, c2w, f_px, _, scale = frame
                pred = point_world(
                    depth, pt, scale, c2w, f_px, self.size, context=self.context
                )
                dz = float(pred[2]) - self.support_z
                if ELEVATED_MIN_DZ < dz < 0.28:
                    world = intersect_horizontal_plane(
                        pt,
                        float(pred[2]) - 0.03,
                        cam2world,
                        focal,
                        self.size,
                        context=self.context,
                    )
                    surface_z, elevated = (float(pred[2]), True)
            except Exception:
                pass
        return (np.asarray(world, dtype=float), norm, surface_z, elevated, dz)

    def _gdino_pool(self, desc):
        if self._relation_pool is not None:
            return self._relation_pool
        pool: list[dict[str, Any]] = []
        image = observation_image(self.obs, "agentview_image", context=self.context)
        cam2world, focal, _ = camera_geometry(self.env, "agentview", self.size)
        target, parsed = _target_and_relation(desc, self.instruction)
        rec: dict[str, Any] = {
            "phrase": desc,
            "target_phrase": target,
            "parsed": parsed,
            "depth_aware": True,
        }
        if self.context.models.detector is None:
            rec["error"] = "no GroundingDINO context"
            self.record["tool_calls"].append({"v23_gdino": rec})
            self._relation_pool = pool
            return pool
        boxes, grec = self.context.models.detector.detect(image, [target], prefix="V17")
        rec["specific"] = {"n": len(boxes), "query": grec.get("query")}
        tagged = [(True, bx) for bx in boxes]
        if len(boxes) < GDINO_MIN_BOXES:
            extra = generic_variants(target)
            if extra:
                more, grec2 = self.context.models.detector.detect(
                    image, [target] + extra, prefix="V17W"
                )
                rec["widened"] = {"n": len(more), "query": grec2.get("query")}
                target_tok = _tokens(target)
                tagged.extend(
                    ((bool(target_tok & _tokens(bx.label)), bx) for bx in more)
                )
        try:
            frame = frame_geometry(
                self.env,
                self.grounder,
                image,
                self.size,
                self.support_z,
                context=self.context,
            )
        except Exception as error:
            frame = None
            rec["depth_error"] = f"{type(error).__name__}: {error}"[:80]
        anchor = _relation_from_record(self.record)
        anchor_pts = (
            [[float(p[0]), float(p[1])] for p in anchor["anchor_points"]]
            if anchor
            else []
        )
        relation = parsed.get("relation") if parsed else None
        ref = np.mean(np.asarray(anchor_pts), axis=0) if anchor_pts else None
        seen: list[np.ndarray] = []
        target_tok = _tokens(target)
        eef_xy = np.asarray(self.obs["robot0_eef_pos"], dtype=float)[:2]
        dropped = {"workspace": 0, "duplicate": 0, "far_from_eef": 0, "projection": 0}
        dropped_detail: list[dict[str, Any]] = []
        for specific, bx in tagged:
            try:
                world, norm, surface_z, elevated, dz = self._world_from_box(
                    bx, frame, cam2world, focal
                )
            except RuntimeError:
                dropped["projection"] += 1
                dropped_detail.append({"label": bx.label[:20], "reason": "projection"})
                continue
            reason = None
            if not in_workspace(world[:2]):
                reason = "workspace"
            elif any((float(np.linalg.norm(world[:2] - q)) < 0.04 for q in seen)):
                reason = "duplicate"
            elif float(np.linalg.norm(world[:2] - eef_xy)) > 0.45:
                reason = "far_from_eef"
            if reason:
                dropped[reason] += 1
                dropped_detail.append(
                    {
                        "label": bx.label[:20],
                        "reason": reason,
                        "world": world[:2].round(3).tolist(),
                        "elevated": elevated,
                        "dz": None if dz is None else round(dz, 3),
                    }
                )
                continue
            relation_dist, relation_ok = (None, True)
            if ref is not None and relation:
                q = np.asarray(norm, dtype=float)
                if relation == "between" and len(anchor_pts) >= 2:
                    a, b = (np.asarray(anchor_pts[0]), np.asarray(anchor_pts[1]))
                    ab = b - a
                    t = float(
                        np.clip(np.dot(q - a, ab) / (np.dot(ab, ab) + 1e-09), 0.0, 1.0)
                    )
                    relation_dist = float(np.linalg.norm(q - (a + t * ab)))
                else:
                    relation_dist = float(np.linalg.norm(q - ref))
                relation_ok = relation_dist <= RELATION_RADIUS.get(relation, 400.0)
            semantic = bool(target_tok & _tokens(bx.label)) or specific
            seen.append(world[:2].copy())
            pool.append(
                {
                    "world": world,
                    "norm_px": norm,
                    "score": float(bx.score),
                    "label": bx.label[:40],
                    "specific": bool(specific),
                    "semantic": semantic,
                    "relation_ok": relation_ok,
                    "relation_dist": relation_dist,
                    "surface_z": surface_z,
                    "elevated": elevated,
                    "dz": None if dz is None else round(dz, 3),
                }
            )
        pool.sort(
            key=lambda p: (
                p["relation_ok"],
                p["semantic"],
                p["specific"],
                -float(p["relation_dist"] or 1000000000.0),
                (
                    -float(np.linalg.norm(p["world"][:2] - np.asarray(self._prior[:2])))
                    if self._prior is not None
                    else 0.0
                ),
            ),
            reverse=True,
        )
        rec.update(
            {
                "kept": len(pool),
                "dropped": dropped,
                "dropped_detail": dropped_detail[:12],
                "elevated_min_dz": ELEVATED_MIN_DZ,
                "elevated": sum((bool(p["elevated"]) for p in pool)),
                "anchor_points": anchor_pts,
                "relation": relation,
                "relation_consistent": sum((bool(p["relation_ok"]) for p in pool)),
                "top": [
                    {
                        "label": p["label"],
                        "world": p["world"][:2].round(3).tolist(),
                        "elevated": p["elevated"],
                        "dz": p["dz"],
                        "relation_ok": p["relation_ok"],
                    }
                    for p in pool[:MAX_ALTERNATES]
                ],
            }
        )
        self.record["tool_calls"].append({"v23_gdino": rec})
        self._relation_pool = pool
        return pool

    def _next_source_candidate_elevated(self, desc, alternatives):
        if self._prior is None:
            self._prior = np.asarray(self.source_world, dtype=float).copy()
        current = np.asarray(self.source_world[:2])
        pool = [
            p
            for p in self._gdino_pool(desc)
            if float(np.linalg.norm(p["world"][:2] - current)) >= 0.06
        ]
        for used in self._used:
            pool = [
                p for p in pool if float(np.linalg.norm(p["world"][:2] - used)) >= 0.06
            ]
        if not pool:
            return self._next_source_candidate_detection(desc, alternatives)
        gated = [p for p in pool if p["relation_ok"]]
        shortlist = (gated or pool)[:MAX_ALTERNATES]
        fallback = shortlist[0]
        matched = [(p, self._bbox_for(p)) for p in pool]
        unmatched = sum((1 for _, b in matched if b is None))
        matched = [(p, b) for p, b in matched if b is not None]
        keep = {id(b) for b in filter_boxes([b for _, b in matched], size=self.size)}
        candidates = [(p, b) for p, b in matched if id(b) in keep][:MAX_MARKS]
        rec: dict[str, Any] = {
            "pool": len(pool),
            "gated_pool": len(gated),
            "unmatched_bbox": unmatched,
            "filtered": len(matched) - len(candidates),
            "marked": len(candidates),
            "fallback_label": fallback["label"],
            "fallback_world": fallback["world"][:2].round(3).tolist(),
        }
        picked = fallback
        image = observation_image(self.obs, "agentview_image", context=self.context)
        rec["used"] = "fallback_no_marks"
        picked_world = np.asarray(picked["world"], dtype=float)
        surface_z = float(picked.get("surface_z", self.support_z))
        rec.update(
            {
                "label": picked["label"],
                "relation_ok": picked["relation_ok"],
                "relation_dist": picked["relation_dist"],
                "elevated": picked.get("elevated"),
                "dz": picked.get("dz"),
                "world": picked_world[:2].round(3).tolist(),
            }
        )
        self.record["tool_calls"].append({"v23_mark_select": rec})
        self._used.append(picked_world[:2].copy())
        self.source_surface_z = surface_z
        return picked_world

    def _approach_source_relation(self, args):
        self._relation_pool = None
        return self._approach_source_detection(args)

    def _approach_source_detection(self, args):
        self._prior = None
        self._pool = None
        self._used = []
        try:
            return self._approach_source_contact(args)
        except RuntimeError as error:
            recoverable = "Could not find" in str(
                error
            ) or "did not reach target" in str(error)
            if not recoverable:
                raise
            budget_left = int(self.env.budget - self.env.steps)
            more_steps_follow = (
                int(self.env.budget) >= 500 and len(self.perception_log) <= 1
            )
            rec = {
                "reason": str(error)[:90],
                "budget_left": budget_left,
                "trigger": (
                    "unreachable" if "did not reach" in str(error) else "identity"
                ),
                "more_steps_follow": more_steps_follow,
                "prior": (
                    None
                    if self._prior is None
                    else np.asarray(self._prior).round(3).tolist()
                ),
            }
            if (
                self._prior is None
                or more_steps_follow
                or budget_left < MIN_BUDGET_FOR_LAST_RESORT
            ):
                rec["applied"] = False
                self.record["tool_calls"].append({"v16_last_resort": rec})
                raise
            desc = args.get("object_description", self.source)
            world = np.asarray(self._prior, dtype=float)
            if world.shape[0] < 3:
                world = np.r_[world[:2], self.support_z]
            self.obs = move_to(
                self.env,
                self.obs,
                np.r_[world[:2], self.travel_z],
                -1.0,
                self.events,
                "approach_source",
                max_steps=45,
            )
            self.obs, self.wrist_center = wrist_servo(
                self.env,
                self.obs,
                self.grounder,
                desc,
                self.instruction,
                "source",
                world,
                self.support_z,
                self.travel_z,
                -1.0,
                self.size,
                self.events,
                [],
                context=self.context,
            )
            self.source_world = np.asarray(world, dtype=float)
            self.source_surface_z = self.support_z
            rec["applied"] = True
            self.record["tool_calls"].append({"v16_last_resort": rec})
            return (
                f"Moved above '{desc}' using the best available grounding. The wrist camera could not confirm its identity, so the object below may not be the requested one. Observe the wrist image and choose a grasp strategy.",
                observation_image(
                    self.obs, "robot0_eye_in_hand_image", context=self.context
                ),
            )

    def _two_scale_text(self, image, norm_px):
        reads = []
        for frac in (0.3, 0.15):
            crop, _b = crop_from_normalized(image, norm_px, frac)
            raw, _v = safe_ask_structured(
                self.grounder,
                crop,
                "Look at the object in the centre of this image. What product is it, and what words are printed on its label? Answer in a few words, no JSON.",
                40,
            )
            reads.append(clean_model_text(raw))
        agreed = content_tokens(reads[0]) & content_tokens(reads[1])
        return " ".join(sorted(agreed)) if agreed else ""

    def _next_source_candidate_detection(self, desc, alternatives):
        if self._prior is None:
            self._prior = np.asarray(self.source_world, dtype=float).copy()
        pool = [
            p
            for p in self._gdino_pool(desc)
            if float(np.linalg.norm(p["world"][:2] - np.asarray(self.source_world[:2])))
            >= 0.06
        ]
        for used in self._used:
            pool = [
                p for p in pool if float(np.linalg.norm(p["world"][:2] - used)) >= 0.06
            ]
        if pool:
            image = observation_image(self.obs, "agentview_image", context=self.context)
            desc_tok = content_tokens(desc, 4)
            picked, how, reads = (None, None, [])
            if desc_tok:
                for item in pool[:MAX_TEXT_SCAN]:
                    txt = self._two_scale_text(image, item["norm_px"])
                    hit = bool(desc_tok & content_tokens(txt))
                    reads.append(
                        {
                            "px": [
                                round(item["norm_px"][0]),
                                round(item["norm_px"][1]),
                            ],
                            "text": txt[:40],
                            "match": hit,
                        }
                    )
                    if hit:
                        picked, how = (item, "two_scale_printed_text")
                        break
            if picked is None:
                picked, how = (
                    pool[0],
                    (
                        "gdino_specific_order"
                        if pool[0]["specific"]
                        else "gdino_generic_order"
                    ),
                )
            self.record["tool_calls"].append(
                {
                    "v16_alternate": {
                        "how": how,
                        "pool": len(pool),
                        "label": picked["label"],
                        "score": round(picked["score"], 3),
                        "reads": reads,
                        "world": picked["world"][:2].round(3).tolist(),
                    }
                }
            )
            self._used.append(picked["world"][:2].copy())
            self.source_surface_z = self.support_z
            return picked["world"]
        return self._next_source_candidate_contact(desc, alternatives)

    def _ocr_select_candidate(self, desc):
        world, rec = self._ocr_select_candidate_contact(desc)
        if world is None or "chosen_px" not in rec:
            return (world, rec)
        image = observation_image(self.obs, "agentview_image", context=self.context)
        confirm = self._two_scale_text(image, rec["chosen_px"])
        rec["two_scale_text"] = confirm[:60]
        if not content_tokens(desc, 4) & content_tokens(confirm):
            rec["two_scale_rejected"] = True
            return (None, rec)
        return (world, rec)

    def _approach_source_contact(self, args):
        self.top_z = None
        self.rim_z = None
        self.box = None
        self.width_finger = None
        self.servo_plane_done = None
        self.infeasible_count = 0
        self.probe_done = False
        self.fingers_closed = False
        self.failure_history = []
        desc = args.get("object_description", self.source)
        self.grasp_attempt = 0
        _, self.source_world, self.source_surface_z = locate_relational_source(
            self.env,
            self.obs,
            self.grounder,
            desc,
            self.instruction,
            self.size,
            self.support_z,
            log=self.record["tool_calls"],
            context=self.context,
        )
        self.record["tool_calls"].append(
            {"source_world": np.asarray(self.source_world).round(4).tolist()}
        )
        if not in_workspace(self.source_world[:2]):
            _, world2, surf2 = locate_source(
                self.env,
                self.obs,
                self.grounder,
                desc,
                self.instruction,
                self.size,
                self.support_z,
                log=self.record["tool_calls"],
                context=self.context,
            )
            self.record["tool_calls"].append(
                {"source_world_retry": np.asarray(world2).round(4).tolist()}
            )
            if in_workspace(world2[:2]):
                self.source_world, self.source_surface_z = (world2, surf2)
            else:
                raise RuntimeError(
                    f"'{desc}' was located outside the reachable workspace at {np.asarray(self.source_world[:2]).round(2).tolist()}; the grounding is unreliable"
                )
        alternatives = list(self.context.source_candidates)
        self._v7_alt_added = False
        self._generic_alt_added = False
        if not ambiguous_source_description(desc, self.instruction):
            try:
                picked, ocr_rec = self._ocr_select_candidate(desc)
            except Exception as e:
                picked, ocr_rec = (None, {"error": f"{type(e).__name__}: {e}"[:100]})
            self.record["tool_calls"].append({"ocr_candidate_select": ocr_rec})
            if picked is not None and in_workspace(picked[:2]):
                if (
                    float(
                        np.linalg.norm(picked[:2] - np.asarray(self.source_world[:2]))
                    )
                    > 0.06
                ):
                    self.ocr_candidate_px = list(ocr_rec["chosen_px"])
                    alternatives.insert(0, self.ocr_candidate_px)
                    self.record["tool_calls"].append(
                        {
                            "ocr_candidate": {
                                "pipeline": np.asarray(self.source_world[:2])
                                .round(3)
                                .tolist(),
                                "ocr": picked[:2].round(3).tolist(),
                            }
                        }
                    )
        identity_note = ""
        first_world, first_verdict = (
            np.asarray(self.source_world, dtype=float).copy(),
            None,
        )
        for hop in range(4):
            surf = (
                float(self.source_surface_z)
                if self.source_surface_z is not None
                else self.support_z
            )
            if surf > self.support_z + 0.06:
                self.travel_z = max(self.travel_z, surf + 0.14)
                servo_plane = surf - 0.03
            else:
                servo_plane = self.support_z
            self.obs = move_to(
                self.env,
                self.obs,
                np.r_[self.source_world[:2], self.travel_z],
                -1.0,
                self.events,
                "approach_source",
                max_steps=45,
            )
            self.obs, self.wrist_center = wrist_servo(
                self.env,
                self.obs,
                self.grounder,
                desc,
                self.instruction,
                "source",
                self.source_world,
                servo_plane,
                self.travel_z,
                -1.0,
                self.size,
                self.events,
                [],
                context=self.context,
            )
            ok, vrec = self._verify_identity_at_wrist(desc)
            vrec["hop"] = hop
            self.record["tool_calls"].append({"identity_check": vrec})
            if hop == 0:
                first_verdict = vrec.get("verdict")
            if ok is None:
                if hop == 0:
                    identity_note = " NOTE: the wrist view could not name the object below, but an object outline is present at the grasp point; its identity is unconfirmed."
                    vrec["decision"] = "proceed_on_prior"
                    break
                ok = False
                vrec["decision"] = "not_accepted_at_switched_candidate"
            elif ok and hop > 0 and (vrec.get("verdict") not in STRONG_IDENTITY):
                ok = False
                vrec["decision"] = "weak_acceptance_not_enough_at_switched_candidate"
            if ok:
                break
            budget_left = int(self.env.budget - self.env.steps)
            nxt = (
                None
                if hop >= 2 or budget_left < 170
                else self._next_source_candidate(desc, alternatives)
            )
            if nxt is None:
                if vrec.get("verdict") == "rejected_by_specific_name" and hop == 0:
                    identity_note = f" NOTE: the wrist view names the object below {vrec.get('object')!r}, not '{desc}', but no alternative candidate exists; proceeding with the best grounding."
                    break
                if (
                    first_verdict == "rejected_by_specific_name"
                    and hop > 0
                    and (budget_left > 120)
                ):
                    self.record["tool_calls"].append(
                        {
                            "identity_fallback_to_first": first_world[:2]
                            .round(3)
                            .tolist()
                        }
                    )
                    self.source_world = first_world
                    self.obs = move_to(
                        self.env,
                        self.obs,
                        np.r_[first_world[:2], self.travel_z],
                        -1.0,
                        self.events,
                        "approach_source",
                        max_steps=45,
                    )
                    self.obs, self.wrist_center = wrist_servo(
                        self.env,
                        self.obs,
                        self.grounder,
                        desc,
                        self.instruction,
                        "source",
                        first_world,
                        self.support_z,
                        self.travel_z,
                        -1.0,
                        self.size,
                        self.events,
                        [],
                        context=self.context,
                    )
                    identity_note = " NOTE: the wrist view could not confirm the object's identity; proceeding with the best candidate."
                    break
                self.wrist_center = None
                raise RuntimeError(
                    f"Could not find '{desc}' under the gripper after {hop + 1} candidate(s); the wrist camera shows {vrec.get('object')!r}. The object may be mis-located or described differently. Call approach_source with a clearer description, or done."
                )
            self.record["tool_calls"].append(
                {
                    "identity_switch": {
                        "from": np.asarray(self.source_world[:2]).round(3).tolist(),
                        "to": np.asarray(nxt[:2]).round(3).tolist(),
                    }
                }
            )
            self.source_world = nxt
        wrist_image = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        return (
            f"Moved above '{desc}'. The wrist camera image shows the object below. Observe its shape and choose a grasp strategy."
            + identity_note,
            wrist_image,
        )

    def _verify_identity_at_wrist_contact(self, desc):
        """Close-up check of the object under the gripper: name it, read its label, judge the match.
        Rejects only on positive evidence of a different object (label text or a non-overlapping name).
        """
        full = observation_image(
            self.obs, "robot0_eye_in_hand_image", context=self.context
        )
        eef = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        plane = max(
            (
                float(self.source_surface_z)
                if self.source_surface_z is not None
                else self.support_z
            ),
            self.support_z + 0.02,
        )
        try:
            exp_px = project_world_to_raw(
                self.env,
                "robot0_eye_in_hand",
                np.r_[eef[:2], plane],
                self.size,
                context=self.context,
            )
        except RuntimeError:
            exp_px = [500.0, 380.0]
        if not (0 <= exp_px[0] <= 1000 and 0 <= exp_px[1] <= 1000):
            exp_px = [500.0, 380.0]
        image = annotate_candidates(
            full,
            {
                "X": [
                    float(np.clip(exp_px[0], 15, 985)),
                    float(np.clip(exp_px[1], 15, 985)),
                ]
            },
        )
        prompt = 'The wrist camera of a robot looks down at a tabletop. The red marker labeled X is drawn at the spot directly below the gripper. Name the single object that the marker X lies on, in one to three words (say \'nothing\' if the marker lies on the bare table surface), and read any label text printed on that object. JSON only: {"object":"<short name>","text":"<label text or empty>"}'
        response, value = safe_ask_structured(self.grounder, image, prompt, 80)
        obj_name = str(value.get("object", "") or "").lower()
        text = str(value.get("text", "") or "").lower().strip()
        if text.strip("-_ .\xad") in {
            "",
            "none",
            "n/a",
            "na",
            "null",
            "no text",
            "nothing",
            "empty",
            "unknown",
            "no label",
            "not visible",
        }:
            text = ""
        match = None
        desc_tok = {t for t in name_tokens(desc) if len(t) >= 3}
        text_tok = {t for t in name_tokens(text) if len(t) >= 3}
        obj_tok = {t for t in name_tokens(obj_name) if len(t) >= 3}
        rec = {
            "object": obj_name[:40],
            "text": text[:60],
            "match": match,
            "response": response[:120],
        }
        ocr_overlap = bool(desc_tok & text_tok)
        ocr_contradicts = (
            bool(text_tok)
            and (not ocr_overlap)
            and any((len(t) >= 4 for t in text_tok))
            and bool(desc_tok)
        )
        name_overlap = bool(desc_tok & obj_tok)
        generic_words = {
            "object",
            "item",
            "container",
            "can",
            "box",
            "bottle",
            "jar",
            "package",
            "carton",
            "cup",
            "thing",
            "food",
            "product",
            "bag",
            "tin",
            "packet",
            "dish",
        }
        specific_name = (
            obj_tok
            - generic_words
            - {
                "black",
                "white",
                "red",
                "blue",
                "green",
                "yellow",
                "small",
                "large",
                "big",
            }
        )
        if ocr_overlap:
            rec["verdict"] = "confirmed_by_label"
            return (True, rec)
        if name_overlap:
            rec["verdict"] = "accepted_by_name"
            parsed = parse_relation(desc, self.instruction)
            if parsed and parsed["relation"] in ("on", "on top of", "in", "inside"):
                q = 'The wrist camera looks down at a tabletop; the red marker X is on an object. What is that object resting on or sitting inside? Answer in one to three words. JSON only: {"support":"<words>"}'
                _, v3 = safe_ask_structured(self.grounder, image, q, 40)
                sup_tok = {
                    t
                    for t in name_tokens(str(v3.get("support", "") or ""))
                    if len(t) >= 3
                }
                anc_tok = {
                    t for t in name_tokens(" ".join(parsed["anchors"])) if len(t) >= 3
                }
                rec["support"] = str(v3.get("support", ""))[:40]
                if (
                    sup_tok
                    and anc_tok
                    and (not sup_tok & anc_tok)
                    and (not sup_tok <= {"table", "tabletop", "surface", "counter"})
                ):
                    rec["verdict"] = "rejected_by_relation"
                    return (False, rec)
            return (True, rec)
        absence = {"nothing", "none", "empty", "surface", "background"}
        says_nothing = bool(obj_tok) and obj_tok <= absence
        box, _ = measure_object_box(
            self.grounder, self.obs, desc, self.instruction, context=self.context
        )
        box_present = False
        if box is not None:
            try:
                _, _, bc = box_metrics(
                    self.env, self.size, box, plane, context=self.context
                )
                rec["box_offset_cm"] = round(
                    float(np.linalg.norm(bc - eef[:2])) * 100, 1
                )
                box_present = rec["box_offset_cm"] <= 9.0
            except RuntimeError:
                pass
        else:
            rec["box_offset_cm"] = None
        if says_nothing and (not box_present):
            rec["verdict"] = "rejected_nothing_below"
            return (False, rec)
        if says_nothing:
            rec["verdict"] = "uncertain_nothing_but_outline_present"
            return (None, rec)
        if (
            not box_present
            and rec.get("box_offset_cm") is not None
            and (rec["box_offset_cm"] > 12.0)
        ):
            rec["verdict"] = "rejected_nothing_below"
            return (False, rec)
        if ocr_contradicts and specific_name:
            rec["verdict"] = "rejected_by_label_and_name"
            return (False, rec)
        if ocr_contradicts or specific_name:
            rec["verdict"] = "uncertain_single_signal"
            parsed = parse_relation(desc, self.instruction)
            if parsed and parsed["relation"] in ("on", "on top of", "in", "inside"):
                q = 'The wrist camera looks down at a tabletop; the red marker X is on an object. What is that object resting on or sitting inside? Answer in one to three words. JSON only: {"support":"<words>"}'
                _, v3 = safe_ask_structured(self.grounder, image, q, 40)
                sup_tok = {
                    t
                    for t in name_tokens(str(v3.get("support", "") or ""))
                    if len(t) >= 3
                }
                anc_tok = {
                    t for t in name_tokens(" ".join(parsed["anchors"])) if len(t) >= 3
                }
                rec["support"] = str(v3.get("support", ""))[:40]
                if (
                    sup_tok
                    and anc_tok
                    and (not sup_tok & anc_tok)
                    and (not sup_tok <= {"table", "tabletop", "surface", "counter"})
                ):
                    rec["verdict"] = "rejected_by_relation"
                    return (False, rec)
            return (True, rec)
        prompt2 = f'Robot instruction: {self.instruction}\nThe wrist camera looks down at the object directly below the gripper. Is that object the {desc}? JSON only: {{"match":true}} or {{"match":false}}.'
        _, v2 = safe_ask_structured(self.grounder, image, prompt2, 32)
        rec["match"] = v2.get("match")
        if v2.get("match") is False and obj_tok:
            rec["verdict"] = "rejected_by_name"
            return (False, rec)
        rec["verdict"] = "accepted"
        return (True, rec)

    def _ocr_select_candidate_contact(self, desc):
        """Packaged goods show their label to the fixed front camera, not to the wrist camera looking at their lid.
        Enumerate tabletop items in the agentview, read each label in a tight crop, pick the one whose text shares a
        distinctive word with the target phrase. Returns (world or None, record)."""
        rec = {"tried": 0}
        desc_tok = {t for t in name_tokens(desc) if len(t) >= 4}
        if not desc_tok:
            rec["skipped"] = "no distinctive word"
            return (None, rec)
        image = observation_image(self.obs, "agentview_image", context=self.context)
        pts = []
        for cat in ("can", "box or carton", "bottle"):
            got, _ = list_instances(
                self.grounder, image, cat, self.instruction, max_points=6
            )
            for g in got:
                if all(
                    (
                        float(np.linalg.norm(np.asarray(g) - np.asarray(q))) > 50
                        for q in pts
                    )
                ):
                    pts.append(g)
        rec["enumerated"] = len(pts)
        cam2world, focal, _ = camera_geometry(self.env, "agentview", self.size)
        best = None
        for pt in pts:
            crop, _b = crop_from_normalized(image, pt, 0.3)
            raw, _v = safe_ask_structured(
                self.grounder,
                crop,
                "Look at the object in the centre of this image. What product is it, and what words are printed on its label? Answer in a few words, no JSON.",
                40,
            )
            text = str(raw or "").lower().replace("parse_error", "")
            tok = {t for t in name_tokens(text) if len(t) >= 3}
            rec["tried"] += 1
            overlap = desc_tok & tok
            rec.setdefault("reads", []).append(
                {
                    "px": [round(pt[0]), round(pt[1])],
                    "text": text[:40],
                    "match": bool(overlap),
                }
            )
            if overlap:
                best = pt
                break
        if best is None and len(rec.get("reads", [])) >= 1:
            reads = rec["reads"]
            listing = "\n".join((f"{i + 1}. {r['text']}" for i, r in enumerate(reads)))
            prompt = f'A robot must pick up the {desc}. These are label readings of the packaged products on the table:\n{listing}\nWhich numbered reading describes the {desc}? Consider brand names and flavours (e.g. a ranch or vinaigrette is a salad dressing). Answer 0 if none of them does. JSON only: {{"choice": <number>}}'
            raw, val = safe_ask_structured(self.grounder, image, prompt, 24)
            choice = None
            try:
                choice = (
                    int(val.get("choice"))
                    if isinstance(val, dict) and val.get("choice") is not None
                    else None
                )
            except (TypeError, ValueError):
                choice = None
            if choice is None:
                m = re.search("\\b(\\d{1,2})\\b", str(raw))
                choice = int(m.group(1)) if m else None
            rec["semantic_choice"] = {"response": str(raw)[:60], "choice": choice}
            if choice is not None and 1 <= choice <= len(reads):
                best = pts[choice - 1] if choice - 1 < len(pts) else None
                if best is not None:
                    reads[choice - 1]["match"] = "semantic"
        if best is None:
            return (None, rec)
        try:
            w = intersect_horizontal_plane(
                best, self.support_z, cam2world, focal, self.size, context=self.context
            )
        except RuntimeError:
            return (None, rec)
        rec["chosen_px"] = best
        return (w, rec)

    def _next_source_candidate_contact(self, desc, alternatives):
        """Next agentview candidate for the source: relational alternatives first, then a fresh detection list.
        Skips candidates within 6cm of the rejected point or outside the workspace."""
        image = observation_image(self.obs, "agentview_image", context=self.context)
        cam2world, focal, _ = camera_geometry(self.env, "agentview", self.size)
        if not getattr(self, "_v7_alt_added", False):
            self._v7_alt_added = True
            try:
                pt7, _, _ = locate_source(
                    self.env,
                    self.obs,
                    self.grounder,
                    desc,
                    self.instruction,
                    self.size,
                    self.support_z,
                    context=self.context,
                )
                if pt7 is not None:
                    at = (
                        1
                        if self.ocr_candidate_px is not None
                        and alternatives
                        and (alternatives[0] == self.ocr_candidate_px)
                        else 0
                    )
                    alternatives.insert(at, pt7)
            except Exception:
                pass
        if not alternatives:
            pts, _ = list_instances(self.grounder, image, desc, self.instruction)
            alternatives.extend(pts)
        if not alternatives and (not getattr(self, "_generic_alt_added", False)):
            self._generic_alt_added = True
            pts, _ = list_instances(
                self.grounder,
                image,
                "loose tabletop object (can, box, bottle, carton, bowl, cup or similar)",
                self.instruction,
                max_points=8,
            )
            cur = np.asarray(self.source_world[:2])

            def _dist(pt):
                try:
                    return float(
                        np.linalg.norm(
                            intersect_horizontal_plane(
                                pt,
                                self.support_z,
                                cam2world,
                                focal,
                                self.size,
                                context=self.context,
                            )[:2]
                            - cur
                        )
                    )
                except RuntimeError:
                    return 9.0

            alternatives.extend(sorted(pts, key=_dist))
        frame = None
        while alternatives:
            pt = alternatives.pop(0)
            try:
                w = intersect_horizontal_plane(
                    pt,
                    self.support_z,
                    cam2world,
                    focal,
                    self.size,
                    context=self.context,
                )
                if frame is None:
                    frame = frame_geometry(
                        self.env,
                        self.grounder,
                        image,
                        self.size,
                        self.support_z,
                        context=self.context,
                    )
                depth, c2w, f_px, _, scale = frame
                pred = point_world(
                    depth, pt, scale, c2w, f_px, self.size, context=self.context
                )
                if self.support_z + 0.06 < float(pred[2]) < self.support_z + 0.28:
                    w = intersect_horizontal_plane(
                        pt,
                        float(pred[2]) - 0.03,
                        cam2world,
                        focal,
                        self.size,
                        context=self.context,
                    )
                    self.source_surface_z = float(pred[2])
                else:
                    self.source_surface_z = self.support_z
            except Exception:
                continue
            if float(
                np.linalg.norm(w[:2] - np.asarray(self.source_world[:2]))
            ) < 0.06 or not in_workspace(w[:2]):
                continue
            return w
        return None
