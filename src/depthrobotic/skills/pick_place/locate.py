"""skills / pick_place / locate: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.geometry.camera import camera_geometry, intersect_horizontal_plane
from depthrobotic.geometry.depth import frame_geometry, point_world
from depthrobotic.perception.images import (
    annotate_candidates,
    candidate_lattice,
    crop_from_normalized,
    crop_point_to_global,
    observation_image,
    point_inside_pixel_crop,
)
from depthrobotic.perception.parsing import (
    normalize_box,
    normalize_point,
    safe_ask_structured,
)
from depthrobotic.prompts import crop_center_prompt
from PIL import Image
import json
import numpy as np
import re

DEST_RELATION_RE = re.compile(
    "^(?:on\\s+|at\\s+)?(?:the\\s+)?(?:to\\s+the\\s+)?(?P<rel>left of|right of)\\s+(?:the\\s+)?(?P<anchor>.+?)\\s*$",
    re.I,
)


def parse_dest_relation(phrase):
    """'to the right of the plate' -> {'relation': 'right of', 'anchor': 'plate'}; None for plain destinations."""
    text = re.sub("\\s+", " ", str(phrase or "").strip().lower())
    m = DEST_RELATION_RE.match(text)
    if m is None:
        return None
    return {
        "relation": m.group("rel"),
        "anchor": re.sub("^(the|a|an)\\s+", "", m.group("anchor").strip()),
    }


DEST_RELATION_DIRS = {
    "right of": np.array([0.0, 1.0]),
    "left of": np.array([0.0, -1.0]),
}


def locate_destination(
    env, obs, grounder, phrase, instruction, size, support_z, is_container, *, context
):
    """Locate destination in agentview. is_container drives projection plane.
    Takes 2 samples and averages to reduce single-query instability (~5cm)."""
    image = observation_image(obs, "agentview_image", context=context)
    prompt = f'Robot instruction: {instruction}\nLocate the visual center of the usable receiving region of: {phrase}. For a container choose the interior. Return JSON: {{{{"point":[x,y]}}}}.'
    points = []
    for _ in range(2):
        _, value = safe_ask_structured(grounder, image, prompt, 96)
        p = normalize_point(value.get("point"))
        if p is not None:
            points.append(p)
    if not points:
        raise RuntimeError(f"Could not locate destination: {phrase}")
    if len(points) == 2:
        dist = float(np.linalg.norm(np.array(points[0]) - np.array(points[1])))
        if dist > 80:
            center = [500, 500]
            d0 = float(np.linalg.norm(np.array(points[0]) - np.array(center)))
            d1 = float(np.linalg.norm(np.array(points[1]) - np.array(center)))
            point = points[0] if d0 < d1 else points[1]
        else:
            point = [
                (points[0][0] + points[1][0]) / 2,
                (points[0][1] + points[1][1]) / 2,
            ]
    else:
        point = points[0]
    depth, cam2world, focal, _, scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    predicted = point_world(
        depth, point, scale, cam2world, focal, size, context=context
    )
    surface_z = float(np.clip(predicted[2], support_z - 0.08, support_z + 0.3))
    if surface_z >= support_z + 0.28:
        surface_z = support_z + (0.05 if is_container else 0.0)
    proj_z = surface_z if is_container else support_z
    world = intersect_horizontal_plane(
        point, proj_z, cam2world, focal, size, context=context
    )
    return (point, world, surface_z)


DEST_RELATION_MARGIN = 0.01
FINGER_AXIS = np.array([0.0, 1.0])


def ambiguous_source_description(phrase, instruction):
    source_clause = instruction.lower().split(" and place", maxsplit=1)[0]
    relational_terms = (
        " between ",
        " next to ",
        " from table center",
        " on the ",
        " in the ",
        " near ",
        " left ",
        " right ",
    )
    return any(
        (
            term in f" {phrase.lower()} "
            or (instruction.lower().startswith("pick") and term in f" {source_clause} ")
            for term in relational_terms
        )
    )


def confirm_instance(grounder, image, target, point):
    """Tight-crop re-query: does this crop really show a {target}? Filters hallucinated scan instances."""
    crop, _ = crop_from_normalized(image, point, 0.35)
    prompt = f'Does this image crop clearly show a {target} (the object itself, not just its surroundings)? JSON only: {{"present":true}} or {{"present":false}}.'
    _, value = safe_ask_structured(grounder, crop, prompt, 32)
    return value.get("present") is True


RELATION_RE = re.compile(
    "^(?P<target>.+?)\\s+(?P<rel>between|on top of|on|in|inside|next to|near|beside|by|behind|in front of|to the left of|to the right of|left of|right of)\\s+(?:the\\s+)?(?P<anchor>.+)$",
    re.I,
)


def parse_relation(phrase, instruction):
    """Relation qualifying the SOURCE only. The plan's source phrase is authoritative; the instruction is
    consulted only for a 'pick up the X <rel> the Y' clause (a 'put X on Y' clause names the destination).
    """
    text = re.sub("\\s+", " ", phrase.strip().lower())
    text = re.sub("^(the|a|an)\\s+", "", text)
    m = RELATION_RE.match(text)
    if m is None:
        clause = (
            instruction.lower()
            .split(" and place", 1)[0]
            .split(" and put", 1)[0]
            .strip()
        )
        pick = re.match("^(pick up|take|grab|lift)\\s+(the\\s+)?(.+)$", clause)
        if pick is not None and text.split()[-1] in pick.group(3):
            m = RELATION_RE.match(pick.group(3))
    if m is None:
        return None
    target, rel, anchor = (m.group("target"), m.group("rel"), m.group("anchor"))
    target = re.sub("^(the|a|an)\\s+", "", target.strip())
    anchors = (
        [a.strip() for a in re.split("\\s+and\\s+", anchor)]
        if rel == "between"
        else [anchor.strip()]
    )
    return {
        "target": target,
        "relation": rel,
        "anchors": [re.sub("^(the|a|an)\\s+", "", a) for a in anchors],
    }


def parse_detection_list(response):
    """Parse a Qwen-style detection answer: a JSON list of {"bbox_2d":[x1,y1,x2,y2],...} (fences tolerated).
    Returns instance center points normalized 0..1000."""
    text = response.replace("```json", "").replace("```", "")
    points = []
    start = text.find("[")
    while start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            box = None
            if isinstance(item, dict):
                box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
                if box is None and item.get("point") is not None:
                    p = normalize_point(item.get("point"))
                    if p is not None:
                        points.append(p)
                        continue
            elif isinstance(item, list) and len(item) == 4:
                box = item
            elif isinstance(item, list) and len(item) == 2:
                p = normalize_point(item)
                if p is not None:
                    points.append(p)
                continue
            box = normalize_box(box) if box is not None else None
            if box is not None:
                points.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        break
    if not points:
        m = re.search(
            '"points"\\s*:\\s*\\[\\s*(\\d+(?:\\.\\d+)?)\\s*,\\s*(\\d+(?:\\.\\d+)?)\\s*\\]',
            text,
        )
        if m:
            p = normalize_point([float(m.group(1)), float(m.group(2))])
            if p is not None:
                points.append(p)
    dedup = []
    for p in points:
        if all(
            (float(np.linalg.norm(np.asarray(p) - np.asarray(q))) > 40 for q in dedup)
        ):
            dedup.append(p)
    return dedup[:6]


def list_instances(grounder, image, target, instruction, max_points=6):
    prompt = (
        f'Detect every {target} in this image, including identical-looking duplicates; there may be several. Output a JSON list with one entry per instance, coordinates normalized 0..1000: [{{"bbox_2d":[x1,y1,x2,y2],"label":"'
        + target
        + '"}, ...]'
    )
    response, _ = safe_ask_structured(grounder, image, prompt, 200)
    return (parse_detection_list(response)[:max_points], response)


def relational_source_point(grounder, image, phrase, instruction):
    """Anchor-first resolution. Returns (point or None, record)."""
    parsed = parse_relation(phrase, instruction)
    record = {"parsed": parsed}
    if parsed is None:
        return (None, record)
    anchor_pts = []
    for anchor in parsed["anchors"]:
        prompt = f'Robot instruction: {instruction}\nLocate the visual center of the {anchor} (the reference object, not the object to be moved). JSON only, normalized 0..1000: {{"point":[x,y]}}.'
        _, value = safe_ask_structured(grounder, image, prompt, 96)
        p = normalize_point(value.get("point"))
        if p is not None:
            anchor_pts.append(p)
    record["anchor_points"] = anchor_pts
    if len(anchor_pts) != len(parsed["anchors"]):
        return (None, record)
    ref = np.mean(np.asarray(anchor_pts), axis=0)
    rel = parsed["relation"]
    near_target = None
    if rel in (
        "on",
        "on top of",
        "in",
        "inside",
        "next to",
        "near",
        "beside",
        "by",
        "between",
    ):
        crop, bounds = crop_from_normalized(image, ref.tolist(), 0.45)
        qprompt = f"""Robot instruction: {instruction}\nThis crop is centered on the {' and '.join(parsed['anchors'])}. Locate the visual center of the {parsed['target']} that is {rel} it, if present in this crop. JSON only, normalized 0..1000 within this crop: {{"point":[x,y]}} or {{"point":null}}."""
        _, qv = safe_ask_structured(grounder, crop, qprompt, 64)
        lp = normalize_point(qv.get("point"))
        if lp is not None:
            near_target = crop_point_to_global(lp, bounds, image.size)
    record["near_anchor_target"] = near_target
    instances, resp = list_instances(grounder, image, parsed["target"], instruction)
    detected = list(instances)
    if len(instances) < 2:
        for cx, cy in ((300, 350), (700, 350), (300, 700), (700, 700)):
            crop, bounds = crop_from_normalized(image, [cx, cy], 0.6)
            qprompt = f"""Is there a {parsed['target']} in this image crop? If yes, point to the center of each one. JSON only, normalized 0..1000 within this crop: {{"points":[[x,y]]}} or {{"points":[]}}."""
            _, qv = safe_ask_structured(grounder, crop, qprompt, 120)
            raw = qv.get("points") if isinstance(qv, dict) else None
            cand = []
            if isinstance(raw, list):
                if len(raw) == 2 and all((isinstance(v, (int, float)) for v in raw)):
                    raw = [raw]
                for item in raw:
                    lp = normalize_point(item if isinstance(item, list) else None)
                    if lp is not None:
                        cand.append(crop_point_to_global(lp, bounds, image.size))
            for gp in cand:
                if all(
                    (
                        float(np.linalg.norm(np.asarray(gp) - np.asarray(q))) > 60
                        for q in instances
                    )
                ):
                    instances.append(gp)
        record["quadrant_scan"] = True
        confirmed = [
            p
            for p in instances
            if p in detected or confirm_instance(grounder, image, parsed["target"], p)
        ]
        record["instances_unconfirmed"] = [p for p in instances if p not in confirmed]
        instances = confirmed
    if near_target is not None and all(
        (
            float(np.linalg.norm(np.asarray(near_target) - np.asarray(q))) > 60
            for q in instances
        )
    ):
        instances.insert(0, near_target)
    record["instances"] = instances
    record["instances_response"] = resp[:160]
    if near_target is not None:
        others = [
            q
            for q in instances
            if float(np.linalg.norm(np.asarray(near_target) - np.asarray(q))) > 60
        ]
        others.sort(key=lambda q: float(np.linalg.norm(np.asarray(q) - ref)))
        record.update(
            {"chosen": near_target, "chosen_by": "anchor_crop", "alternatives": others}
        )
        return (near_target, record)
    if len(instances) < 2:
        return (None, record)
    if rel in ("on", "on top of", "in", "inside"):
        scored = [
            (
                float(np.linalg.norm(np.asarray(p) - ref))
                + (60.0 if p[1] > ref[1] + 40 else 0.0),
                p,
            )
            for p in instances
        ]
    else:
        scored = [(float(np.linalg.norm(np.asarray(p) - ref)), p) for p in instances]
    scored.sort(key=lambda t: t[0])
    best_d, best = scored[0]
    record.update(
        {
            "chosen": best,
            "chosen_anchor_distance": best_d,
            "runner_up_distance": scored[1][0] if len(scored) > 1 else None,
            "alternatives": [p for _, p in scored[1:]],
        }
    )
    if (
        rel in ("on", "on top of", "in", "inside", "next to", "near", "beside", "by")
        and best_d > 320
    ):
        record["rejected_far_from_anchor"] = True
        return (None, record)
    return (best, record)


def container_destination(destination, instruction):
    text = f" {destination.lower()} {instruction.lower()} "
    container_words = (
        " in ",
        " inside ",
        " basket",
        " drawer",
        " microwave",
        " compartment",
        " caddy",
    )
    return any((word in text for word in container_words))


def marked_grid_refine(
    grounder, image, instruction, phrase, role, local_prompt, anchor_point=None
):
    candidates = candidate_lattice([500.0, 500.0], 400.0)
    marked = annotate_candidates(image, candidates)
    if role == "source":
        prompt = f"""Robot instruction: {instruction}\nThe full scene has 25 labeled red markers. Select the marker closest to the visual center of the exact source object that must be moved. Use all spatial relations in the instruction to reject same-category distractors. Valid labels are: {', '.join(candidates)}. Return JSON only, for example: {{"label":"C3"}}."""
    else:
        prompt = f"""Robot instruction: {instruction}\nThe full scene has 25 labeled red markers. Select the marker closest to the center of the usable receiving region of {phrase}. Use spatial relations to reject distractors. Valid labels are: {', '.join(candidates)}. Return JSON only: {{"label":"C3"}}."""
    select_response, select_value = safe_ask_structured(grounder, marked, prompt, 64)
    label = str(select_value.get("label", "")).upper().strip()
    center = candidates.get(label)
    record = {
        "select_response": select_response,
        "selected_label": label,
        "selected_center": center,
    }
    if role == "source" and anchor_point is not None:
        nearest_label = min(
            candidates,
            key=lambda candidate: float(
                np.linalg.norm(
                    np.asarray(candidates[candidate]) - np.asarray(anchor_point)
                )
            ),
        )
        nearest_center = candidates[nearest_label]
        selected_distance = (
            float(np.linalg.norm(np.asarray(center) - np.asarray(anchor_point)))
            if center is not None
            else float("inf")
        )
        record.update(
            {
                "anchor_point": anchor_point,
                "nearest_anchor_label": nearest_label,
                "selected_anchor_distance": selected_distance,
            }
        )
        if selected_distance > 150.0:
            center = nearest_center
            record["selected_center"] = center
            record["anchor_consensus_override"] = True
    if center is None:
        return (None, record)
    crop, bounds = crop_from_normalized(image, center, 0.42)
    refine_response, refine_value = safe_ask_structured(
        grounder, crop, local_prompt, 96
    )
    local = normalize_point(refine_value.get("point"))
    refined = None if local is None else crop_point_to_global(local, bounds, image.size)
    record.update(
        {
            "crop_bounds": list(bounds),
            "refine_response": refine_response,
            "refined_point": refined,
        }
    )
    return (refined, record)


def global_point_with_depth(
    env, obs, grounder, phrase, instruction, size, support_z, role, *, context
):
    image = observation_image(obs, "agentview_image", context=context)
    if role == "source":
        coarse_prompt = f'Robot instruction: {instruction}\nLocate the visual center of only the current source: {phrase}. Resolve spatial distractors using the full instruction. Return JSON only with one point normalized 0..1000: {{"point":[x,y]}}.'
        crop_prompt_text = f'Robot instruction: {instruction}\nThis crop is around the coarse location of the current source: {phrase}. Locate its visual center. Return JSON only with one point normalized 0..1000 within this crop: {{"point":[x,y]}}.'
    else:
        coarse_prompt = f'Robot instruction: {instruction}\nLocate the visual center of the exact usable receiving region of the current destination: {phrase}. For a basket/container choose the interior; for a stove choose the burner/cooking surface rather than its base or knob; for a divided holder choose the specified compartment. Return JSON only with one point normalized 0..1000: {{"point":[x,y]}}.'
        crop_prompt_text = f'Instruction: {instruction}\nThis is a crop around the coarse destination. Locate the visual center of the usable receiving region of {phrase}. For a container choose its interior. Return JSON only with a point normalized 0..1000 in this crop: {{"point":[x,y]}}.'
    coarse_response, coarse_value = safe_ask_structured(
        grounder, image, coarse_prompt, 96
    )
    coarse = normalize_point(coarse_value.get("point"))
    if coarse is None:
        raise RuntimeError(f"Invalid global {role} point: {coarse_response}")
    crop, crop_bounds = crop_from_normalized(image, coarse, 0.5)
    crop_response, crop_value = safe_ask_structured(
        grounder, crop, crop_prompt_text, 96
    )
    local = normalize_point(crop_value.get("point"))
    direct_refined = (
        None if local is None else crop_point_to_global(local, crop_bounds, image.size)
    )
    point = coarse
    direct_refined_rejected = False
    if direct_refined is not None:
        refinement_distance = float(
            np.linalg.norm(np.asarray(direct_refined) - np.asarray(coarse))
        )
        boundary_clipped = (
            crop_bounds[0] == 0
            or crop_bounds[1] == 0
            or crop_bounds[2] == image.width
            or (crop_bounds[3] == image.height)
        )
        if role == "destination" and boundary_clipped and (refinement_distance > 30.0):
            direct_refined_rejected = True
        elif refinement_distance <= (120.0 if role == "destination" else 220.0):
            point = direct_refined
        else:
            direct_refined_rejected = True
    grid_record = None
    if role == "source" and ambiguous_source_description(phrase, instruction):
        grid_local_prompt = (
            crop_center_prompt(instruction) if role == "source" else crop_prompt_text
        )
        grid_point, grid_record = marked_grid_refine(
            grounder,
            image,
            instruction,
            phrase,
            role,
            grid_local_prompt,
            anchor_point=coarse,
        )
        if grid_point is not None:
            if (
                float(np.linalg.norm(np.asarray(grid_point) - np.asarray(coarse)))
                <= 220.0
            ):
                point = grid_point
            else:
                grid_record["rejected_inconsistent_with_coarse"] = True
    depth, camera_to_world, focal_px, _, depth_scale = frame_geometry(
        env, grounder, image, size, support_z, context=context
    )
    predicted = point_world(
        depth, point, depth_scale, camera_to_world, focal_px, size, context=context
    )
    predicted_surface_z = float(
        np.clip(predicted[2], support_z - 0.08, support_z + 0.3)
    )
    if role == "destination" and "basket" in phrase.lower():
        surface_z = support_z + 0.05
    elif role == "destination" and "bowl" in phrase.lower():
        surface_z = support_z + 0.02
    else:
        surface_z = predicted_surface_z
    projection_z = (
        surface_z
        if role == "destination" and container_destination(phrase, instruction)
        else support_z
    )
    world = intersect_horizontal_plane(
        point, projection_z, camera_to_world, focal_px, size, context=context
    )
    record = {
        "role": role,
        "phrase": phrase,
        "coarse_point": coarse,
        "direct_refined_point": direct_refined,
        "direct_refined_rejected": direct_refined_rejected,
        "refined_point": point,
        "coarse_response": coarse_response,
        "crop_response": crop_response,
        "crop_bounds": list(crop_bounds),
        "marked_grid": grid_record,
        "depth_scale": depth_scale,
        "predicted_depth_world": predicted.tolist(),
        "predicted_surface_z": predicted_surface_z,
        "surface_z": surface_z,
        "projection_z": projection_z,
        "world": world.tolist(),
    }
    return (point, world, surface_z, record)


def name_tokens(text):
    ignored = {"the", "and", "box", "bottle", "can", "package"}
    return {
        token
        for token in re.findall("[a-z]+", text.lower())
        if len(token) >= 3 and token not in ignored
    }


def ocr_verified_packaged_source(
    env,
    obs,
    grounder,
    source,
    instruction,
    size,
    support_z,
    direct_point,
    tile_zero_record,
    *,
    context,
):
    """Select a package only when target-free crop text matches its name."""
    image = observation_image(obs, "agentview_image", context=context)
    width, height = image.size
    side = round(min(width, height) * 0.62)
    boxes = (
        (0, 0, side, side),
        (width - side, 0, width, side),
        (0, height - side, side, height),
        (width - side, height - side, width, height),
    )
    candidates = [("direct", direct_point)]
    proposal_log = []
    if tile_zero_record.get("point") is not None:
        candidates.append(("tile_0", tile_zero_record["point"]))
    for tile_index, bounds in enumerate(boxes[1:], start=1):
        tile = image.crop(bounds).resize(image.size, Image.BILINEAR)
        response, value = safe_ask_structured(
            grounder,
            tile,
            f'Robot instruction: {instruction}\nThis is an overlapping magnified region of the original RGB scene. Locate the visual center of the exact source product {source}. Reject similar packages by label, color, and shape. Return JSON only with a point normalized 0..1000 in this image: {{"point":[x,y]}}.',
            96,
        )
        local = normalize_point(value.get("point"))
        point = (
            None if local is None else crop_point_to_global(local, bounds, image.size)
        )
        proposal_log.append(
            {
                "tile": tile_index,
                "bounds": list(bounds),
                "response": response,
                "local_point": local,
                "point": point,
            }
        )
        if point is not None:
            candidates.append((f"tile_{tile_index}", point))
    required_tokens = name_tokens(source)
    observations = []
    for label, point in candidates:
        crop, bounds = crop_from_normalized(image, point, 0.28)
        response, value = safe_ask_structured(
            grounder,
            crop,
            "Identify the single physical object at the exact center of this RGB crop. Do not use any robot task or requested target name. Transcribe all visible package text. Return JSON only with keys object_name and visible_text.",
            96,
        )
        observed = f"{value.get('object_name', '')} {value.get('visible_text', '')}"
        observed_tokens = name_tokens(observed)
        overlap = len(required_tokens & observed_tokens)
        observations.append(
            {
                "label": label,
                "point": point,
                "response": response,
                "parsed": value,
                "observed_tokens": sorted(observed_tokens),
                "source_token_overlap": overlap,
                "crop_bounds": list(bounds),
            }
        )
    positive = [entry for entry in observations if entry["source_token_overlap"] > 0]
    selected = (
        max(positive, key=lambda entry: entry["source_token_overlap"])
        if positive
        else None
    )
    record = {
        "method": "target_free_crop_ocr_name_match",
        "source_tokens": sorted(required_tokens),
        "proposals": proposal_log,
        "observations": observations,
        "selected": selected,
    }
    if selected is None:
        return (None, record)
    camera_to_world, focal_px, fovy = camera_geometry(env, "agentview", size)
    world = intersect_horizontal_plane(
        selected["point"], support_z, camera_to_world, focal_px, size, context=context
    )
    record.update({"world": world.tolist(), "fovy": fovy})
    return (world, record)


def retry_tile_source_grounding(
    env, obs, grounder, source, instruction, size, support_z, *, context
):
    """Use a magnified overlapping view as an alternate source hypothesis."""
    image = observation_image(obs, "agentview_image", context=context)
    width, height = image.size
    side = round(min(width, height) * 0.62)
    bounds = (0, 0, side, side)
    tile = image.crop(bounds).resize(image.size, Image.BILINEAR)
    response, value = safe_ask_structured(
        grounder,
        tile,
        f'Robot instruction: {instruction}\nThis is a magnified overlapping region of the original RGB scene. Locate the visual center of the exact source object {source}. Reject similar packages using label, color, and shape. Return JSON only with a point normalized 0..1000 in this image: {{"point":[x,y]}}.',
        96,
    )
    local_point = normalize_point(value.get("point"))
    point = (
        None
        if local_point is None
        else crop_point_to_global(local_point, bounds, image.size)
    )
    record = {
        "method": "magnified_overlapping_rgb_retry",
        "response": response,
        "local_point": local_point,
        "crop_bounds": list(bounds),
        "point": point,
    }
    if point is None:
        return (None, record)
    camera_to_world, focal_px, fovy = camera_geometry(env, "agentview", size)
    world = intersect_horizontal_plane(
        point, support_z, camera_to_world, focal_px, size, context=context
    )
    record.update({"world": world.tolist(), "fovy": fovy})
    return (world, record)


def locate_source(
    env, obs, grounder, phrase, instruction, size, support_z, log=None, *, context
):
    """Source grounding = v6's name-free pipeline, with the OCR verification UNGATED.

    1. global_point_with_depth(role="source"): coarse -> crop refine -> 5x5 marked-grid
       refinement for relational phrases (between/next to/on the/...), consistency-checked.
    2. retry_tile_source_grounding: magnified-tile re-grounding, accepted only when the direct
       point lies inside the magnified crop and the two disagree strongly (v6 rule).
    3. ocr_verified_packaged_source: target-name-free crop OCR + token match. In v6 this was
       gated on '"basket" in destination' -- the gate was the hack, the mechanism is generic,
       so it now runs for every source and is accepted only when visible text matches.
    Returns (point, world, surface_z). surface_z has no per-category override for role=source.
    """
    point, world, surface_z, rec = global_point_with_depth(
        env,
        obs,
        grounder,
        phrase,
        instruction,
        size,
        support_z,
        "source",
        context=context,
    )
    method = "grid_or_crop"
    mag_world, mag_rec = retry_tile_source_grounding(
        env, obs, grounder, phrase, instruction, size, support_z, context=context
    )
    mag_point, direct_point = (mag_rec.get("point"), rec.get("refined_point"))
    if mag_world is not None and mag_point is not None and (direct_point is not None):
        disagreement = float(
            np.linalg.norm(np.asarray(mag_point) - np.asarray(direct_point))
        )
        if (
            point_inside_pixel_crop(direct_point, mag_rec["crop_bounds"], size)
            and disagreement > 150.0
        ):
            world = mag_world
            method = "magnified_tile"
    if direct_point is not None and (
        not ambiguous_source_description(phrase, instruction)
    ):
        ocr_world, ocr_rec = ocr_verified_packaged_source(
            env,
            obs,
            grounder,
            phrase,
            instruction,
            size,
            support_z,
            direct_point,
            mag_rec,
            context=context,
        )
        if ocr_world is not None:
            world = ocr_world
            method = "ocr_token_match"
    if log is not None:
        log.append(
            {
                "source_grounding_method": method,
                "grid_used": rec.get("marked_grid") is not None,
            }
        )
    return (point, world, surface_z)


def locate_relational_source(
    env, obs, grounder, phrase, instruction, size, support_z, log=None, *, context
):
    """Relational phrases: anchor-first resolver, refined by a crop and projected with the depth frame.
    Everything else (and any failure above): v7's name-free pipeline (grid/tile/OCR)."""
    context.source_candidates.clear()
    if ambiguous_source_description(phrase, instruction):
        image = observation_image(obs, "agentview_image", context=context)
        point, rec = relational_source_point(grounder, image, phrase, instruction)
        if point is not None:
            context.source_candidates.extend(rec.get("alternatives", []))
            crop, bounds = crop_from_normalized(image, point, 0.32)
            crop_prompt = f'Robot instruction: {instruction}\nThis crop is centered on the {phrase}. Locate its visual center. JSON only, normalized 0..1000 within this crop: {{"point":[x,y]}}.'
            _, cv = safe_ask_structured(grounder, crop, crop_prompt, 96)
            local = normalize_point(cv.get("point"))
            refined = (
                crop_point_to_global(local, bounds, image.size)
                if local is not None
                else None
            )
            if (
                refined is not None
                and float(np.linalg.norm(np.asarray(refined) - np.asarray(point)))
                <= 120
            ):
                point = refined
            depth, cam2world, focal, _, scale = frame_geometry(
                env, grounder, image, size, support_z, context=context
            )
            elevated = False
            try:
                predicted = point_world(
                    depth, point, scale, cam2world, focal, size, context=context
                )
                surface_z = float(
                    np.clip(predicted[2], support_z - 0.08, support_z + 0.3)
                )
            except Exception:
                predicted = None
                surface_z = support_z
            if (
                predicted is not None
                and surface_z > support_z + 0.06
                and (surface_z < support_z + 0.28)
            ):
                elevated = True
                world = intersect_horizontal_plane(
                    point, surface_z - 0.03, cam2world, focal, size, context=context
                )
            else:
                world = intersect_horizontal_plane(
                    point, support_z, cam2world, focal, size, context=context
                )
            rec["elevated"] = elevated
            if log is not None:
                log.append(
                    {
                        "source_grounding_method": "anchor_relational",
                        "grid_used": False,
                        "relational": rec,
                    }
                )
            return (point, world, surface_z)
        if log is not None:
            log.append({"relational_resolver_fallback": rec})
    return locate_source(
        env,
        obs,
        grounder,
        phrase,
        instruction,
        size,
        support_z,
        log=log,
        context=context,
    )


STRONG_IDENTITY = {"confirmed_by_label", "accepted_by_name", "accepted"}
ADJUSTMENT_OFFSETS = {
    "lower": (0.0, 0.0, -0.02),
    "higher": (0.0, 0.0, 0.02),
    "shift_left": (0.0, 0.015, 0.0),
    "shift_right": (0.0, -0.015, 0.0),
    "shift_forward": (0.015, 0.0, 0.0),
    "shift_back": (-0.015, 0.0, 0.0),
}
STRATEGY_DROP = {"rim": 0.008, "side": 0.035, "center": 0.035, "top": 0.018}
MAX_FEASIBLE_WIDTH = 0.095
GRIPPER_OPENING = 0.08
RIM_INSET = 0.003
RETRY_HEIGHT = {
    "rim": (0.0, 0.01, -0.01),
    "side": (0.0, -0.03, 0.015),
    "center": (0.0, -0.03, 0.015),
    "top": (0.0, -0.012, 0.01),
}
RELEASE_RETREAT = 0.04
RELEASE_MIN_SEP_PX = 60.0
RELEASE_GAP_ATTACHED = 0.065
RELEASE_PINCH_EMPTY = 0.003
STRATEGY_PARAMS = {
    "rim": {
        "candidate_radius": 55.0,
        "exclude_center": True,
        "min_depth_height": 0.0,
        "retry_offsets": (0.0, -0.035, 0.035),
    },
    "side": {
        "candidate_radius": 75.0,
        "exclude_center": False,
        "min_depth_height": 0.0,
        "retry_offsets": (0.0, 0.035, 0.07),
    },
    "top": {
        "candidate_radius": 75.0,
        "exclude_center": False,
        "min_depth_height": -0.025,
        "retry_offsets": (0.0, -0.035, 0.035),
    },
    "center": {
        "candidate_radius": 75.0,
        "exclude_center": False,
        "min_depth_height": 0.0,
        "retry_offsets": (0.0, 0.035, 0.07),
    },
}


def make_grasp_candidates(center, strategy):
    """Generate labeled grasp candidate points around center, parameterized by strategy."""
    params = STRATEGY_PARAMS[strategy]
    radius = params["candidate_radius"]
    diag = radius / np.sqrt(2.0)
    offsets = {
        "C": (0.0, 0.0),
        "N": (0.0, -radius),
        "NE": (diag, -diag),
        "E": (radius, 0.0),
        "SE": (diag, diag),
        "S": (0.0, radius),
        "SW": (-diag, diag),
        "W": (-radius, 0.0),
        "NW": (-diag, -diag),
    }
    result = {
        label: [
            float(np.clip(center[0] + dx, 15, 985)),
            float(np.clip(center[1] + dy, 15, 985)),
        ]
        for label, (dx, dy) in offsets.items()
    }
    if params["exclude_center"]:
        result.pop("C", None)
    return result


def select_grasp_point(
    grounder, obs, source, instruction, center, candidates, *, context
):
    """Let model select from candidates. No forced override."""
    marked = annotate_candidates(
        observation_image(obs, "robot0_eye_in_hand_image", context=context), candidates
    )
    valid_labels = ", ".join(candidates)
    prompt = f'Robot instruction: {instruction}\nThis wrist image has labeled red candidate points near {source}. Select the best point for a top-down parallel-jaw grasp. For open containers grasp a rim point; for solid objects grasp near the center. Ignore gripper fingers. Valid labels: {valid_labels}. Return JSON only: {{"label":"..."}}'
    _, value = safe_ask_structured(grounder, marked, prompt, 64)
    label = str(value.get("label", "")).upper().strip()
    return candidates.get(label, center)


WRIST_SERVO_ITERS = 2
