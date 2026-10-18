"""perception / matching: v30 control behavior with explicit dependencies."""

from __future__ import annotations
import numpy as np
import torch


def flip_point(point, flipped):
    return [float(point[0]), 1000.0 - float(point[1]) if flipped else float(point[1])]


def consensus_match(reference, current, box, point, side, min_similarity=0.65):
    """Mutual-nearest patches, bounded fixed-camera displacement, translation consensus.

    Inputs are normalized NxD tensors on CPU. Points and boxes use upright 0..1000.
    No forced nearest-neighbor acceptance when occluded or visually ambiguous.
    """
    xy = torch.stack(
        torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1
    )
    xy = (xy.reshape(-1, 2)[:, [1, 0]].float() + 0.5) * (1000.0 / side)
    b = torch.tensor(box)
    mask = ((xy >= b[:2]) & (xy <= b[2:])).all(-1)
    ids = mask.nonzero().flatten()
    rec = {"reference_patches": len(ids), "accepted": False}
    if len(ids) < 4:
        return (None, dict(rec, reason="too_few_reference_patches"))
    similarity = reference[ids] @ current.T
    distance = torch.cdist(xy[ids], xy)
    similarity = similarity.masked_fill(distance > 200.0, -2.0)
    scores, nearest = similarity.max(1)
    reverse = (current[nearest] @ reference.T).argmax(1)
    cycle_error = torch.linalg.vector_norm(xy[reverse] - xy[ids], dim=1)
    keep = (scores >= min_similarity) & (cycle_error <= 1.5 * 1000 / side)
    shifts = xy[nearest] - xy[ids]
    if int(keep.sum()) < 4:
        return (
            None,
            dict(rec, reason="insufficient_mutual_matches", matches=int(keep.sum())),
        )
    median = shifts[keep].median(0).values
    inliers = keep & (
        torch.linalg.vector_norm(shifts - median, dim=1) <= 1.5 * 1000 / side
    )
    rec.update(matches=int(keep.sum()), inliers=int(inliers.sum()))
    if int(inliers.sum()) < max(4, int(keep.sum()) // 2 + 1):
        return (None, dict(rec, reason="inconsistent_motion"))
    shift = shifts[inliers].median(0).values
    result = np.asarray(point) + shift.numpy()
    rec.update(
        accepted=bool(np.all((result >= 0) & (result <= 1000))),
        shift=shift.tolist(),
        similarity=float(scores[inliers].median()),
        point=result.tolist(),
    )
    return (result.tolist() if rec["accepted"] else None, rec)
