"""perception / images: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np


def image_from_buffer(buffer, *, context):
    from PIL import Image as _Image

    array = np.asarray(buffer)
    if context.settings.orient_fix:
        array = array[::-1]
    return _Image.fromarray(np.ascontiguousarray(array)).convert("RGB")


def observation_image(obs, key, *, context):
    return image_from_buffer(obs[key], context=context)


def upright(image, *, context):
    return (
        image
        if context.settings.orient_fix
        else image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    )


def crop_from_normalized(image, point, fraction):
    width, height = image.size
    side = max(32, round(min(width, height) * fraction))
    cx = float(point[0]) / 1000.0 * (width - 1)
    cy = float(point[1]) / 1000.0 * (height - 1)
    left = int(round(np.clip(cx - side / 2, 0, width - side)))
    top = int(round(np.clip(cy - side / 2, 0, height - side)))
    box = (left, top, left + side, top + side)
    return (image.crop(box).resize((832, 832), Image.BILINEAR), box)


def crop_point_to_global(point, box, image_size):
    left, top, right, bottom = box
    width, height = image_size
    x = left + float(point[0]) / 1000.0 * (right - left - 1)
    y = top + float(point[1]) / 1000.0 * (bottom - top - 1)
    return [x / (width - 1) * 1000.0, y / (height - 1) * 1000.0]


def annotate_candidates(image, candidates):
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    width, height = annotated.size
    for label, point in candidates.items():
        x = int(round(point[0] / 1000 * (width - 1)))
        y = int(round(point[1] / 1000 * (height - 1)))
        radius = 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 30, 30),
            outline=(255, 255, 255),
            width=2,
        )
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = (box[2] - box[0], box[3] - box[1])
        tx, ty = (x + 9, y - th - 5)
        draw.rectangle((tx - 2, ty - 2, tx + tw + 2, ty + th + 2), fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return annotated


def raw_image(obs, *, context):
    return image_from_buffer(obs["agentview_image"], context=context)


def point_inside_pixel_crop(point, bounds, image_size):
    if point is None:
        return False
    x = float(point[0]) / 1000.0 * (image_size - 1)
    y = float(point[1]) / 1000.0 * (image_size - 1)
    left, top, right, bottom = bounds
    return left <= x < right and top <= y < bottom


def candidate_lattice(center, radius):
    values = np.linspace(-radius, radius, 5)
    result = {}
    for row, dy in enumerate(values):
        for col, dx in enumerate(values):
            label = f"{chr(ord('A') + row)}{col + 1}"
            result[label] = [
                float(np.clip(center[0] + dx, 20, 980)),
                float(np.clip(center[1] + dy, 20, 980)),
            ]
    return result
