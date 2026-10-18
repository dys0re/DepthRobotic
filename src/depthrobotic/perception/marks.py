"""perception / marks: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.perception.parsing import safe_ask_structured
from depthrobotic.prompts import mark_prompt
from PIL import ImageDraw
from PIL import ImageFont
import re
import string

MAX_MARKS = 12


def _contains(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def filter_boxes(boxes, size=832, max_area_frac=0.12, cap=MAX_MARKS):
    """Presentation filter for style='tag': drop whole-scene boxes, the robot's own body at bottom-centre, and
    'cluster' boxes that enclose the centres of two or more other candidates.  Keeps score order.
    """
    kept = []
    for b in boxes:
        x1, y1, x2, y2 = b.bbox
        if (x2 - x1) * (y2 - y1) > max_area_frac * size * size:
            continue
        cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
        if cy > 0.66 * size and abs(cx - size / 2) < 0.25 * size:
            continue
        kept.append(b)
    centers = [((b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2) for b in kept]
    result = []
    for i, b in enumerate(kept):
        inside = sum(
            (1 for j, c in enumerate(centers) if j != i and _contains(b.bbox, c))
        )
        if inside >= 2:
            continue
        result.append(b)
        if len(result) >= cap:
            break
    return result


MARK_RE = re.compile('"?mark"?\\s*[:=]\\s*"?\\s*(\\d{1,2})')


def parse_mark(value, raw=""):
    """Accepts {"mark": 3}, {"mark": "3"} and, when the JSON was truncated by the token budget (the model
    writes a long reason after the number), the mark digit from the raw text."""
    if isinstance(value, dict):
        try:
            return int(str(value.get("mark")).strip())
        except (TypeError, ValueError):
            pass
    m = MARK_RE.search(raw or "")
    return int(m.group(1)) if m else None


ANCHOR_COLOR = (255, 255, 255)
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/wqy-microhei-fonts/wqy-microhei.ttc",
)


def load_font(px=26):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, px)
        except Exception:
            continue
    return ImageFont.load_default(size=px)


FONT = load_font()


def iou(a, b):
    ix1, iy1, ix2, iy2 = (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _place_tag(draw, placed, x, y, label, color, size):
    tb = draw.textbbox((0, 0), label, font=FONT)
    tw, th = (tb[2] - tb[0] + 10, tb[3] - tb[1] + 8)
    x = min(max(0, x), size - tw)
    y = max(0, y)
    rect = [x, y, x + tw, y + th]
    tries = 0
    while any((iou(rect, p) > 0.0 for p in placed)) and tries < 12:
        rect = (
            [rect[0], rect[1] + th + 2, rect[2], rect[3] + th + 2]
            if tries % 2 == 0
            else [
                rect[0] + tw + 2,
                rect[1] - th - 2,
                rect[2] + tw + 2,
                rect[3] - th - 2,
            ]
        )
        rect = [min(max(0, rect[0]), size - tw), min(max(0, rect[1]), size - th)]
        rect = [rect[0], rect[1], rect[0] + tw, rect[1] + th]
        tries += 1
    placed.append(rect)
    draw.rectangle(rect, fill=color, outline=(0, 0, 0), width=2)
    draw.text(
        (rect[0] + 5 - tb[0], rect[1] + 4 - tb[1]),
        label,
        font=FONT,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


PALETTE = [
    (255, 40, 40),
    (40, 220, 40),
    (40, 160, 255),
    (255, 220, 0),
    (255, 0, 255),
    (0, 230, 230),
    (255, 140, 0),
    (160, 80, 255),
    (0, 180, 120),
    (255, 105, 180),
    (120, 200, 255),
    (200, 200, 60),
]


def draw_marks(image, bboxes, style="circle", anchor_bboxes=None):
    canvas = image.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    size = canvas.width
    if style == "circle":
        r = 17
        for i, bbox in enumerate(bboxes):
            color = PALETTE[i % len(PALETTE)]
            x1, y1, x2, y2 = [float(v) for v in bbox]
            d.rectangle([x1, y1, x2, y2], outline=color, width=3)
            cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
            d.ellipse(
                [cx - r, cy - r, cx + r, cy + r], fill=color, outline=(0, 0, 0), width=2
            )
            label = str(i + 1)
            tb = d.textbbox((0, 0), label, font=FONT)
            tw, th = (tb[2] - tb[0], tb[3] - tb[1])
            d.text(
                (cx - tw / 2 - tb[0], cy - th / 2 - tb[1]),
                label,
                font=FONT,
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        return canvas
    placed: list[list[float]] = []
    for i, bbox in enumerate(anchor_bboxes or []):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        d.rectangle([x1, y1, x2, y2], outline=ANCHOR_COLOR, width=3)
        _place_tag(
            d, placed, x2 - 40, y2 + 2, string.ascii_uppercase[i], (60, 60, 60), size
        )
    for i, bbox in enumerate(bboxes):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        _place_tag(d, placed, x1, y1 - 34, str(i + 1), color, size)
    return canvas


def _ask(grounder, image, prompt, max_tokens):
    raw, value = safe_ask_structured(grounder, image, prompt, max_tokens)
    return (raw, value)


def choose_mark(
    grounder, image, bboxes, instruction, desc, style="circle", max_tokens=100
):
    """Single-step choice.  Returns (0-based index of the chosen box or None on abstain/invalid, record, marked image)."""
    marked = draw_marks(image, bboxes, style=style)
    raw, value = _ask(
        grounder, marked, mark_prompt(instruction, desc, len(bboxes)), max_tokens
    )
    mark = parse_mark(value, raw)
    rec = {
        "n": len(bboxes),
        "mark": mark,
        "response": raw[:160],
        "parsed_json": isinstance(value, dict) and bool(value),
        "reason": str(value.get("reason", ""))[:80] if isinstance(value, dict) else "",
    }
    index = mark - 1 if mark is not None and 1 <= mark <= len(bboxes) else None
    rec["abstained"] = index is None
    return (index, rec, marked)
