"""models / grounding_dino: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
import time


@dataclass
class Box:
    """One phrase-grounded detection in absolute pixels of the source image."""

    box_id: str
    bbox: list[float]
    score: float
    label: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return [float((x1 + x2) / 2), float((y1 + y2) / 2)]

    @property
    def base(self):
        """Where the object meets the support surface: bottom-centre of the box.  Projecting a box
        CENTRE onto the table plane overshoots away from the camera for anything with height; the
        bottom edge is the contact line and projects correctly."""
        x1, _y1, x2, y2 = self.bbox
        return [float((x1 + x2) / 2), float(y2)]

    def json_record(self):
        return {
            "box_id": self.box_id,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "center": [round(v, 1) for v in self.center],
            "base": [round(v, 1) for v in self.base],
            "score": round(float(self.score), 4),
            "label": self.label[:40],
            "prompt": self.prompt[:60],
            "metadata": self.metadata,
        }


DEFAULT_GDINO = Path("weights/grounding-dino-tiny")


class GroundingDino:
    """Lazy-loaded HF GroundingDINO.  A failed load is reported, never silently faked."""

    def __init__(
        self,
        model_path=DEFAULT_GDINO,
        device="auto",
        box_threshold=0.15,
        text_threshold=0.15,
    ):
        self.model_path = Path(model_path)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None
        self._device = device
        self._load_seconds = None
        self._infer_seconds = 0.0
        self._calls = 0
        self._error = None

    def _ensure(self):
        if self._model is not None or self._error is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            t0 = time.perf_counter()
            dev = self._device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(str(self.model_path))
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(self.model_path)
            ).to(dev)
            self._model.eval()
            self._dev = dev
            self._load_seconds = round(time.perf_counter() - t0, 2)
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"[:200]

    @staticmethod
    def _as_query(phrases):
        clean = []
        for p in phrases:
            p = " ".join(str(p).lower().split()).strip(" .")
            if p and p not in clean:
                clean.append(p)
        return ". ".join(clean) + "." if clean else "object."

    def detect(self, image, phrases, prefix="G"):
        """Return boxes sorted by descending phrase-alignment score."""
        self._ensure()
        record: dict[str, Any] = {
            "phrases": list(phrases),
            "model": str(self.model_path),
        }
        if self._error is not None:
            record["error"] = self._error
            return ([], record)
        query = self._as_query(phrases)
        record["query"] = query
        t0 = time.perf_counter()
        try:
            inputs = self._processor(images=image, text=query, return_tensors="pt").to(
                self._dev
            )
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            post = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"[:200]
            return ([], record)
        self._infer_seconds += time.perf_counter() - t0
        self._calls += 1
        result = post[0]
        boxes: list[Box] = []
        scores = result.get("scores")
        labels = result.get("text_labels", result.get("labels"))
        bxs = result.get("boxes")
        for i in range(len(scores)):
            b = [float(v) for v in bxs[i].tolist()]
            if b[2] - b[0] < 4 or b[3] - b[1] < 4:
                continue
            lab = labels[i] if isinstance(labels[i], str) else str(labels[i])
            boxes.append(
                Box(
                    box_id=f"{prefix}{len(boxes) + 1:03d}",
                    bbox=b,
                    score=float(scores[i]),
                    label=lab,
                    prompt=query,
                )
            )
        boxes.sort(key=lambda x: -x.score)
        for i, b in enumerate(boxes, 1):
            b.box_id = f"{prefix}{i:03d}"
        record.update(
            {
                "n": len(boxes),
                "seconds": round(time.perf_counter() - t0, 3),
                "boxes": [b.json_record() for b in boxes[:12]],
            }
        )
        return (boxes, record)

    def metadata(self):
        return {
            "model": "GroundingDINO",
            "model_path": str(self.model_path),
            "load_seconds": self._load_seconds,
            "calls": self._calls,
            "inference_seconds_total": round(self._infer_seconds, 2),
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "error": self._error,
            "device": getattr(self, "_dev", None),
        }
