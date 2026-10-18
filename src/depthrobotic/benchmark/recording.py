"""Video recording as an environment observer, with no runtime method patching."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


class EpisodeRecorder:

    def __init__(self, out_dir, settings):
        import imageio.v2 as imageio

        self.imageio = imageio
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.frames = []
        self.key = None

    def start(self, suite, task_id, episode):
        self.frames = []
        self.key = f"{suite}_{task_id}_ep{episode}"

    def observe(self, obs, steps):
        a = Image.fromarray(np.ascontiguousarray(obs["agentview_image"])).convert("RGB")
        w = Image.fromarray(
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
        ).convert("RGB")
        a, w = (x.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for x in (a, w))
        a, w = (
            x.resize((x.width // 2, x.height // 2), Image.Resampling.BILINEAR)
            for x in (a, w)
        )
        canvas = Image.new("RGB", (a.width + w.width, a.height + 28))
        canvas.paste(a, (0, 28))
        canvas.paste(w, (a.width, 28))
        ImageDraw.Draw(canvas).text((6, 6), f"{self.key} step {steps}", fill="white")
        self.frames.append(np.asarray(canvas))

    def finish(self, result):
        if not self.frames:
            return None
        path = self.out_dir / f"{self.key}.mp4"
        self.imageio.mimwrite(path, self.frames, fps=20)
        self.frames = []
        return str(path)
