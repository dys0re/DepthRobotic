"""models / eupe: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from PIL import Image
from pathlib import Path
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F


class EUPEFeatures:

    def __init__(self, repo, checkpoint, resolution=512, device="cuda"):
        sys.path.insert(0, str(Path(repo).resolve()))
        from eupe.hub.backbones import eupe_vitb16

        self.model = eupe_vitb16(pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        loaded = self.model.load_state_dict(state, strict=False)
        if loaded.missing_keys:
            raise RuntimeError(f"EUPE missing backbone weights: {loaded.missing_keys}")
        self.model.eval().to(device)
        self.device, self.resolution = (device, resolution)
        self.calls, self.seconds = (0, 0.0)
        self.extra_keys = loaded.unexpected_keys

    @torch.inference_mode()
    def features(self, image):
        start = time.perf_counter()
        a = np.asarray(
            image.convert("RGB").resize(
                (self.resolution, self.resolution), Image.Resampling.BICUBIC
            )
        ).copy()
        x = torch.from_numpy(a).permute(2, 0, 1).float().div(255).to(self.device)
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        with torch.autocast(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        ):
            y = self.model.forward_features(((x - mean) / std)[None])[
                "x_norm_patchtokens"
            ][0]
        result = F.normalize(y.float(), dim=-1).cpu()
        if not torch.isfinite(result).all():
            raise RuntimeError("EUPE nonfinite features")
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return result

    def metadata(self):
        return {
            "model": "EUPE-ViT-B",
            "resolution": self.resolution,
            "calls": self.calls,
            "seconds": self.seconds,
            "extra_checkpoint_keys": self.extra_keys,
        }
