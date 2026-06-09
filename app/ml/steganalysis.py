"""Steganalysis inference: score an image as cover vs stego with a chosen detector."""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from app.ml.device import get_device
from app.registry.analyzers import resolve
from steganalyzers.models import (
    EfficientNetSteg,
    SRNet,
    XuNet,
    YedroudjNet,
    YeNet,
)

_ARCH = {
    "xunet": XuNet,
    "yenet": YeNet,
    "srnet": SRNet,
    "yedroudjnet": YedroudjNet,
    "efficientnetsteg": EfficientNetSteg,
}


@lru_cache(maxsize=12)
def load_analyzer(key: str):
    entry = resolve(key)
    if not entry.available:
        raise FileNotFoundError(f"Weight missing for {key!r}: {entry.path}")
    cls = _ARCH[entry.arch]
    model = cls(in_channels=3, num_classes=2)
    bundle = torch.load(str(entry.path), map_location=get_device(), weights_only=False)
    state = bundle["model_state"] if isinstance(bundle, dict) and "model_state" in bundle else bundle
    model.load_state_dict(state)
    model.eval().to(get_device())
    return model


def analyze(key: str, image_bytes: bytes) -> dict:
    """Return cover/stego probabilities for the image."""
    model = load_analyzer(key)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.asarray(img, dtype="float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(get_device())
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    prob_stego = float(probs[1])
    return {
        "prob_stego": prob_stego,
        "prob_cover": float(probs[0]),
        "label": "stego" if prob_stego >= 0.5 else "cover",
        "logits": [float(v) for v in logits[0]],
    }
