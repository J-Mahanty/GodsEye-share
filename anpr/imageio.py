"""Loading images the way people actually supply them.

A photograph off a phone carries its orientation in an EXIF tag rather than in
the pixels, and OpenCV ignores that tag — so a plate shot in portrait arrives
rotated 90 degrees and nothing downstream can read it. Both the API and the
dashboard load uploads through here.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 2400          # anything larger is downscaled before processing


def load_bytes(raw: bytes, max_side: int = MAX_SIDE) -> np.ndarray | None:
    """Decode an uploaded image to BGR, honouring EXIF rotation. None if unreadable."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception:
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return _shrink(arr, max_side) if arr is not None else None
    return _shrink(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), max_side)


def _shrink(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr
    f = max_side / longest
    return cv2.resize(arr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
