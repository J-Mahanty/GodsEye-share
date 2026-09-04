"""EasyOCR backend — an evaluated alternative, not a default engine path.

Prompted by a real question: our CRNN (`anpr/crnn.py`) is trained entirely on
synthetic captures and reads 0% of held-out real plates exactly right (see
CLAUDE.md). EasyOCR is a pretrained scene-text reader trained on large
real-world text corpora — the exact domain our CRNN never saw. This wrapper
exists to measure whether that generalises to real plate crops, via
`anpr/compare_backends.py --real`.

Deliberately not wired into anything else: no confidence-floor calibration
(`PlateRead.floor` falls back to the classical scale for this backend, which
is almost certainly wrong for it), no `core/repair.py` integration (EasyOCR
returns text + a per-detection score, not a scoreable lattice over arbitrary
candidate strings the way CRNN's CTC logits are), and no default-engine
wiring in sim/api/dashboard. Construct explicitly via `load()` and pass as
`ANPREngine(easyocr_reader=...)` to use it.
"""
from __future__ import annotations

import numpy as np

from anpr.ocr import PlateRead


def available() -> bool:
    """Is the easyocr package importable? Never assume it's installed."""
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


class EasyOCRReader:
    def __init__(self, reader):
        self._reader = reader

    def read(self, image: np.ndarray) -> PlateRead:
        """Read a plate crop. Mirrors the CRNN/classical PlateRead contract
        so it drops into ANPREngine.read()/fuse_reads() unchanged."""
        try:
            detections = self._reader.readtext(image)
        except Exception as exc:
            return PlateRead(text="", confidence=0.0, plate_found=False, backend="easyocr",
                             reason=f"easyocr raised while reading this crop: {exc}")
        if not detections:
            return PlateRead(text="", confidence=0.0, plate_found=False, backend="easyocr",
                             reason="easyocr found no text in this crop")

        # Reading order top-to-bottom then left-to-right, so a two-row plate
        # (state code above, registration number below) concatenates in the
        # right order instead of by detection-confidence order.
        def order_key(det):
            bbox, _text, _score = det
            cy = sum(p[1] for p in bbox) / 4
            cx = sum(p[0] for p in bbox) / 4
            return (cy, cx)

        parts, scores = [], []
        for bbox, text, score in sorted(detections, key=order_key):
            cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
            if cleaned:
                parts.append(cleaned)
                scores.append(score)
        text = "".join(parts)
        # Weakest detection sets confidence, same idiom as char_confidence
        # elsewhere in this codebase (crnn.py, ocr.py's classical path): one
        # bad segment should not be hidden by an average.
        confidence = float(min(scores)) if scores else 0.0
        return PlateRead(text=text, confidence=confidence, backend="easyocr",
                         variant="easyocr", plate_found=bool(text))


def load() -> EasyOCRReader | None:
    """Construct an EasyOCRReader, or None if easyocr isn't installed."""
    if not available():
        return None
    import easyocr
    return EasyOCRReader(easyocr.Reader(["en"], gpu=False))
