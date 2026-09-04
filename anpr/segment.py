"""Plate normalisation, binarisation and character segmentation.

Deliberately kept as a standalone module: both the OCR engine and the glyph
model trainer need the exact same segmentation, so the classifier is trained on
the distribution it will actually see at inference time.
"""
from __future__ import annotations

import cv2
import numpy as np

import config

NORM_H = config.PLATE_NORM_HEIGHT
G = config.GLYPH_SIZE


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def deskew(gray: np.ndarray) -> np.ndarray:
    """Undo the in-plane rotation of an angled capture using the ink cloud."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    pts = cv2.findNonZero(th)
    if pts is None or len(pts) < 40:
        return gray
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.6 or abs(angle) > 20:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def normalize(img: np.ndarray) -> np.ndarray:
    """Grayscale -> deskewed, contrast-equalised, fixed-height plate crop.

    A single-row Indian plate is about 4.5:1; a two-row motorcycle plate is
    nearer 2:1. Normalising both to the same pixel height would leave the
    two-row characters half the size, so the target height follows the aspect
    ratio and every character arrives at the classifier about as tall.
    """
    gray = to_gray(img)
    h, w = gray.shape
    if h < 8 or w < 24:
        gray = cv2.resize(gray, (max(w, 96), max(h, 24)))
        h, w = gray.shape
    target = NORM_H if w / max(h, 1) >= 3.0 else int(NORM_H * 1.7)
    scale = target / h
    gray = cv2.resize(gray, (max(32, int(w * scale)), target), interpolation=cv2.INTER_CUBIC)
    gray = deskew(gray)
    gray = cv2.bilateralFilter(gray, 5, 45, 45)          # denoise, keep edges
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    return gray


def flatten_illumination(norm: np.ndarray, k: int = 15) -> np.ndarray:
    """Divide out the background field.

    A morphological closing with a kernel wider than a character estimates the
    plate background including mud, shadow and glare gradients; dividing by it
    leaves the strokes and almost nothing else. This is what makes dirty and
    night captures readable.
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.GaussianBlur(cv2.morphologyEx(norm, cv2.MORPH_CLOSE, ker), (0, 0), 3)
    flat = norm.astype(np.float32) / (bg.astype(np.float32) + 1.0) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def binarize(norm: np.ndarray) -> np.ndarray:
    """Default single-hypothesis ink mask (illumination-flattened Otsu).

    Plate characters are always darker than their local background - even on a
    yellow commercial plate or an under-exposed night capture - so no polarity
    guess is needed.
    """
    flat = flatten_illumination(norm)
    blur = cv2.GaussianBlur(flat, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return _post(ink)


def _clear_frame(mask: np.ndarray) -> np.ndarray:
    """Erase the plate frame without erasing characters it happens to touch:
    only border-hugging components that are hollow, full-width or full-height."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue
        fill = area / max(bw * bh, 1)
        frame_like = ((bw > 0.80 * w and fill < 0.45)      # the surrounding border
                      or bh >= 0.97 * h                     # a full-height edge
                      or (bw > 0.35 * w and bh < 0.20 * h))  # a top or bottom rail
        # The blue IND band of a High Security plate: a solid, tall, narrow bar
        # against one edge. It is not part of the registration, and left in it
        # costs ten points of accuracy by eating the first character.
        band_like = bh > 0.70 * h and bw < 0.22 * w and fill > 0.70
        if frame_like or band_like:
            out[labels == i] = 0
    return out


def correct_shear(ink: np.ndarray) -> np.ndarray:
    """Remove the slant an off-axis camera puts on the characters.

    A plate shot from the side leaves upright strokes leaning; the vertical ink
    projection then has no clean valleys between characters and segmentation
    fuses them. Searching the shear that maximises the energy of the projection
    profile (i.e. the deepest gaps) puts the columns back upright.
    """
    h, w = ink.shape
    best, best_score = None, -1.0
    for shear in np.arange(-0.45, 0.46, 0.075):
        m = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        cand = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST) if abs(shear) > 1e-6 else ink
        proj = cand.sum(axis=0).astype(np.float32) / 255.0
        score = float(np.mean(proj ** 2))          # peaky profile == clean gaps
        if score > best_score:
            best, best_score = cand, score
    return best


def ink_variants(norm: np.ndarray):
    """Yield (name, ink mask) for several plausible binarisations.

    No single threshold survives every condition - glare wants a local method,
    mud wants illumination flattening, a low-resolution crop wants upscaling
    first. The engine decodes all of them and keeps the most confident read,
    which is far cheaper than being clever about picking one up front.
    """
    up = cv2.resize(norm, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    for name, src, k in (("flat15", norm, 15), ("flat25", norm, 25), ("flat15x2", up, 27)):
        flat = flatten_illumination(src, k)
        blur = cv2.GaussianBlur(flat, (3, 3), 0)
        _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        yield name, _post(ink)

    _, otsu = cv2.threshold(cv2.GaussianBlur(norm, (3, 3), 0), 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    yield "otsu", _post(otsu)

    adap = cv2.adaptiveThreshold(cv2.GaussianBlur(norm, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 14)
    yield "adaptive", _post(adap)

    # Three more aimed at the low-contrast case - a plate under a film of dust,
    # where the ink and the background are only a few grey levels apart. Each
    # earns its place on the benchmark: together they lift the dirty and mixed
    # conditions without costing anything on the clean ones.
    lo, hi = np.percentile(norm, [4, 96])
    if hi > lo + 4:
        stretched = np.clip((norm.astype(np.float32) - lo) * 255.0 / (hi - lo),
                            0, 255).astype(np.uint8)
        _, ink = cv2.threshold(cv2.GaussianBlur(stretched, (3, 3), 0), 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        yield "stretch", _post(ink)

    tight = flatten_illumination(norm, 9)     # kernel just wider than a stroke
    _, ink = cv2.threshold(cv2.GaussianBlur(tight, (3, 3), 0), 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    yield "flat9", _post(ink)

    yield "mean41", _post(cv2.adaptiveThreshold(
        cv2.GaussianBlur(norm, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 41, 8))


def _post(ink: np.ndarray) -> np.ndarray:
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # Close vertically only: rejoins the broken strokes of a blurred glyph
    # without bridging neighbouring characters into one blob.
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    return correct_shear(_clear_frame(ink))


def segment(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Ordered character atoms (x, y, w, h): top row left-to-right, then the
    second row if the plate has one."""
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # loose enough to keep the characters of a two-row plate, where each is
        # only about a third of the crop height
        if bh < 0.16 * h or bh > 0.98 * h:
            continue
        if bw < 0.010 * w or bw > 0.45 * w:            # too thin / a whole word
            continue
        if area < 0.08 * bw * bh or area < 20:         # hollow smears, speckle
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    if not boxes:
        return []

    out: list[tuple[int, int, int, int]] = []
    for row in _rows(boxes, h):
        row = _split_merged(_drop_odd_heights(row), ink)
        row.sort(key=lambda b: b[0])
        out.extend(row)
    return out[:20]


def _rows(boxes, img_h):
    """Cluster boxes into horizontal text lines by vertical centre.

    Single-row plates yield one cluster; a motorcycle plate yields two, and the
    IND side band or a stray reflection yields a small cluster that is dropped.
    """
    heights = sorted(b[3] for b in boxes)
    hmed = heights[len(heights) // 2] or 1
    ordered = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    clusters: list[list] = [[ordered[0]]]
    for box in ordered[1:]:
        prev = clusters[-1][-1]
        if (box[1] + box[3] / 2) - (prev[1] + prev[3] / 2) > 0.62 * hmed:
            clusters.append([box])
        else:
            clusters[-1].append(box)
    # keep the substantial lines only, in top-to-bottom order
    keep = [c for c in clusters if len(c) >= 2 and
            sum(b[2] * b[3] for b in c) >= 0.10 * sum(b[2] * b[3] for b in boxes)]
    if not keep:
        keep = [max(clusters, key=len)]
    keep.sort(key=lambda c: np.median([b[1] + b[3] / 2 for b in c]))
    return keep[:2]


def _drop_odd_heights(row):
    """Within one line, characters are all about the same height."""
    hs = sorted(b[3] for b in row)
    hmed = hs[len(hs) // 2] or 1
    kept = [b for b in row if 0.55 * hmed <= b[3] <= 1.7 * hmed]
    return kept or row


def _split_merged(boxes, ink):
    """Cut wide blobs at the minima of the vertical ink projection.

    Deliberately aggressive: these boxes are *atoms*, not characters. The OCR
    decoder can merge neighbouring atoms back into one character but can never
    invent a cut that was not offered, so over-cutting is the cheap error.
    """
    if not boxes:
        return []
    # Estimate a single character's width from the boxes that already look like
    # one character (taller than wide); merged blobs must not skew the median.
    singles = [b[2] for b in boxes if b[2] <= 0.95 * b[3]]
    widths = sorted(singles or [b[2] for b in boxes])
    wmed = widths[len(widths) // 2]
    out = []
    for (x, y, bw, bh) in sorted(boxes, key=lambda b: b[0]):
        k = min(4, int(round(bw / max(wmed, 1))))
        # Only cut a blob that is far wider than one character. M and W are
        # legitimately wide, and cutting one produces two entirely convincing
        # letters that the decoder has no way to argue with.
        if k < 2 or bw < 1.4 * wmed or bw / max(bh, 1) < 1.15:
            out.append((x, y, bw, bh))
            continue
        strip = ink[y:y + bh, x:x + bw]
        proj = strip.sum(axis=0).astype(np.float32)
        cuts = []
        for j in range(1, k):
            target = int(j * bw / k)
            lo, hi = max(1, target - bw // (2 * k)), min(bw - 1, target + bw // (2 * k))
            cuts.append(lo + int(np.argmin(proj[lo:hi])) if hi > lo else target)
        edges = [0, *cuts, bw]
        for a, b in zip(edges, edges[1:]):
            if b - a >= 3:
                out.append((x + a, y, b - a, bh))
    return out


def glyph(ink: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop one character and normalise it into a GxG feature vector."""
    x, y, w, h = box
    crop = ink[y:y + h, x:x + w]
    side = max(w, h)
    pad_x, pad_y = (side - w) // 2, (side - h) // 2
    square = np.zeros((side, side), np.uint8)
    square[pad_y:pad_y + h, pad_x:pad_x + w] = crop
    square = cv2.copyMakeBorder(square, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
    small = cv2.resize(square, (G, G), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).ravel()


def plate_glyphs(img: np.ndarray):
    """Full front end: image -> (normalised, ink mask, boxes, feature matrix)."""
    norm = normalize(img)
    ink = binarize(norm)
    boxes = segment(ink)
    feats = np.array([glyph(ink, b) for b in boxes], np.float32) if boxes else np.zeros((0, G * G), np.float32)
    return norm, ink, boxes, feats
