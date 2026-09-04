"""Indian licence-plate synthesis: renderer + real-world degradation model.

Two jobs:

1. Give the OCR engine a training/benchmark corpus with ground truth, so
   accuracy is a *measured* number rather than a claim.
2. Give the city simulator a real image per sighting, so the pipeline that runs
   in the demo is the same pipeline that gets benchmarked.

The degradation model is the interesting half — it reproduces the conditions the
problem statement calls out: lighting, weather, angle, motion blur, dirt and
physical damage.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- plate grammar ------------------------------------------------------
# Bharat series state codes, weighted towards West Bengal for a Kolkata demo,
# with the neighbouring states that actually show up on Kolkata's roads next.
STATE_CODES = [
    "WB", "WB", "WB", "WB", "WB", "WB", "WB", "WB",
    "OD", "OD", "BR", "BR", "JH", "JH", "AS", "SK", "TR", "UP",
    "DL", "MH", "KA", "TN", "AP", "TS", "GJ", "RJ", "MP", "PB", "HR", "CG",
]
SERIES_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # I and O are not issued in series

# The capture conditions are now camera scenarios: a mounting, an exposure and
# the weather, run through anpr/camera.py in physical order. The old flat list
# ("glare", "rain", "low_res") described filters; these describe sites.
from anpr import camera as _camera            # noqa: E402  (cycle-free: camera imports nothing here)

CONDITIONS = list(_camera.SCENARIOS)

# Grime and damage live on the plate, not in the camera, so they compose with
# any scenario: a filthy plate can also be a night shot in the rain.
SURFACE_FAULTS = ["clean", "dirty", "damaged"]

# Indian plates are supposed to use one prescribed typeface and in practice use
# whatever the shop had. Training across a spread of condensed and heavy faces is
# what stops the classifier from learning one font's quirks instead of the
# letters - it is the difference between reading a rendered plate and reading a
# photograph of a real one.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ARIALNB.TTF",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
    r"C:\Windows\Fonts\framd.ttf",
    r"C:\Windows\Fonts\bahnschrift.ttf",
    r"C:\Windows\Fonts\seguibl.ttf",
    r"C:\Windows\Fonts\tahomabd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _available_fonts() -> list[str]:
    found = [p for p in _FONT_CANDIDATES if Path(p).exists()]
    import matplotlib  # bundled with the dashboard deps; ships DejaVu everywhere
    mpl = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    # sans faces only: no registration plate anywhere in India is set in a serif
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSansMono-Bold.ttf"):
        if (mpl / name).exists():
            found.append(str(mpl / name))
    return found


FONTS = _available_fonts()
FONT_PATH = FONTS[0]
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, path: str | None = None) -> ImageFont.FreeTypeFont:
    key = (path or FONT_PATH, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(key[0], size)
    return _font_cache[key]


# --- plate strings ------------------------------------------------------
def random_plate(rng: random.Random | None = None) -> str:
    """A syntactically valid Bharat-series plate, e.g. ``KA05MJ1234``."""
    rng = rng or random
    state = rng.choice(STATE_CODES)
    district = f"{rng.randint(1, 68):02d}"
    series = "".join(rng.choice(SERIES_LETTERS) for _ in range(rng.choice([1, 2, 2, 2, 3])))
    number = f"{rng.randint(1, 9999):04d}"
    return f"{state}{district}{series}{number}"


def pretty(plate: str) -> str:
    """``KA05MJ1234`` -> ``KA 05 MJ 1234`` for display."""
    p = plate.replace(" ", "").upper()
    if len(p) < 8:
        return p
    head, tail = p[:2], p[2:]
    digits = ""
    while tail and tail[0].isdigit():
        digits, tail = digits + tail[0], tail[1:]
    return f"{head} {digits} {tail[:-4]} {tail[-4:]}".replace("  ", " ").strip()


# --- rendering ----------------------------------------------------------
@dataclass
class PlateImage:
    image: np.ndarray      # uint8 grayscale
    text: str              # ground truth, no spaces
    condition: str         # camera scenario
    two_row: bool = False
    fault: str = "clean"   # what is wrong with the plate itself
    detail: str = ""       # the rig and exposure, for explaining a failure


def split_two_row(text: str) -> tuple[str, str]:
    """Where a two-row plate breaks: state and district on top, series and
    number underneath, the way Indian motorcycle and truck plates are laid out."""
    p = text.replace(" ", "")
    i = 2
    while i < len(p) and p[i].isdigit():
        i += 1
    return p[:i], p[i:]


def render_plate(text: str, *, commercial: bool = False, width: int = 440,
                 rng: random.Random | None = None, font_path: str | None = None,
                 two_row: bool = False, ind_strip: bool | None = None) -> np.ndarray:
    """Render a clean, head-on plate as a BGR uint8 array."""
    rng = rng or random
    font_path = font_path or rng.choice(FONTS)
    if ind_strip is None:
        ind_strip = rng.random() < 0.45
    height = int(width * (0.46 if two_row else 0.22))
    # A real High Security plate is wider to carry the IND band rather than
    # squeezing the registration into less room, so the band must not cost the
    # characters their size.
    if ind_strip:
        width += max(14, int(width * 0.055)) + 8
    bg = (0, 200, 235) if commercial else (245, 245, 245)   # BGR: yellow / white
    fg = (10, 10, 10)
    img = Image.new("RGB", (width, height), tuple(reversed(bg)))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, width - 3, height - 3], outline=tuple(reversed(fg)), width=3)

    left = 6
    if ind_strip:
        # The blue IND band down the left edge of a High Security plate. It is
        # not part of the registration, so the reader has to ignore it.
        band = max(14, int(width * 0.055))
        draw.rectangle([4, 4, 4 + band, height - 5], fill=(20, 40, 150))
        f = font(max(8, int(height * (0.16 if two_row else 0.26))), font_path)
        draw.text((6, height // 2), "IND", font=f, fill=(255, 255, 255), anchor="lm")
        left = 4 + band + 4

    def fit(label: str, box_w: int, box_h: int):
        size = int(box_h * 0.95)
        while size > 9:
            f = font(size, font_path)
            bb = draw.textbbox((0, 0), label, font=f)
            if bb[2] - bb[0] <= box_w and bb[3] - bb[1] <= box_h:
                return f
            size -= 2
        return font(10, font_path)

    def centre(label: str, f, y0: int, y1: int):
        bb = draw.textbbox((0, 0), label, font=f)
        x = left + ((width - left - 8) - (bb[2] - bb[0])) // 2 - bb[0]
        y = y0 + ((y1 - y0) - (bb[3] - bb[1])) // 2 - bb[1]
        draw.text((x, y), label, font=f, fill=tuple(reversed(fg)))

    if two_row:
        top, bottom = split_two_row(text)
        band_h = int(height * 0.38)
        head = (top[:2] + " " + top[2:]).strip()
        tail = (bottom[:-4] + " " + bottom[-4:]).strip() if len(bottom) > 4 else bottom
        centre(head, fit(head, width - left - 12, band_h), int(height * 0.06), int(height * 0.46))
        centre(tail, fit(tail, width - left - 12, band_h), int(height * 0.52), int(height * 0.94))
    else:
        label = pretty(text)
        centre(label, fit(label, width - left - 14, int(height * 0.66)), 0, height)

    # mounting screws - small dark discs that segmentation must learn to ignore
    cy = height // 2
    for cx in (left + int(width * 0.02), int(width * 0.95)):
        if 6 < cx < width - 6:
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(90, 90, 90))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --- degradations -------------------------------------------------------
def _noise(img: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    g = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)


def _warp_angled(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    h, w = img.shape[:2]
    j = severity * 0.22
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * rng.uniform(0, j), h * rng.uniform(0, j)],
        [w * (1 - rng.uniform(0, j)), h * rng.uniform(0, j)],
        [w * (1 - rng.uniform(0, j)), h * (1 - rng.uniform(0, j))],
        [w * rng.uniform(0, j), h * (1 - rng.uniform(0, j))],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    angle = rng.uniform(-9, 9) * severity
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(out, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _motion_blur(img: np.ndarray, rng: random.Random, severity: float) -> np.ndarray:
    k = max(3, int(3 + 9 * severity) | 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = rng.uniform(-18, 18)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0), (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _blob_mask(shape, rng: random.Random, scale: int = 24) -> np.ndarray:
    """Smooth random field in [0,1] — used for dirt, glare and uneven lighting."""
    h, w = shape[:2]
    small = np.random.default_rng(rng.randint(0, 2**31)).random((max(2, h // scale), max(2, w // scale)))
    field = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(field, (0, 0), max(1.0, min(h, w) / 12))


def degrade(img: np.ndarray, condition: str, rng: random.Random,
            severity: float | None = None) -> np.ndarray:
    """Apply a named real-world condition. Input/output are BGR uint8."""
    sev = rng.uniform(0.35, 0.90) if severity is None else severity
    out = img.astype(np.float32)

    if condition == "mixed":
        for c in rng.sample(["night", "rain", "motion_blur", "angled", "dirty", "low_res"], k=2):
            img = degrade(img, c, rng, sev * 0.75)
        return img

    if condition == "night":
        field = _blob_mask(img.shape, rng)
        out *= (0.30 + 0.35 * sev * field)[..., None] + (1 - sev) * 0.55
        out = np.clip(out, 0, 255)
        out = _noise(out.astype(np.uint8), 9 * sev, rng).astype(np.float32)

    elif condition == "glare":
        h, w = img.shape[:2]
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.28) ** 2)))
        out = np.clip(out + (215 * sev * r)[..., None], 0, 255)

    elif condition == "rain":
        h, w = img.shape[:2]
        veil = np.zeros((h, w), np.float32)
        for _ in range(int(60 * sev)):
            x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
            ln = rng.randint(4, 14)
            cv2.line(veil, (x, y), (x + rng.randint(-2, 2), min(h - 1, y + ln)), 1.0, 1)
        veil = cv2.GaussianBlur(veil, (0, 0), 1.2)
        out = np.clip(out * (1 - 0.35 * sev) + 70 * sev + 110 * veil[..., None], 0, 255)
        out = cv2.GaussianBlur(out, (0, 0), 0.9 + 1.4 * sev)

    elif condition == "motion_blur":
        out = _motion_blur(out.astype(np.uint8), rng, sev).astype(np.float32)

    elif condition == "angled":
        out = _warp_angled(out.astype(np.uint8), rng, sev).astype(np.float32)

    elif condition == "dirty":
        mask = _blob_mask(img.shape, rng, scale=14)
        mask = (mask - mask.min()) / max(float(np.ptp(mask)), 1e-6)
        # Two effects, both capped: a film of dust that flattens contrast
        # across the whole plate, and splashes that darken parts of it. Neither
        # is opaque - grime dims a character, it does not delete it, and a
        # character no light escapes cannot be recovered by any reader.
        mud = cv2.GaussianBlur(np.clip((mask - (1 - 0.85 * sev)) * 3, 0, 0.72), (0, 0), 2.0)
        tone = np.array([58, 78, 96], np.float32)          # BGR mud
        out = out * (1 - mud[..., None]) + tone * mud[..., None]
        film = 0.30 * sev
        out = out * (1 - film) + film * float(out.mean())

    elif condition == "damaged":
        h, w = img.shape[:2]
        canvas = out.astype(np.uint8).copy()
        for _ in range(int(1 + 3 * sev)):
            p1 = (rng.randint(0, w), rng.randint(0, h))
            p2 = (p1[0] + rng.randint(-70, 70), p1[1] + rng.randint(-24, 24))
            cv2.line(canvas, p1, p2, (rng.randint(70, 190),) * 3, rng.randint(2, 5))
        for _ in range(int(1 + 2 * sev)):
            cv2.ellipse(canvas, (rng.randint(0, w), rng.randint(0, h)),
                        (rng.randint(6, 22), rng.randint(4, 12)), rng.randint(0, 180),
                        0, 360, (rng.randint(120, 210),) * 3, -1)
        out = 0.25 * out + 0.75 * canvas.astype(np.float32)   # scratches, not erasure

    elif condition == "low_res":
        h, w = img.shape[:2]
        f = 1.0 - 0.72 * sev
        small = cv2.resize(out.astype(np.uint8), (max(24, int(w * f)), max(8, int(h * f))))
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def surface_fault(img: np.ndarray, fault: str, rng: random.Random,
                  severity: float | None = None) -> np.ndarray:
    """Apply what has happened to the plate itself, before any camera sees it."""
    if fault in ("dirty", "damaged"):
        return degrade(img, fault, rng, severity)
    return img


def capture(text: str | None = None, condition: str | None = None,
            rng: random.Random | None = None, severity: float | None = None,
            fault: str | None = None, two_row: bool | None = None) -> PlateImage:
    """One camera capture of one plate, end to end.

    Renders the plate at high resolution, ages the surface, then shoots it
    through the camera model for the named scenario. What comes back is the
    region of interest an ANPR node would hand the reader: small, foreshortened,
    compressed and digitally zoomed.
    """
    rng = rng or random
    text = text or random_plate(rng)
    condition = condition or rng.choice(CONDITIONS)
    if fault is None:
        fault = rng.choices(SURFACE_FAULTS, [0.72, 0.20, 0.08])[0]
    if two_row is None:
        two_row = rng.random() < 0.18          # bikes, autos, trucks

    plate = render_plate(text, commercial=rng.random() < 0.18,
                         width=rng.choice([640, 760, 900]), rng=rng, two_row=two_row)
    plate = surface_fault(plate, fault, rng, severity)
    cap = _camera.scenario(condition, rng)
    shot = _camera.shoot(plate, cap, rng)
    gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    return PlateImage(image=gray, text=text, condition=condition, two_row=two_row,
                      fault=fault, detail=cap.describe())


# --- whole-frame scenes -------------------------------------------------
@dataclass
class Scene:
    image: np.ndarray          # BGR frame containing one plate
    text: str                  # ground truth
    box: tuple                 # (x, y, w, h) of the plate in the frame
    condition: str


def scene(text: str | None = None, condition: str | None = None,
          rng: random.Random | None = None, size: tuple[int, int] = (960, 640)) -> Scene:
    """Composite a plate into a wider vehicle scene.

    The tight-crop benchmark measures the reader. This measures the whole
    upload path - localisation included - by putting the plate somewhere in a
    frame that also contains bumpers, shadows, grilles and other edge-dense
    clutter for the localiser to reject. It is a composite, not a photograph:
    it tests whether the engine can find a plate among distractors, not whether
    it survives a real camera's optics.
    """
    rng = rng or random
    text = text or random_plate(rng)
    condition = condition or rng.choice(CONDITIONS)
    W, H = size
    frame = np.zeros((H, W, 3), np.uint8)

    # ground and body panels, lit top to bottom
    base = rng.randint(45, 150)
    for y in range(H):
        frame[y, :] = np.clip(base + 55 * (y / H) + rng.uniform(-3, 3), 0, 255)
    body = np.array([rng.randint(20, 210) for _ in range(3)], np.uint8)
    cv2.rectangle(frame, (0, int(H * 0.12)), (W, int(H * 0.78)), body.tolist(), -1)
    cv2.rectangle(frame, (0, int(H * 0.70)), (W, int(H * 0.80)),
                  (body * 0.75).astype(np.uint8).tolist(), -1)

    # grille slats and lamps: the edge-dense clutter a localiser must reject
    gx, gy = rng.randint(40, W // 3), rng.randint(int(H * 0.18), int(H * 0.36))
    for i in range(rng.randint(4, 9)):
        cv2.rectangle(frame, (gx, gy + i * 11), (gx + rng.randint(160, 320), gy + i * 11 + 5),
                      (int(body[0]) // 2, int(body[1]) // 2, int(body[2]) // 2), -1)
    for _ in range(2):
        cv2.ellipse(frame, (rng.randint(0, W), rng.randint(int(H * 0.15), int(H * 0.4))),
                    (rng.randint(40, 90), rng.randint(20, 40)), 0, 0, 360,
                    (rng.randint(150, 245),) * 3, -1)
    for _ in range(rng.randint(1, 3)):      # stickers / badges with text-like edges
        x, y = rng.randint(0, W - 120), rng.randint(int(H * 0.15), int(H * 0.7))
        cv2.putText(frame, rng.choice(["TATA", "4x4", "TURBO", "BS-VI", "DIESEL"]),
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.5, 1.1),
                    (rng.randint(120, 255),) * 3, 2)

    # the plate itself, degraded then perspective-warped into the scene
    two_row = rng.random() < 0.18
    pw = rng.randint(int(W * 0.16), int(W * 0.34))
    plate = render_plate(text, commercial=rng.random() < 0.18, width=pw * 3, rng=rng,
                         two_row=two_row)
    plate = surface_fault(plate, rng.choices(SURFACE_FAULTS, [0.72, 0.20, 0.08])[0], rng)
    cam = _camera.scenario(condition, rng)
    plate = _camera.shoot(plate, cam, rng, canvas_scale=1.05)
    plate = cv2.resize(plate, (pw, max(8, int(pw * plate.shape[0] / plate.shape[1]))))
    ph = plate.shape[0]
    px = rng.randint(10, max(11, W - pw - 10))
    py = rng.randint(int(H * 0.42), max(int(H * 0.42) + 1, int(H * 0.72) - ph))

    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    j = 0.02      # the camera model already applied the real perspective
    dst = np.float32([[pw * rng.uniform(0, j), ph * rng.uniform(0, j)],
                      [pw * (1 - rng.uniform(0, j)), ph * rng.uniform(0, j)],
                      [pw * (1 - rng.uniform(0, j)), ph * (1 - rng.uniform(0, j))],
                      [pw * rng.uniform(0, j), ph * (1 - rng.uniform(0, j))]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(plate, m, (pw, ph), borderMode=cv2.BORDER_REPLICATE)
    frame[py:py + ph, px:px + pw] = warped

    # a soft shadow under the plate and overall camera noise
    cv2.rectangle(frame, (px - 4, py + ph), (px + pw + 4, min(H - 1, py + ph + 8)),
                  (30, 30, 30), -1)
    frame = cv2.GaussianBlur(frame, (0, 0), rng.uniform(0.3, 1.0))
    frame = _noise(frame, rng.uniform(2, 8), rng)
    return Scene(image=frame, text=text, box=(px, py, pw, ph), condition=condition)
