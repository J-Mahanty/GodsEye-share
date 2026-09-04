"""A physical model of an Indian traffic camera looking at a number plate.

Why this exists
---------------
The first version of this project rendered a plate head-on at 440 px and layered
filters on top — a translucent grey wash called "fog", a bright blob called
"glare". That is not what a gantry camera sees, and an OCR engine tuned against
it learns the wrong invariances.

A real enforcement camera on an Indian arterial sits five to eight metres up on
a pole or gantry, angled down at the carriageway. The consequences dominate
everything else:

* the plate is seen from **above and off to one side**, so it is foreshortened
  vertically and keystoned horizontally — it is never a rectangle;
* at 15-30 m the plate is **40-120 px wide in the sensor**, not 440;
* the camera **JPEG-compresses** that small plate at a modest bitrate;
* an operator (or an ROI pipeline) then **crops and digitally zooms** it, which
  upscales the compression blocks along with the characters, and usually saves
  it again.

So the order of operations is the whole point. Blur belongs before sensor noise,
because a lens blurs the scene and the sensor then adds noise to the blurred
image — do it the other way and you get smooth, blurred noise that no denoiser
ever sees in the field. Compression belongs after noise. Digital zoom belongs
after compression, because you are upscaling artefacts, not detail. Each stage
below is a separate function so the order stays visible and testable.

Why not Albumentations
----------------------
Its `RandomRain`, `RandomFog`, `ISONoise` and `ImageCompression` cover roughly
this vocabulary and are well tested, and if you want a broader corpus they slot
in cleanly at stage 3-7 (see `AUGMENTATION.md`). What they cannot express is the
*pipeline*: an ordered chain where geometry precedes atmosphere, atmosphere
precedes optics, and a digital zoom happens after compression. They also know
nothing about retroreflective plate sheeting under an IR flash, which is the
single most distinctive thing about a night ANPR capture. That specific physics
plus the ordering is what makes this corpus worth training on, so the core is
written here in OpenCV and the library stays optional.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np


# --- how a camera is mounted and what it is looking at ------------------
@dataclass
class CameraRig:
    """Where the camera is, and what it can see from there."""
    height_m: float = 6.0          # gantry or pole height above the carriageway
    distance_m: float = 20.0       # along-road distance to the vehicle
    lateral_m: float = 2.5         # sideways offset from the vehicle's lane
    focal_px: float = 4200.0       # telephoto: these cameras are specced
                                   # so the plate clears ~100 px of width
    plate_width_m: float = 0.50    # a one-row Indian plate is ~500 mm wide

    @property
    def depression_deg(self) -> float:
        """How far down the camera is looking. 20-45 degrees in practice."""
        return math.degrees(math.atan2(self.height_m, max(self.distance_m, 1e-3)))

    @property
    def yaw_deg(self) -> float:
        """How far off-axis the plate is, left or right."""
        return math.degrees(math.atan2(self.lateral_m, max(self.distance_m, 1e-3)))

    @property
    def plate_px(self) -> float:
        """Apparent plate width in sensor pixels — the number that decides
        whether this capture is readable at all."""
        slant = math.sqrt(self.height_m ** 2 + self.distance_m ** 2 + self.lateral_m ** 2)
        return self.focal_px * self.plate_width_m / max(slant, 1e-3)


@dataclass
class Weather:
    fog: float = 0.0               # 0 clear, 1 dense
    rain: float = 0.0              # 0 dry, 1 downpour
    droplets: float = 0.0          # water on the lens housing


@dataclass
class Capture:
    """Everything about one exposure, so a failure can be explained."""
    rig: CameraRig
    weather: Weather
    exposure_s: float = 1 / 250    # shutter speed
    iso: int = 400                 # sensor gain
    speed_kmph: float = 40.0       # vehicle speed, sets motion blur length
    ir_flash: bool = False         # night: IR illuminator instead of visible light
    ambient: float = 1.0           # 1 daylight, <0.2 night
    headlight: float = 0.0         # oncoming specular glare, 0-1
    jpeg_quality: int = 60         # camera-side encoder
    digital_zoom: float = 1.0      # operator crop-and-upscale factor
    defocus_px: float = 0.6
    rolling_shutter: bool = True
    # How much of the light shortfall the camera's auto-exposure recovers. Real
    # ANPR cameras meter continuously and mostly succeed; 1.0 is a camera that
    # nails it, 0.0 one with the exposure nailed open in manual. The interesting
    # failures live in between, where AE runs out of range and hands you a dark
    # frame it then tries to gamma-lift.
    auto_exposure: float = 0.9
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        r = self.rig
        return (f"{r.height_m:.0f} m pole at {r.distance_m:.0f} m "
                f"({r.depression_deg:.0f}° down, {r.yaw_deg:.0f}° off-axis), "
                f"plate {r.plate_px:.0f} px, {self.speed_kmph:.0f} km/h, "
                f"1/{1/max(self.exposure_s,1e-6):.0f}s ISO {self.iso}, "
                f"JPEG q{self.jpeg_quality}, zoom x{self.digital_zoom:.1f}"
                + (", IR flash" if self.ir_flash else ""))


# ======================================================================
# Stage 1 — the plate surface itself
# ======================================================================
def retroreflect(plate_bgr: np.ndarray, cap: Capture) -> np.ndarray:
    """Model the plate's retroreflective sheeting.

    An Indian HSRP is retroreflective: it throws light straight back at the
    source. Under a night IR flash the sheet returns far more light than the
    car around it, so the plate face blows out towards white while the
    characters stay black — high contrast, but with blooming that closes up the
    counters of 8, B, 0 and 6. In daylight the effect is mild.
    """
    out = plate_bgr.astype(np.float32)
    if cap.ir_flash:
        # IR: monochrome response, plate face driven bright, blooming into the
        # characters. This is why a night capture often reads *better* than dusk.
        grey = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gain = 1.10 + 0.25 * (1.0 - cap.ambient)
        grey = np.clip(grey * gain + 25.0, 0, 255)
        bloom = cv2.GaussianBlur(grey, (0, 0), 2.2)
        grey = np.clip(0.78 * grey + 0.30 * bloom, 0, 255)
        out = cv2.cvtColor(grey.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    else:
        sheen = 1.0 + 0.25 * (1.0 - cap.ambient)
        out = np.clip(out * sheen, 0, 255)
    return out.astype(np.uint8)


# ======================================================================
# Stage 2 — geometry: where the camera is, not where we wish it were
# ======================================================================
def project(plate_bgr: np.ndarray, cap: Capture, rng: random.Random,
            canvas_scale: float = 1.12) -> np.ndarray:
    """Warp a head-on plate into the view from a raised, off-axis camera.

    Builds the homography from the rig geometry rather than from arbitrary
    corner jitter, so the distortion is the one the mounting actually produces:
    the top edge shorter than the bottom (looking down), one side shorter than
    the other (looking across), and a slight roll from a pole that is never
    quite plumb.
    """
    h, w = plate_bgr.shape[:2]
    pitch = math.radians(cap.rig.depression_deg)
    yaw = math.radians(cap.rig.yaw_deg) * (1 if rng.random() < 0.5 else -1)
    roll = math.radians(rng.uniform(-4.0, 4.0))

    # Plate corners in metres, centred on the plate, then rotated into the
    # camera frame and projected. This is a pinhole camera, nothing fancier.
    pw = cap.rig.plate_width_m
    ph = pw * (h / w)
    obj = np.float32([[-pw / 2, -ph / 2, 0], [pw / 2, -ph / 2, 0],
                      [pw / 2, ph / 2, 0], [-pw / 2, ph / 2, 0]])

    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    Rx = np.float32([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.float32([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.float32([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    depth = math.sqrt(cap.rig.height_m ** 2 + cap.rig.distance_m ** 2)
    cam = (R @ obj.T).T + np.float32([0, 0, depth])
    f = cap.rig.focal_px
    proj = np.stack([f * cam[:, 0] / cam[:, 2], f * cam[:, 1] / cam[:, 2]], axis=1)

    # Scale the projection to the requested apparent plate width, then centre it
    # on a canvas with room for the vehicle body around it.
    span = float(np.linalg.norm(proj[1] - proj[0])) or 1.0
    proj *= cap.rig.plate_px / span
    out_w = int(max(cap.rig.plate_px * canvas_scale, 48))
    out_h = int(max(out_w * 0.42, 24))
    proj += np.float32([out_w / 2, out_h / 2]) - proj.mean(axis=0)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, proj.astype(np.float32))
    return cv2.warpPerspective(plate_bgr, m, (out_w, out_h),
                               flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


# ======================================================================
# Stage 3 — atmosphere between the plate and the lens
# ======================================================================
def fog(img: np.ndarray, cap: Capture, rng: random.Random) -> np.ndarray:
    """Koschmieder scattering: contrast falls exponentially with distance.

    Not a grey overlay — the airlight term depends on the extinction
    coefficient and the path length, so a plate 30 m away in the same fog is
    far worse than one at 12 m, which is the behaviour that matters when the
    engine is asked why it read the near lane and not the far one.
    """
    if cap.weather.fog <= 0.01:
        return img
    beta = 0.02 + 0.16 * cap.weather.fog              # extinction per metre
    d = math.sqrt(cap.rig.distance_m ** 2 + cap.rig.height_m ** 2)
    t = math.exp(-beta * d)                            # transmission
    airlight = 235.0 * (0.55 + 0.45 * cap.ambient)
    out = img.astype(np.float32) * t + airlight * (1 - t)
    # fog is not perfectly uniform - it drifts in patches
    h, w = img.shape[:2]
    field = np.random.default_rng(rng.randint(0, 2**31)).random((max(2, h // 12), max(2, w // 12)))
    field = cv2.GaussianBlur(cv2.resize(field.astype(np.float32), (w, h)), (0, 0), 6)
    out += ((field - field.mean()) * 26.0 * cap.weather.fog)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def rain(img: np.ndarray, cap: Capture, rng: random.Random) -> np.ndarray:
    """Rain as the sensor records it: streaks, not dots.

    A raindrop crosses the frame during the exposure, so it images as a
    motion-blurred bright streak whose length is set by shutter speed and whose
    angle is set by wind. Add the veiling glow that scattered light puts over
    the whole frame, and the wet-road specular sheen that lifts the black parts
    of the plate.
    """
    if cap.weather.rain <= 0.01:
        return img
    h, w = img.shape[:2]
    intensity = cap.weather.rain
    # streak length: drops fall ~9 m/s, so exposure time sets pixels travelled
    px_per_m = cap.rig.focal_px / max(cap.rig.distance_m, 1e-3)
    length = int(np.clip(9.0 * cap.exposure_s * px_per_m, 3, max(4, h // 2)))
    angle = rng.uniform(-25, 25)                       # wind shear
    layer = np.zeros((h, w), np.float32)
    drops = int(intensity * (h * w) / 900)
    for _ in range(drops):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        dx = int(length * math.tan(math.radians(angle)))
        cv2.line(layer, (x, y), (x + dx, min(h - 1, y + length)),
                 rng.uniform(0.45, 1.0), 1, cv2.LINE_AA)
    # the streak is itself blurred along its direction of travel
    k = max(3, length | 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[:, k // 2] = 1.0 / k
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0), (k, k))
    layer = cv2.filter2D(layer, -1, kernel / max(kernel.sum(), 1e-6))

    out = img.astype(np.float32)
    out += (layer[..., None] * 150.0 * (0.4 + 0.6 * cap.ambient))
    veil = 55.0 * intensity * (0.35 + 0.65 * cap.ambient)
    out = out * (1.0 - 0.22 * intensity) + veil
    return np.clip(out, 0, 255).astype(np.uint8)


def droplets(img: np.ndarray, cap: Capture, rng: random.Random) -> np.ndarray:
    """Water sitting on the lens housing, which *refracts* rather than dims.

    Each droplet acts as a small lens: the image under it is displaced and
    inverted-ish and blurred. Implemented as a per-pixel remap so characters
    genuinely smear across a droplet edge instead of being covered by a circle.
    """
    if cap.weather.droplets <= 0.01:
        return img
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32),
                               np.arange(h, dtype=np.float32))
    blurred = cv2.GaussianBlur(img, (0, 0), 1.8)
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(2 + 10 * cap.weather.droplets)):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r = rng.uniform(0.05, 0.16) * min(h, w) * (0.6 + cap.weather.droplets)
        yy, xx = np.mgrid[0:h, 0:w]
        dx, dy = xx - cx, yy - cy
        d = np.sqrt(dx * dx + dy * dy) / max(r, 1e-3)
        inside = d < 1.0
        if not inside.any():
            continue
        # refraction: sample from further out towards the droplet centre
        k = np.zeros_like(d)
        k[inside] = (1.0 - d[inside] ** 2) * 0.65
        map_x -= (dx * k).astype(np.float32)
        map_y -= (dy * k).astype(np.float32)
        mask = np.maximum(mask, inside.astype(np.float32))
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask3 = cv2.GaussianBlur(mask, (0, 0), 1.5)[..., None]
    return np.clip(warped * (1 - mask3) + blurred * mask3, 0, 255).astype(np.uint8)


# ======================================================================
# Stage 4 — illumination
# ======================================================================
def headlight_glare(img: np.ndarray, cap: Capture, rng: random.Random) -> np.ndarray:
    """Specular glare from an oncoming headlight: core, halo, and veiling flare.

    Three components, because that is what a lens actually produces — a blown
    core, a wide diffraction halo around it, and a low-frequency wash across the
    whole frame that eats contrast far from the source.
    """
    if cap.headlight <= 0.01:
        return img
    h, w = img.shape[:2]
    strength = cap.headlight
    cx = rng.uniform(-0.15, 1.15) * w
    cy = rng.uniform(-0.2, 1.0) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    core = np.exp(-r2 / (2 * (0.07 * w) ** 2)) * 255.0
    halo = np.exp(-r2 / (2 * (0.30 * w) ** 2)) * 165.0
    veil = np.full((h, w), 40.0, np.float32)
    glare = strength * (core + halo + veil * strength)

    out = img.astype(np.float32) + glare[..., None]
    # streaked flare along the sensor rows, as cheap lenses produce
    if rng.random() < 0.5:
        streak = cv2.GaussianBlur(core * strength, (int(w * 0.4) | 1, 1), 0)
        out += streak[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# A correctly metered daylight frame: 1/250 s at ISO 400 on a scene of relative
# luminance 1.0. Everything else is measured against it in stops.
_REFERENCE_EV = (1 / 250) * 400.0


def expose(img: np.ndarray, cap: Capture) -> np.ndarray:
    """Exposure as a light budget, not a brightness knob.

    Scene luminance times shutter time times sensor gain, divided by the budget a
    correctly metered daylight frame gets. ISO *compensates* for missing light,
    it does not add scene brightness - getting that backwards is what made a
    midday frame at ISO 100 come out nearly black. Under an IR flash the
    illumination comes from the lamp on the housing, so ambient barely matters.
    """
    light = 0.80 if cap.ir_flash else max(cap.ambient, 0.01)
    ev = light * (cap.exposure_s * cap.iso) / _REFERENCE_EV
    # Auto-exposure pulls the frame back towards correct by its own gain: at
    # auto_exposure = 1 the camera compensates fully, at 0 not at all.
    ev = float(ev) ** (1.0 - float(np.clip(cap.auto_exposure, 0.0, 1.0)))
    out = img.astype(np.float32) * float(np.clip(ev, 0.10, 1.8))
    if ev < 0.7:                                    # the camera lifts the shadows
        out = 255.0 * np.power(np.clip(out / 255.0, 0, 1), 0.78)
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================================================================
# Stage 5 — optics and motion
# ======================================================================
def motion_blur(img: np.ndarray, cap: Capture) -> np.ndarray:
    """Directional blur whose length comes from speed and shutter, not taste.

    A vehicle at 60 km/h crossing a 1/125 s exposure moves 13 cm; at 20 m from a
    2200 px lens that is about 15 px of smear. This is the single most common
    reason a daylight capture is unreadable.
    """
    px_per_m = cap.rig.focal_px / max(cap.rig.distance_m, 1e-3)
    metres = cap.speed_kmph / 3.6 * cap.exposure_s
    # Only motion *across* the sensor smears the image. A vehicle driving at the
    # camera is moving almost along the optical axis: it looms rather than
    # smears, and the transverse component goes with the sine of the off-axis
    # angle, not the cosine. Using the cosine put a quarter-plate of blur on
    # every head-on capture, which is why the whole contact sheet was a streak.
    transverse = abs(math.sin(math.radians(cap.rig.yaw_deg)))
    transverse = max(transverse, 0.12)          # lane change, steering, vibration
    length = metres * px_per_m * transverse
    k = int(np.clip(length, 1, 60))
    if k < 2:
        return img
    k |= 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = cap.rig.depression_deg * 0.25            # slight downward component
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0), (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def defocus(img: np.ndarray, cap: Capture) -> np.ndarray:
    if cap.defocus_px <= 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), cap.defocus_px)


def rolling_shutter(img: np.ndarray, cap: Capture) -> np.ndarray:
    """CMOS rows are exposed in sequence, so a fast vehicle shears vertically."""
    if not cap.rolling_shutter or cap.speed_kmph < 25:
        return img
    h, w = img.shape[:2]
    px_per_m = cap.rig.focal_px / max(cap.rig.distance_m, 1e-3)
    shear = cap.speed_kmph / 3.6 * (cap.exposure_s * 0.6) * px_per_m
    if shear < 0.7:
        return img
    m = np.float32([[1, shear / max(h, 1), -shear / 2], [0, 1, 0]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ======================================================================
# Stage 6 — the sensor
# ======================================================================
def sensor_noise(img: np.ndarray, cap: Capture, rng: random.Random) -> np.ndarray:
    """Poisson shot noise plus Gaussian read noise, scaled by ISO.

    Real sensor noise is signal-dependent: bright areas are noisier in absolute
    terms, dark areas are dominated by read noise. Adding flat Gaussian noise
    instead teaches a model an invariance that does not exist.
    """
    gen = np.random.default_rng(rng.randint(0, 2**31))
    # Full-well electrons at base ISO. Shot noise is sqrt(electrons), so this
    # sets the signal-to-noise ratio: ~45 dB at ISO 100 falling to ~26 dB at
    # ISO 3200, which is roughly what a 1/2.8" surveillance sensor delivers.
    well = 9000.0 * (100.0 / max(cap.iso, 50))
    x = np.clip(img.astype(np.float32), 0, 255) / 255.0
    electrons = gen.poisson(np.clip(x * well, 0, None))
    shot = electrons / well * 255.0
    read = gen.normal(0.0, 1.1 + 0.0022 * cap.iso, x.shape)
    out = shot + read
    if cap.iso >= 800:                              # low-frequency chroma blotches
        h, w = img.shape[:2]
        amp = 1.2 + 0.0016 * cap.iso
        blotch = gen.normal(0, amp, (max(2, h // 6), max(2, w // 6), 3))
        out += cv2.resize(blotch.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================================================================
# Stage 7-8 — encoding, then the operator's crop and digital zoom
# ======================================================================
def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def digital_zoom(img: np.ndarray, cap: Capture) -> np.ndarray:
    """Crop-and-upscale, the way an operator or an ROI stage delivers a plate.

    This is the stage that fools people: the output is a large, apparently
    detailed image, but every pixel of that detail was interpolated from a
    40-120 px original that had already been through a JPEG encoder. The blocks
    get magnified with the characters.
    """
    if cap.digital_zoom <= 1.01:
        return img
    h, w = img.shape[:2]
    out = cv2.resize(img, (int(w * cap.digital_zoom), int(h * cap.digital_zoom)),
                     interpolation=cv2.INTER_CUBIC)
    return jpeg(out, max(35, cap.jpeg_quality - 10))   # saved again after zooming


# ======================================================================
# The pipeline
# ======================================================================
def shoot(plate_bgr: np.ndarray, cap: Capture, rng: random.Random,
          canvas_scale: float = 1.12) -> np.ndarray:
    """Run one plate through the whole camera, in physical order.

    The default canvas is a little wider than the plate: an ANPR node delivers a
    region of interest, not a whole carriageway. Pass a larger canvas_scale to
    produce a scene for testing the localiser instead of the reader.
    """
    img = retroreflect(plate_bgr, cap)          # 1 surface
    img = project(img, cap, rng, canvas_scale)  # 2 geometry
    img = fog(img, cap, rng)                    # 3 atmosphere
    img = rain(img, cap, rng)
    img = droplets(img, cap, rng)
    img = headlight_glare(img, cap, rng)        # 4 illumination
    img = expose(img, cap)
    img = motion_blur(img, cap)                 # 5 optics and motion
    img = defocus(img, cap)
    img = rolling_shutter(img, cap)
    img = sensor_noise(img, cap, rng)           # 6 sensor
    img = jpeg(img, cap.jpeg_quality)           # 7 encoder
    img = digital_zoom(img, cap)                # 8 operator crop and zoom
    return img


# --- named scenarios ----------------------------------------------------
def rig_for(rng: random.Random, far: bool = False) -> CameraRig:
    """A plausible Indian gantry or pole mounting."""
    height = rng.uniform(5.0, 8.5)
    distance = rng.uniform(22.0, 38.0) if far else rng.uniform(11.0, 24.0)
    # Focal lengths chosen so the near lane lands at 100-220 px of plate and the
    # far lane at 45-90 px, which is the spread a real enforcement site produces.
    return CameraRig(height_m=height, distance_m=distance,
                     lateral_m=rng.uniform(-6.0, 6.0),
                     focal_px=rng.uniform(3600.0, 7200.0))


SCENARIOS: dict[str, str] = {
    "daylight": "midday, well-lit, moderate speed",
    "night_ir": "no ambient light, IR illuminator, retroreflective plate",
    "night_glare": "night with an oncoming headlight in frame",
    "monsoon": "heavy rain, streaks, veiling and a wet lens",
    "fog": "winter morning fog on the bypass",
    "high_speed": "expressway speed against a slow shutter",
    "far_lane": "the far carriageway - a small plate and a hard angle",
    "dusk_highiso": "the worst hour: too dark for the shutter, no IR yet",
    "cheap_cam": "an old low-bitrate camera, heavy compression, digital zoom",
    "storm": "everything at once - the case that should be refused",
}


def scenario(name: str, rng: random.Random) -> Capture:
    """Build a Capture for one named real-world scenario."""
    rig = rig_for(rng, far=(name == "far_lane"))
    cap = Capture(rig=rig, weather=Weather(), speed_kmph=rng.uniform(25, 65))

    if name == "daylight":
        cap.exposure_s = rng.choice([1 / 2000, 1 / 1000, 1 / 500])
        cap.iso, cap.ambient, cap.jpeg_quality = rng.choice([100, 200]), 1.0, rng.randint(65, 85)
        cap.digital_zoom = rng.uniform(1.6, 3.0)
    elif name == "night_ir":
        cap.ir_flash, cap.ambient = True, rng.uniform(0.03, 0.12)
        cap.exposure_s, cap.iso = rng.choice([1 / 500, 1 / 350]), rng.choice([400, 800])
        cap.jpeg_quality, cap.digital_zoom = rng.randint(55, 75), rng.uniform(2.0, 3.5)
    elif name == "night_glare":
        cap.ir_flash = rng.random() < 0.5
        cap.ambient, cap.headlight = rng.uniform(0.05, 0.2), rng.uniform(0.55, 1.0)
        cap.auto_exposure = rng.uniform(0.35, 0.6)   # glare fools the meter
        cap.exposure_s, cap.iso = rng.choice([1 / 350, 1 / 250]), rng.choice([800, 1600])
        cap.digital_zoom = rng.uniform(2.0, 3.5)
    elif name == "monsoon":
        cap.weather = Weather(rain=rng.uniform(0.55, 1.0), fog=rng.uniform(0.05, 0.25),
                              droplets=rng.uniform(0.3, 0.9))
        cap.ambient, cap.exposure_s = rng.uniform(0.35, 0.7), rng.choice([1 / 500, 1 / 250])
        cap.iso, cap.digital_zoom = rng.choice([400, 800]), rng.uniform(1.8, 3.0)
    elif name == "fog":
        cap.weather = Weather(fog=rng.uniform(0.5, 0.95))
        cap.ambient, cap.iso = rng.uniform(0.5, 0.9), 400
        cap.digital_zoom = rng.uniform(1.8, 3.2)
    elif name == "high_speed":
        cap.speed_kmph = rng.uniform(70, 110)
        # a site whose shutter is too slow for the speed limit it polices
        cap.exposure_s = rng.choice([1 / 250, 1 / 125, 1 / 90])
        cap.iso, cap.digital_zoom = 400, rng.uniform(1.8, 3.0)
    elif name == "far_lane":
        cap.rig.focal_px = rng.uniform(2600.0, 4200.0)
        cap.exposure_s, cap.iso = 1 / 250, 400
        cap.digital_zoom = rng.uniform(3.0, 5.0)       # heavy zoom on a tiny plate
        cap.jpeg_quality = rng.randint(45, 65)
    elif name == "dusk_highiso":
        cap.ambient, cap.iso = rng.uniform(0.15, 0.35), rng.choice([1600, 3200])
        cap.auto_exposure = rng.uniform(0.45, 0.7)   # out of range at dusk
        cap.exposure_s, cap.defocus_px = rng.choice([1 / 125, 1 / 60]), rng.uniform(0.8, 1.6)
        cap.digital_zoom = rng.uniform(2.0, 3.5)
    elif name == "cheap_cam":
        cap.rig.focal_px = rng.uniform(2200.0, 3400.0)
        cap.jpeg_quality, cap.iso = rng.randint(22, 38), 800
        cap.defocus_px, cap.digital_zoom = rng.uniform(1.0, 2.0), rng.uniform(3.0, 5.0)
    elif name == "storm":
        cap.weather = Weather(rain=rng.uniform(0.6, 1.0), fog=rng.uniform(0.3, 0.7),
                              droplets=rng.uniform(0.4, 1.0))
        cap.ambient, cap.headlight = rng.uniform(0.05, 0.2), rng.uniform(0.4, 0.9)
        cap.iso, cap.exposure_s = rng.choice([1600, 3200]), rng.choice([1 / 125, 1 / 60])
        cap.speed_kmph, cap.jpeg_quality = rng.uniform(50, 90), rng.randint(25, 45)
        cap.digital_zoom, cap.defocus_px = rng.uniform(3.0, 5.0), rng.uniform(0.8, 2.0)
        cap.auto_exposure = rng.uniform(0.3, 0.55)
    else:
        raise KeyError(f"unknown scenario {name!r}; expected one of {sorted(SCENARIOS)}")

    cap.notes.append(SCENARIOS[name])
    return cap
