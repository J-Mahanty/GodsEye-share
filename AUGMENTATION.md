# The camera model

How GodsEye generates the images it trains and benchmarks on, and why it is
built as a camera rather than a stack of filters.

## The thing that changes everything

An Indian enforcement camera is **five to eight metres up on a pole or gantry,
looking down at the carriageway**. Almost every property of the resulting image
follows from that one fact, and none of them are what an image-filter library
gives you:

| Consequence | What it does to the plate |
|---|---|
| Looking **down** at 15–30° | Vertical foreshortening — the plate is never a rectangle |
| Looking **across** at up to 25° | Horizontal keystone; the far end of the plate is smaller |
| A pole that is never plumb | A degree or two of roll on every frame |
| Distance of 12–38 m | The plate is **40–260 px wide in the sensor**, not 440 |
| A camera-side H.264/JPEG encoder | Blocking and ringing *at that small size* |
| An operator or ROI stage cropping and zooming | Those artefacts get **upscaled with the characters** |

The last two matter more than any weather effect. A plate that is 90 px wide,
JPEG-compressed at quality 45, then upscaled 4× arrives looking large and
detailed — but every pixel of that detail was interpolated from compression
blocks. Train on clean 440 px renders and a model never learns this; deploy it
and it fails on the majority of real captures.

## Order is the point

The stages run in the order the physics happens. Getting this wrong produces
images that look plausible and teach the wrong invariances.

```
1  surface        retroreflective sheeting · grime · physical damage
2  geometry       pole height and offset -> homography · apparent size
3  atmosphere     fog (Koschmieder) · rain streaks · lens droplets
4  illumination   headlight glare (core/halo/veil) · IR flash · exposure
5  optics         motion blur · defocus · rolling shutter
6  sensor         Poisson shot noise + read noise, scaled by ISO
7  encoder        JPEG at the camera's bitrate
8  operator       crop, digital zoom, second JPEG
```

Three orderings that are load-bearing:

* **Blur before noise.** A lens blurs the scene and the sensor then samples it.
  Noise added first and blurred afterwards is smooth and correlated — a
  denoiser will happily remove it, and no real sensor produces it.
* **Noise before compression.** JPEG has to spend its bits encoding the noise,
  which is why a noisy night frame compresses so much worse than a clean one.
* **Compression before digital zoom.** The zoom magnifies the artefacts. Zoom
  first and you get a clean upscale that flatters the reader.

## What each stage actually models

**Retroreflection.** Indian High Security plates are retroreflective: they throw
light back at its source. Under a night IR illuminator the plate face blows out
towards white while the characters stay black, with bloom closing the counters
of `8`, `B`, `0` and `6`. This is why a night IR capture often reads *better*
than a dusk one — a genuinely counter-intuitive result the benchmark reproduces.

**Fog** uses the Koschmieder model: contrast decays as `exp(-βd)` with an
airlight term, so a plate 30 m away in the same fog is far worse than one at
12 m. A flat grey overlay cannot express that distance dependence.

**Rain** is streaks, not dots. A drop falls ~9 m/s and crosses the frame during
the exposure, so it images as a motion-blurred line whose length comes from the
shutter speed and whose angle comes from wind. Plus the veiling glow of
scattered light, which is what actually destroys contrast.

**Droplets** on the lens housing *refract* rather than dim — implemented as a
per-pixel remap, so characters smear across a droplet edge instead of being
hidden behind a circle.

**Glare** is three components: a blown core, a wide diffraction halo, and a
low-frequency veil across the whole frame. Only the third explains why contrast
dies far from the light source.

**Motion blur** length comes from `speed × exposure × pixels-per-metre`, and only
the component **across** the sensor smears — a vehicle driving at the camera
looms rather than smears. Getting this backwards (using the cosine of the
off-axis angle instead of the sine) put a quarter-plate of blur on every
head-on capture; the contact sheet made it obvious immediately.

**Sensor noise** is Poisson shot noise plus Gaussian read noise against a
full-well capacity that scales with ISO, giving roughly 45 dB SNR at ISO 100
falling to 26 dB at ISO 3200. Signal-dependent, as real noise is: bright regions
are noisier in absolute terms, dark regions are dominated by read noise.

**Auto-exposure** is modelled explicitly, because the interesting failures are
where it runs out of range. `auto_exposure=0.9` is a camera that meters well;
the dusk and storm scenarios drop it to 0.3–0.6, which is when you get a dark
frame the camera then tries to gamma-lift into visibility.

## The scenarios

Ten named sites rather than ten filters. Each fixes a mounting, an exposure and
the weather; the plate's own condition (clean, dirty, damaged) is drawn
independently, so a filthy plate can also be a night shot in the rain.

| Scenario | What it is |
|---|---|
| `daylight` | midday, fast shutter, the case that should always work |
| `night_ir` | IR illuminator on a retroreflective plate |
| `night_glare` | oncoming headlight in frame, auto-exposure fooled |
| `monsoon` | heavy rain, streaks, veiling and a wet lens |
| `fog` | winter morning on the bypass |
| `high_speed` | expressway speed against a shutter too slow for it |
| `far_lane` | the far carriageway: a small plate at a hard angle |
| `dusk_highiso` | the worst hour — too dark for the shutter, no IR yet |
| `cheap_cam` | an old low-bitrate camera and heavy digital zoom |
| `storm` | everything at once; the case that should be **refused** |

## On Albumentations

It was worth considering and I recommend against taking the dependency here.

`RandomRain`, `RandomFog`, `RandomSunFlare`, `ISONoise`, `MotionBlur`,
`ImageCompression`, `Downscale` and `Perspective` cover much of this vocabulary
and are well tested. What they cannot express is the part that matters:

* **Ordering.** A transform pipeline applies effects in a list, but it has no
  notion that compression must follow noise and precede the digital zoom.
* **Shared physical state.** Here, one `Capture` object drives every stage —
  the same exposure time sets both rain-streak length and motion-blur length,
  and the same rig geometry sets both perspective and apparent size. In a
  transform list those are unrelated random parameters that can contradict each
  other, producing a frozen plate in streaking rain.
* **Plate-specific physics.** Retroreflective sheeting under IR has no
  equivalent transform, and it is the single most distinctive property of a
  night ANPR capture.

**If you want it anyway**, the seam is clean: every stage in `anpr/camera.py` is
a standalone function taking `(img, cap, rng)`, so an Albumentations `Compose`
drops in at stage 3–7 for extra variety:

```python
import albumentations as A
extra = A.Compose([A.ISONoise(p=0.3), A.ImageCompression(quality_lower=25, p=0.3)])

def shoot(plate, cap, rng):
    ...
    img = sensor_noise(img, cap, rng)
    img = extra(image=img)["image"]        # slot in here
    img = jpeg(img, cap.jpeg_quality)
```

Keep it after the physical stages, not instead of them.

## What a harder corpus did to training

Worth recording, because it was the most instructive failure of the exercise.

The classifier is trained in stages: stage 1 keeps only plates whose
segmentation box count exactly matches the plate length, so every label is
certain; later stages harvest crops from plates the decoder read correctly.
Moving to the camera model **starved stage 1**. On camera-realistic captures the
segmenter rarely cuts a plate perfectly, so the certain-label set collapsed from
30,234 glyphs to 6,964 and held-out glyph accuracy fell from 98.7% to 80.0%.
Benchmarked, that model managed 13.9%.

The scheme had been quietly depending on the corpus being easy. The fix is a
curriculum:

| Stage | Corpus | Why |
|---|---|---|
| 1 | **bootstrap captures** — good site, good day: near lane, long lens, 1/1000 s, ISO 100 | still fully camera-modelled, but segmentation can be trusted to label itself (~45% align) |
| 2 | the **ordinary** mix — daylight, IR night, some weather | the stage-1 engine can decode these, so their crops can be harvested by the decoder's own alignment |
| 3 | the **hard** mix — far lane, cheap camera, dusk, storm | the stage-2 engine reaches captures stage 1 could not read at all |

Each pass reads plates the previous one could not, so the training set grows
into the difficulty instead of being defeated by it. If you make the corpus
harder again, expect to add a stage rather than to turn the difficulty down.

## What the corpus did to the architecture

The curriculum rescued the classical classifier's *training*, but not its
accuracy: it reached 16% on camera-realistic captures against 87% on the old
filter-based ones. Ruling things out one at a time — perspective rectification,
decoder merge depth, atom caps, morphology kernel scaling, training mix — moved
that by a point or two at most, which is itself the finding. The limit was not a
parameter. It was that **segmentation is a decision made too early**: after a
JPEG encoder has worked on a 90-pixel plate and an operator has zoomed it 3x,
the gaps between characters are no longer reliably in the image, so a pipeline
that must find them before recognising anything is betting on information that
has been destroyed.

That is what motivated the CRNN in `anpr/crnn.py`, which never makes the
decision: CTC marginalises over every alignment between the network's output
columns and the string, so character boundaries stay latent. On identical
captures it doubles accuracy (16.4% to 35.0%) at a seventh of the cost per read.

The lesson for the corpus is the one worth keeping: a weak augmentation model
does not just inflate your headline number, it hides which part of your
architecture is wrong. The filter-based corpus made a segment-and-classify
pipeline look like a reasonable design.

## Calibrating it

The model was wrong three times before it was right, and each error was caught
by looking rather than reasoning:

1. **Focal length too short** — plates came out 30–107 px, which nothing can
   read. Enforcement cameras use telephoto lenses precisely so the plate clears
   the ~20 px character height that OCR needs.
2. **ISO multiplying scene brightness** instead of compensating for its absence,
   so a midday frame at ISO 100 rendered at 18% brightness.
3. **Motion blur using the wrong velocity component**, smearing every head-on
   capture into a streak.

The tool that caught all three is `python -m anpr.camera_sheet` — render every
scenario, put them side by side, and look. If the `daylight` tile is not
comfortably readable by eye, the model is wrong, not the OCR.

The objective check is the benchmark: `daylight` and `night_ir` should be the
easiest conditions and `storm` the hardest, with `far_lane` and `cheap_cam`
sitting where the plate is simply too small to recover.
