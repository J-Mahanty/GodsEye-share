# GodsEye — running it

City-wide ANPR intelligence platform. SIH 2026, Team Prometheus.

## Two minutes to a running control room

```bash
pip install -r requirements.txt

python seed.py                          # ~6 h of simulated city traffic into SQLite (~65 s)
streamlit run dashboard/app.py          # the control room -> http://localhost:8501
```

That is enough to demo. The dashboard reads the database directly and works
with the API stopped. For the live camera feed and the incident injectors, in a
second terminal:

```bash
uvicorn api.main:app --port 8000        # http://localhost:8000/docs
```

Python 3.11+. No GPU, no internet connection and no dataset download is needed
for any of the above.

## What to try first

- **Live network** — the Kolkata camera network, roads coloured by congestion.
- **Track a plate** — hit "Pick a busy vehicle", then retype the plate with one
  character wrong. It still finds the vehicle: search is confusion-aware and
  charges 0.4 rather than 1.0 for a substitution the OCR engine actually makes.
- **ANPR engine** — pick a camera scenario (try `monsoon` or `far_lane`) and
  press Capture & read.

## Checking it yourself

```bash
python tests.py                                  # 58-check end-to-end self-check
python -m viz.charts                             # regenerate the 13 charts in charts/
python -m anpr.benchmark --samples 100 --burst 5 # re-measure the reader (~7 min)
python -m anpr.camera_sheet                      # see what the camera model produces
python -m core.repair --hours 6                  # recover refused captures
```

`python tests.py` points the database path at a scratch temp file, so it never
touches `data/godseye.db`. Run it before any demo.

## The models are here, and that matters

This bundle carries `models/crnn.pt`, `models/glyph_mlp.joblib` and
`models/yolov8n-obb-plate.pt`. **A `git clone` does not.** `models/*.joblib` is
in `.gitignore` and the YOLO detector was never committed, so a fresh clone
silently retrains the glyph classifier on first use and falls back to the
classical morphological localiser. Working from this folder avoids both.

`torch` is an optional dependency. Without it `anpr/ocr.py` falls back to the
classical segment-and-classify reader and the platform still runs:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## What this bundle cannot do

It **cannot reproduce the 91.0% real-photograph accuracy**. That number comes
from the `anpr_ocr` backend, which needs `paddlepaddle`:

- paddle has no Python 3.14 wheels, so it needs its own environment on 3.12;
- on first use it clones PaddleOCR (~2.3 GB) and downloads the HF weights;
- importing `paddle` before torch's models are loaded corrupts torch's tensors —
  a binary symbol clash between the two frameworks' compiled extensions.

**`PADDLE_SETUP.md` is the step-by-step guide to setting all of that up** —
Python 3.12, the second virtualenv, the weights, and the load-order hazard.
The real-photo corpus needed to reproduce the 91.0% is included in this bundle
(`dataset_v1_bbox/`, `indian-number-plates-dataset/`, `real_plate_crops/` and
the label files), so the benchmark commands run from here once paddle is set
up. See the licence note at the end of this file. The
training datasets, the YOLO run directories and both virtualenvs are all left
out of this bundle deliberately — some of the datasets are CC BY-NC-ND and
should not be redistributed.

## Where to read

| File | What is in it |
|---|---|
| `README.md` | what it does, how it is built, every measured number |
| `charts/CAPTIONS.md` | the 13 judge-facing charts and what each one argues |
| `AUGMENTATION.md` | the camera model — the 8 stages and why they are in that order |
| `REAL_LABELS.md` | how the real-photo ground truth was rebuilt |
| `PADDLE_SETUP.md` | how to set up Python 3.12 + paddle for the 91% reader |
| `CLAUDE.md` | working notes: what has been tried, what failed, and why |
| `share.html` | a single self-contained page — open it in any browser, no install |

## Licence note on the bundled datasets

This bundle carries the real-photo corpus so the accuracy numbers can be
re-measured rather than taken on trust. Two of those directories are
third-party data, redistributed unmodified:

- `indian-number-plates-dataset/` — Dataclusterlabspvtltd/indian-number-plates-dataset
  (Hugging Face), **CC BY-NC-ND 4.0**: attribution required, non-commercial use
  only, and no distribution of modified versions. Keep it inside the team and
  do not republish altered copies.
- `dataset_v1_bbox/` — a Roboflow export; check its own `README.roboflow.txt`
  for the licence that applies to it.

`real_plate_frame_labels.json` is this project's own work: 494 dashcam frames
re-transcribed by hand, replacing cluster-propagated labels that were wrong on
37% of frames. See `REAL_LABELS.md`.
