# Chart captions

Thirteen charts, all rendered from files in this repo by `python -m viz.charts`.
Each PNG carries its own source line in the bottom-left corner. Nothing here is
an artist's impression: where a number has no JSON behind it, the caption says
which README table it was copied from.

**The four strongest for a five-minute pitch:** 02, 05, 07, 12.

---

## The plate reader — synthetic benchmark

**01_burst_vs_single.png — Reading a burst, not a photograph**
Exact whole-registration match per capture condition, 100 plates each. A camera
is triggered as a vehicle enters the zone and contributes several frames, which
are decoded independently and voted by CTC score. That takes the platform from
**42.4%** on one frame to **73.1%** on five, with identical weights. `storm`
reads 0% and should: ranking the true plate against 199 decoys under the exact
likelihood succeeds 2.5% of the time, so the registration is not recoverable
from those pixels by any method.
Source: `models/benchmark.json`.

**02_stored_vs_refused.png — A refused read is not a wrong read**
The honest-precision chart, and the one to show first. Each bar is 100 captures
split three ways: stored and correct, stored and wrong, refused below the
confidence floor. Of 1,000 captures the engine stores 713, and **96.2%** of
those are exactly right — the number an operator actually experiences, because
a read below the floor is discarded rather than displayed. The floor is an
operating point chosen on the benchmark, not a probability.
Source: `models/benchmark.json`, `config.MIN_PLATE_CONFIDENCE_CRNN`.

**03_accuracy_funnel.png — What reaches the operator**
The same argument in three numbers: 1,000 captured, 713 confident enough to
store, 686 exactly right.
Source: `models/benchmark.json`.

**04_layout_difficulty.png — Where the plate itself is the problem**
Single frame, 100 plates per layout. A clean one-row plate reads 46%; grime
takes it to 28% and a second row to 32%. Grime and damage are drawn
independently of the weather, so a filthy plate can also be a night shot in the
rain.
Source: `models/benchmark.json`.

## Real photographs

**05_backends_real.png — Synthetic training does not transfer**
532 real Indian plate crops, five backends, scored on exact match of the whole
registration. Two arguments in one image. First, the CRNN trained on our own
camera model reads **0.9%** of real plates: it learned that renderer's artifact
signature, not the shapes of the glyphs. Second, taking a recogniser pretrained
on 558k real plates and fixing its decoder — grammar-constrained beam search,
two-row splitting, legality ranked above confidence — moves it from **79.5% to
91.0% with no retraining and no new data**.
Source: `README.md:462-468`; reproduce with `python -m anpr.compare_backends --real-all`.

**06_seed_stability.png — One seed is not a number**
Each dot is one of eight vehicle-held-out splits; the diamond is the mean. With
~118 unique vehicles, which ones land in validation moves the result more than
a small code change does — so a single seed is not a result. Computed live from
the per-seed file rather than transcribed.
Source: `models/backend_comparison_real.json`.

**07_label_fix.png — The ground truth was wrong**
548 dashcam frames had been labelled by propagating one hand-typed plate across
each cluster of adjacent frames. Adjacency is not vehicle identity: one cluster
held six different vehicles under one label, another held eight. **37% of frames
carried a plate that is not in the picture** — a ceiling no recogniser could
ever have passed. Re-labelling them one frame at a time moved the *unchanged*
upstream decoder from 48.7% to 79.5%. Worth showing to any judge who asks how
the accuracy was validated.
Source: `README.md:498-505`, `REAL_LABELS.md`, `real_plate_frame_labels.json`.

**08_whole_photo_e2e.png — End to end on whole photographs**
Not pre-cropped plates: whole photos, detector included — the path the platform
actually deploys. Eight splits, mean with the min–max range. YOLOv8-OBB plus
`anpr_ocr` averages **91.5%**, every seed clears 90%, and the spread is 4.8
points, the tightest of any mode — the detector contributes essentially no
variance.
Source: `README.md:541-547`.

## The platform

**09_camera_network.png — 35 cameras, 52 road links**
Real Kolkata junction coordinates. Roads are drawn from their own shape points,
so a corridor bends the way it bends instead of running straight between two
dots. Camera size is read volume over the seeded six hours; link colour is
measured congestion, not an estimate.
Source: `data/cameras.json` + `data/godseye.db`.

**10_alerts_by_rule.png — Alerts raised in six hours**
Six ingest-time rules, each checked as a sighting lands against a short history
of the same plate. De-duplicated per plate per rule within a rolling window,
which is why a listed plate seen all day is many alerts rather than one. Only a
`confident` clone verdict reaches this queue — a suspicion stays on screen with
its qualification.
Source: `data/godseye.db` via `core.alerts`.

**11_traffic_6h.png — Six hours of city traffic**
Reads and distinct vehicles per five-minute bucket, both counts on one axis. The
gap between the two lines is how often the same vehicle is seen twice, which is
the primitive every trajectory and origin–destination figure is built from.
Source: `data/godseye.db` via `core.analytics.flow_trend`.

**12_bottlenecks.png — Ranked by delay caused, not by slowness**
A slow empty road is not a problem. Ranking by vehicle-minutes of delay —
volume times delay per vehicle — reorders the list against speed alone, and it
is the ordering an operator should be dispatched on. Each corridor appears once
per direction, because a road that crawls inbound and runs free outbound is the
normal case and an average would hide it.
Source: `data/godseye.db` via `core.analytics.bottlenecks`.

## The detector

**13_yolo_training.png — Fine-tuning the plate detector**
mAP50-95 over 50 epochs for three successive fine-tuning runs, each resuming
from the previous checkpoint. **State the caveat if this chart is shown:** the
Roboflow split is image-level and does not separate video sequences, so
near-identical frames of the same vehicle leak between train and validation and
the absolute mAP is inflated. The trustworthy statement about this detector is
the behavioural one — it finds a candidate plate region in 100% of real photos
tried — and chart 08, which measures it end to end on a vehicle-held-out split.
Source: `runs/obb/runs/*/results.csv`, `yolo_accuracy.txt`, `CLAUDE.md`.
