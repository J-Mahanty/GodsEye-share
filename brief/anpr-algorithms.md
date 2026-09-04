# The ANPR engine — algorithms in detail

*GodsEye · Team Prometheus · technical reference*

Follows the path an image actually takes through the system. File references are to the
repository; every figure quoted was measured, and the command that produced it is named.

---

## 0 · Dispatch

`ANPREngine.read()` (`anpr/ocr.py`) selects a backend at construction time:

- the **CRNN** when PyTorch and `models/crnn.pt` are both present
- otherwise the **classical segment-and-classify** pipeline

Each is judged against its own confidence floor — `MIN_PLATE_CONFIDENCE_CRNN = 0.96`,
`MIN_PLATE_CONFIDENCE = 0.40` — because the two produce quantities that are not comparable.
`PlateRead.accepted` asks the read which floor applies rather than consulting a constant.

Both paths are described below; the classical one is live code that runs whenever torch is
absent.

---

## 1 · Localisation — finding the plate in a photograph

`detect_plate_candidates()` runs only on the whole-frame path (the dashboard upload box and
`POST /api/anpr/read`). A plate is modelled as **a horizontal band of tightly-spaced
vertical strokes**.

1. Downscale to ≤ 900 px wide; bilateral filter (5, 40, 40) — denoise while preserving edges
2. **Morphological gradient**, 3×3 rectangular element → Otsu threshold → binary edge map
3. **Wide rectangular closing** at three kernel scales — (21×5), (35×9), (13×5) — welding
   the characters of a plate into a single blob while leaving most scene texture separate
4. Contours → bounding boxes, filtered on:
   - aspect ratio **1.6 – 7.5** (admits both one-row and two-row layouts)
   - fill ratio **≥ 0.35**
   - interior edge density **≥ 0.12** — characters are busy, a wall is not
5. Score each candidate by `density × area`; sort; greedy **non-maximum suppression at
   IoU 0.35**; keep at most 12
6. Each surviving box is decoded at several insets (`_crop_variants`), because a box that
   carries a strip of bumper into the reader performs worse than one cut slightly tight

**Measured: 56.7% recall** at IoU > 0.3. Recall barely moves between IoU 0.1 and 0.5, which
means these are hard misses rather than sloppy boxes. This is the weakest component in the
system, and it is a hand-tuned heuristic — `detect_plate_candidates()` is the documented
seam for a trained detector.

---

## 2 · Primary recogniser — CRNN + CTC

### 2.1 Why not segment first

The classical pipeline must commit to character boundaries **before** recognising anything.
After a gantry capture has passed through a JPEG encoder at 90 px wide and a 3× digital
zoom, those boundaries are not reliably present in the image.

> Segmentation is a decision made too early. Any pipeline that must find character
> boundaries before recognising anything is betting on information the image no longer
> contains.

A CRNN never makes that decision. Character boundaries remain latent, and CTC marginalises
over every alignment between output columns and the output string.

### 2.2 Preprocessing — `crnn.prepare()`

Deliberately minimal, because each classical step discards information on the assumption
that what remains is sufficient — and on a 90 px JPEG plate that assumption is what fails.

- optional two-row unwrap (§2.3)
- **CLAHE**, clip limit 2.5, 8×8 tiles — recovers local contrast on fogged and
  under-exposed captures without committing to a threshold
- resize to **32 × 160**, `INTER_AREA`
- per-image standardisation, `(x − μ) / σ`

No binarisation, no deskew, no segmentation.

### 2.3 Row-layout hypotheses — `row_hypotheses()`

A two-row motorcycle plate squashed into a 32-pixel strip leaves each row roughly fourteen
pixels tall. It must be unwrapped into a single line first. Three candidate layouts are
generated:

| hypothesis | method |
|---|---|
| **as-is** | the crop unchanged |
| **band split** | horizontal ink projection (Gaussian σ 1.5, row means, normalised) → threshold at 0.45 → contiguous bands. Requires *exactly two*, of similar height, neither exceeding 0.6 × crop height. Each half trimmed to its own column extent, which removes the blue IND band, then stacked side by side |
| **forced split** | cut at the **argmin of the ink profile within the middle third**; halves trimmed and stacked |

All three pass through the network in a **single batched forward pass**, and the exact CTC
score selects the winner, preferring a grammatical string over an ungrammatical one.

This replaced a single threshold that was wrong in both directions:

- it **skipped 30%** of genuine two-row plates, which then read at **3.3%**
- it **misfired on 6.7%** of single-row plates, every one of which read at **0%** — splitting
  one line in half places the back of the registration in front of its own start

Worth **+1.0 points** measured in isolation on identical plates.

### 2.4 Network architecture — `build_model()`

```
Conv 3×3 (1   → 32)   BN  ReLU  MaxPool(2,2)    32×160 → 16×80
Conv 3×3 (32  → 64)   BN  ReLU  MaxPool(2,2)           → 8×40
Conv 3×3 (64  → 128)  BN  ReLU  MaxPool(2,1)           → 4×40    width preserved
Conv 3×3 (128 → 128)  BN  ReLU  MaxPool(4,1)           → 1×40
squeeze                                                 → (B, 40, 128)
BiGRU, 2 layers, hidden 128, dropout 0.1                → (B, 40, 256)
Linear (256 → 37)                            36 alphabet classes + CTC blank
```

**745k parameters.** Height collapses to 1 while width resolution is deliberately kept,
giving **40 timesteps** for a 10–11 character plate — sufficient for CTC to place a blank
between every adjacent pair. The bidirectional GRU carries context along the plate, which is
what allows the model to use *"character four is a digit"* when character four is smeared.

Note that no parameter shape depends on width: a checkpoint trained at `IMG_W = 160` will
run at any width. It is simply not *trained* for it — zero-shot accuracy at 320 is 0.0%.

### 2.5 Decoding

**Greedy CTC collapse** — argmax per timestep, drop repeats, drop blanks.

**Grammar-constrained prefix beam search** — a proper CTC prefix beam tracking **two**
probabilities per prefix: the mass of paths ending in a blank (`p_b`) and the mass ending in
the prefix's own last character (`p_nb`). Collapsing them, as a naive beam does, mishandles
repeated characters and double-counts merging paths. One pass per registration shape; within
each slot only the top-5 legal extensions are expanded; beam width 12. A digit can never win
a letter slot.

**The policy is greedy-first**, with the beam run only when greedy returns a string that is
not a valid registration. The measurements justify it:

- greedy 44.3% vs always-on beam 45.0%
- widening the beam from 12 → 48 → 128 moves the result **not at all** (38.7% at every width)
- decisively: on **251 failed reads, zero** had the true plate scoring higher than the string
  the model emitted; median score gap −0.586 against the truth

Search is exhausted. The model prefers the wrong answer, so every remaining point is a model
problem rather than a decoding one.

### 2.6 Confidence — exact CTC forward–backward, `ctc_score()`

Build the extended label sequence `l′ = [blank, l₁, blank, l₂, …, blank]`, length `S = 2K+1`.

- **Forward**  `α[t,s] = em[t,s] + logsumexp(α[t−1,s], α[t−1,s−1], α[t−1,s−2]·[skip])`
- **Backward** `β[t,s]` symmetrically
- **Skip mask** — the `s−2 → s` transition is permitted only when `l′[s]` is a real character
  *and* differs from `l′[s−2]`; a doubled letter requires a blank between its two instances
- `Z = logaddexp(α[T−1, S−1], α[T−1, S−2])`
- **Sequence confidence** = `exp(Z / K)` — the probability normalised per character, so
  plates of different lengths compare
- **Per-character confidence** = `max_t exp(α + β − em − Z)` evaluated at state `2i+1`

Validated two ways: the forward score matches `torch.nn.CTCLoss` to 1e-4 across random
cases, and the occupancy posteriors sum to exactly 1.0 at every column.

**This replaced a heuristic** that walked the argmax path with an advancing cursor, taking
each character's highest posterior in turn. Whenever the cursor mis-advanced — which it did
whenever a character peaked later than the one after it — every subsequent character was
searched in a window that no longer contained it.

| estimator | AUROC | reads kept at 90% precision |
|---|---|---|
| old heuristic (min of cursor scan) | **0.367** | 13.6% |
| min posterior at the true alignment | 0.971 | 45.0% |
| **normalised CTC sequence probability** | **0.978** | **45.3%** |

An AUROC below 0.5 means it ranked correct reads *below* wrong ones. Median confidence was
0.000 on correct reads against 0.070 on wrong ones.

### 2.7 State-code repair — `_repair_state()`

If the first two characters are not a real Indian state code, snap to the nearest one — but
only when *every* differing character is a member of `config.CONFUSION_PAIRS`
(`0↔O`, `8↔B`, `5↔S`, `1↔I`, `M↔N`, `2↔Z`, `6↔G` …).

A single confusable substitution is a misread. Two arbitrary substitutions are a different
plate, and the repair declines.

---

## 3 · Burst fusion — `read_burst()` / `fuse_reads()`

A real ANPR node is triggered by a loop or tripwire and captures 5–15 frames as the vehicle
crosses the zone — different distances, exposures and motion blurs of the same plate.

- each frame is decoded independently
- `score[text] += confidence`, summed across frames
- winner = argmax of that tally
- **reported confidence = the highest-confidence agreeing frame**, so the storage floor keeps
  exactly the meaning it was calibrated with
- `agreement` = share of frames that voted for the winner, reported separately

| frames fused | best-by-score | score-weighted vote |
|---|---|---|
| 1 | 40.4% | 40.4% |
| 3 | 62.8% | 62.8% |
| 5 | 71.2% | 71.2% |
| 8 | **76.8%** | **77.2%** |

The largest single gain in the system, achieved with unchanged weights — the frames fail in
*independent* ways. It depends entirely on §2.6: a vote is only as good as its ability to
distinguish a confident frame from a lucky one.

`config.BURST_FRAMES` sets the deployed value (default 5). `INLINE_OCR_MAX_PER_TICK` is a
budget of decoded *frames*, divided by the burst size, so raising the burst buys accuracy
rather than wall-clock.

---

## 4 · Classical fallback (no torch)

### 4.1 Normalisation — `segment.normalize()`

Target height follows the **aspect ratio** rather than being fixed: `NORM_H` when
`w/h ≥ 3.0`, else `1.7 × NORM_H`. A single-row Indian plate is about 4.5∶1 and a two-row
motorcycle plate nearer 2∶1 — normalising both to one pixel height would leave the two-row
characters half the size.

Then: deskew via `minAreaRect` on the ink cloud (applied only for rotations between 0.6° and
20°), bilateral filter (5, 45, 45), CLAHE.

### 4.2 Illumination flattening — `flatten_illumination()`

Divide out a morphological background estimate, so mud, shadow and glare gradients disappear
before thresholding.

### 4.3 Eight binarisation hypotheses — `ink_variants()`

No single threshold survives every condition: glare wants a local method, mud wants
illumination flattening, a low-resolution crop wants upscaling first.

| variant | method |
|---|---|
| `flat15`, `flat25` | illumination-flattened Otsu, kernel 15 / 25 |
| `flat15x2` | same, on a 2× cubic upscale, kernel 27 |
| `otsu` | plain Otsu |
| `adaptive` | adaptive Gaussian, block 31, C 14 |
| `stretch` | 4–96 percentile contrast stretch, then Otsu |
| `flat9` | flattening kernel just wider than one stroke |
| `mean41` | adaptive mean, block 41, C 8 |

All eight are decoded and the highest-scoring read wins, which is cheaper than being clever
about picking one up front.

### 4.4 Frame removal — `_clear_frame()`

Erase border-hugging connected components **only** when they are hollow, full-width or
full-height — so a plate frame that happens to touch a character does not take the character
with it.

### 4.5 Shear correction — `correct_shear()`

A plate shot from the side leaves upright strokes leaning, and the vertical ink projection
then has no clean valleys between characters. Search shear in `[−0.45, 0.45]` maximising
`mean(projection²)` — a peaky profile means deep gaps.

### 4.6 Segmentation — `segment()`

1. Connected components → bounding boxes
2. **Row clustering** by vertical centre: a gap greater than `0.62 × median height` starts a
   new line; keep lines with ≥ 2 boxes and ≥ 10% of total ink area; at most 2 lines
3. **Drop odd heights** — within a line, keep boxes between 0.55× and 1.7× the median
4. **Aggressive over-splitting** of wide blobs at vertical-projection minima

Step 4 is deliberate. These boxes are **atoms, not characters**: the decoder can merge
adjacent atoms back into one character but can never invent a cut that was not offered, so
over-cutting is the cheap error. `M` and `W` are protected by an aspect guard, because
cutting one produces two entirely convincing letters the decoder cannot argue with.

### 4.7 Dynamic-programming decoder — `_decode()`

`dp[a][k]` = best score for explaining atoms `0:a` as the first `k` characters of a grammar
pattern. One pass per pattern. Three moves at each step:

- take the next **1…`max_merge` (3) atoms as one character**, scored `log p` from the glyph
  classifier restricted to the slot's legal class set, plus `char_bonus = 0.25`
- **drop one atom as noise**, at `skip_penalty = 1.6`, discounted to 0.35× for specks below
  0.62 × median height or 0.42 × median area

`char_bonus` is an insertion reward borrowed from speech decoding. Without it the DP is free
to drop a difficult character to keep its average confidence high; a plate is not allowed to
be partially read, so paying for coverage is the correct trade. The skip move exists because
mud specks, rivets and frame fragments segment like characters, and without an escape hatch
the DP must fold them into a neighbour and corrupt it.

### 4.8 Confidence

The weakest character **×** the share of the eight binarisations that decoded the *same*
string.

Agreement is the informative half. The winner is chosen as the best-scoring of eight, so its
own confidence is inflated by that selection — measured over 500 reads it sits at 0.99 on
correct reads and 0.89 on wrong ones, which barely separates them. Agreement is not selected
for in the same way and scores 0.80 against 0.26.

---

## 5 · The glyph classifier — `anpr/model.py`

`MLPClassifier(hidden_layer_sizes=(384, 192), activation="relu", alpha=1e-4)` over 36
classes, trained in three stages, all on synthetic plates with known ground truth:

1. **Segmentation-aligned** — plates pushed through the real segmenter, kept only where the
   box count matches the plate length, so every label is certain
2. **Decoder-aligned self-training** — the stage-1 engine reads a fresh batch of hard plates;
   where the decode matches ground truth, the decoder's own character spans are harvested as
   labelled crops. This recovers exactly what stage 1 discards: merged characters,
   mud-covered strokes, fragments glued back together
3. **Jitter augmentation**

**53,665 training crops, 98.7% held-out accuracy** over 36 classes.

That figure also illustrates the trap: 98.7% per glyph across ten characters is 0.987¹⁰ ≈
88% per plate *at best*, and in practice the classical engine reaches 16.5% on
camera-realistic captures. Per-character accuracy is a misleading headline for a
whole-string task.

---

## 6 · Network-constrained repair — `core/repair.py`

Not image processing, but part of the recognition path. A capture refused by the confidence
floor retains its **CTC lattices** — float16, `savez_compressed`, roughly 32 KB per capture,
dropped after `REPAIR_EVIDENCE_TTL_S`.

1. **Candidate generation** — for each neighbouring camera, a travel-time window derived from
   the road graph: `free_flow × [0.6, 4.0] + 90 s`, evaluated both upstream and downstream,
   since the failed capture does not reveal direction of travel. Collect the plates seen there
2. **Scoring** — `ctc_score(lattice, candidate)` for each candidate, best row-hypothesis per
   frame, averaged across frames
3. **Acceptance** — an absolute likelihood floor **and** a margin over the runner-up, which
   together encode a "none of these" hypothesis for the case where the vehicle entered off a
   road no camera covers

This converts open-vocabulary recognition over **13.6 trillion** grammar-legal registrations
(summed across the eight registration shapes in `PATTERNS`) into closed-set retrieval over a
few hundred candidates.

| candidate set size | true plate ranked #1 |
|---|---|
| 10 | 78.2% |
| 50 | 72.9% |
| 200 | 69.2% |

Measured on a seeded six-hour day: **14 of 25** refused captures recovered, **14/14
correct**, 11 of them genuine repairs of a wrong decode rather than borderline reads.

It is explicitly **not** "detect which characters are missing and fill them in". That cannot
be built here: per-character CTC posteriors are peaked almost everywhere, so 99.8% of
characters return above 0.9 confidence *including the wrong ones*, and per-character
confidence predicts per-character correctness at only AUROC 0.692. The engine knows which
*strings* it doubts, not which *character* let it down.

---

## 7 · Where the difficulty actually lies

| stage | status |
|---|---|
| Decoding | **exhausted** — 0 of 251 failures recoverable, beam width irrelevant |
| Row layout | fixed, +1.0 point |
| Confidence | fixed, AUROC 0.367 → 0.974 |
| Burst fusion | largest gain, 40% → 77% |
| Network repair | 14/14 on refused captures |
| **The model** | **the entire remaining gap** |
| Localiser | 56.7%, and the reason end-to-end is 21.7% rather than 73% |

**Information ceiling.** Ranking the true plate against 199 decoys under the exact CTC
likelihood succeeds **83.2%** of the time, while greedy decoding reads **43.8%**. That ~40
point gap is recogniser, not optics.

Per condition, this separates three different problems:

- `storm` — ceiling **2.5%**. The registration is not recoverable from those pixels by any
  method; refusing is correct behaviour, not failure
- `far_lane` (ceiling 95%) and `cheap_cam` (ceiling 75%) — **not pixel-limited**. The plate
  is present and the recogniser is not finding it
- everything else clears 80% on the burst path

**What has been ruled out by experiment:**

- **Not capacity** — the current 745k architecture and a 2.95M one both memorise 400 captures
  at 100%
- **Not schedule** — an 18-epoch warm-started run beat the shipped weights only at epoch 1,
  then spent seventeen epochs with falling loss and flat accuracy, and benchmarked 2 points
  *worse* head to head

**The remaining hypothesis** is the training distribution. The model scores **4% on a clean,
undegraded render** and improves as the image is degraded toward what it trained on. No
scenario in the corpus has a digital zoom below 1.6, so it has never seen a sharp plate — it
has learned this camera model's artefact signature rather than the shapes of the glyphs.

---

## Reproducing the figures

```bash
python -m anpr.benchmark --samples 100 --scenes 60 --layouts 100 --burst 5 --json models/benchmark.json
python -m anpr.compare_backends --samples 60 --json models/backend_comparison.json
python -m core.repair --hours 6
python tests.py
```
