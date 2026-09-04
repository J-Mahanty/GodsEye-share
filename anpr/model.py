"""Glyph classifier — the learned half of the ANPR engine.

Trained in two stages, both on synthetic plates with known ground truth:

* **Stage 1 — segmentation-aligned.** Plates are pushed through the exact
  inference-time segmenter and kept only when it produced as many boxes as the
  plate has characters, so every crop's label is certain. Cheap, but it only
  ever sees plates the segmenter already handles well, which biases the
  classifier away from the hard conditions.

* **Stage 2 — decoder-aligned self-training.** The stage-1 engine reads a fresh
  batch of plates; where the decoded string matches ground truth, the decoder's
  own character spans are harvested as labelled crops. This recovers exactly the
  cases stage 1 threw away — merged characters, mud-covered strokes, fragments
  glued back together — and is what lifts accuracy on the dirty and damaged
  conditions.

Training takes 3-5 minutes on a laptop CPU and is cached in models/.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import joblib
import numpy as np

import config
from anpr import camera, ocr, plates, segment


@dataclass
class GlyphModel:
    clf: object
    classes: np.ndarray
    trained_on: int
    val_accuracy: float
    stages: int = 1

    def predict(self, feats: np.ndarray):
        """-> (chars, confidences) for a stack of glyph feature vectors."""
        if len(feats) == 0:
            return [], np.zeros(0)
        proba = self.clf.predict_proba(feats)
        idx = proba.argmax(axis=1)
        return [self.classes[i] for i in idx], proba[np.arange(len(idx)), idx]


def _plate_text(rng: random.Random) -> str:
    """Mostly real Bharat-series plates, plus a slice using the letters the
    series never issues (I, O) so the classifier still knows them."""
    text = plates.random_plate(rng)
    if rng.random() < 0.08:
        head = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(2))
        text = head + text[2:]
    if rng.random() < 0.08:
        n = sum(c.isalpha() for c in text[4:-4]) or 1
        mid = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
        text = text[:4] + mid + text[-4:]
    return text


def bootstrap_capture(text: str, rng: random.Random):
    """A capture from a camera doing its job well.

    Not a clean render - it still goes through the whole camera model, so the
    crops carry perspective, compression and sensor noise. It is simply a good
    site on a good day: near lane, long lens, fast shutter, low gain. These are
    the captures whose segmentation can be trusted to label itself.
    """
    rig = camera.CameraRig(height_m=rng.uniform(5.0, 7.0),
                           distance_m=rng.uniform(10.0, 17.0),
                           lateral_m=rng.uniform(-2.5, 2.5),
                           focal_px=rng.uniform(5000.0, 7200.0))
    cap = camera.Capture(rig=rig, weather=camera.Weather(),
                         exposure_s=rng.choice([1 / 2000, 1 / 1000]), iso=rng.choice([100, 200]),
                         speed_kmph=rng.uniform(20, 45), ambient=1.0,
                         jpeg_quality=rng.randint(84, 95),
                         digital_zoom=rng.uniform(1.0, 1.4),
                         defocus_px=rng.uniform(0.2, 0.55))
    plate = plates.render_plate(text, commercial=rng.random() < 0.18,
                                width=rng.choice([640, 760, 900]), rng=rng,
                                two_row=rng.random() < 0.18)
    if rng.random() < 0.22:
        plate = plates.surface_fault(plate, rng.choice(["dirty", "damaged"]), rng, 0.45)
    return camera.shoot(plate, cap, rng)


def build_stage1(n_plates: int = 3000, seed: int = 7, verbose: bool = True):
    """Segmentation-aligned crops from bootstrap captures."""
    rng = random.Random(seed)
    X, y = [], []
    kept = 0
    for i in range(n_plates):
        text = _plate_text(rng)
        image = bootstrap_capture(text, rng)
        _, ink, boxes, feats = segment.plate_glyphs(image)
        if len(boxes) != len(text):
            continue
        kept += 1
        X.extend(feats)
        y.extend(list(text))
        if verbose and (i + 1) % 750 == 0:
            print(f"  stage 1: {i+1}/{n_plates} bootstrap plates, {kept} aligned, "
                  f"{len(y)} glyphs")
    return X, y


ORDINARY_POOL = (["daylight"] * 5 + ["night_ir"] * 4 + ["high_speed"] * 2 +
                 ["monsoon"] * 2 + ["fog"] * 2 + ["night_glare"] * 2 +
                 ["far_lane"] + ["dusk_highiso"] + ["cheap_cam"])
HARD_POOL = (["far_lane"] * 3 + ["cheap_cam"] * 3 + ["dusk_highiso"] * 3 +
             ["storm"] * 3 + ["night_glare"] * 3 + ["monsoon"] * 2 +
             ["fog"] * 2 + ["high_speed"] * 2 + ["night_ir"] + ["daylight"])


def build_stage2(eng, n_plates: int = 2200, seed: int = 21, verbose: bool = True,
                 pool: list[str] | None = None, stage: int = 2):
    """Decoder-aligned crops harvested from plates this engine read correctly.

    Run once against the ordinary mix, then again with the improved engine
    against the hard mix: the second pass reaches captures the first could not
    decode, which is the whole point of doing it twice.
    """
    rng = random.Random(seed)
    X, y = [], []
    kept = 0
    pool = pool or HARD_POOL
    for i in range(n_plates):
        text = _plate_text(rng)
        cap = plates.capture(text, rng.choice(pool), rng)
        read, dbg = eng.read_detailed(cap.image)
        if read.text != cap.text or dbg.get("ink") is None:
            continue
        kept += 1
        ink, atoms, spans = dbg["ink"], dbg["atoms"], dbg["spans"]
        for ch, (a, b) in zip(read.text, spans):
            X.append(segment.glyph(ink, ocr.union_box(atoms, a, b)))
            y.append(ch)
        if verbose and (i + 1) % 500 == 0:
            print(f"  stage {stage}: {i+1}/{n_plates} plates, {kept} decoded correctly, "
                  f"{len(y)} glyphs")
    return X, y


def _jitter(X, y, copies: int = 2, seed: int = 0):
    """Extra copies of each crop, shifted and scaled a little.

    Harvested crops are the ones that resemble real captures, but there are far
    fewer of them than bootstrap crops, so the classifier ends up fitted to the
    easy distribution. Held-out glyph accuracy said 97.4% while daylight plates
    read at 40% - and 0.40 over ten characters implies about 91% per character,
    which is the gap between the two distributions. Jittering the realistic
    crops rebalances the mix without another hour of harvesting.
    """
    import cv2

    g = int(np.sqrt(len(X[0]))) if len(X) else segment.G
    rng = np.random.default_rng(seed)
    outX, outY = list(X), list(y)
    for _ in range(copies):
        for feat, label in zip(X, y):
            img = np.asarray(feat, np.float32).reshape(g, g)
            dx, dy = rng.uniform(-1.2, 1.2, 2)
            scale = rng.uniform(0.94, 1.06)
            m = cv2.getRotationMatrix2D((g / 2, g / 2), rng.uniform(-4, 4), scale)
            m[:, 2] += (dx, dy)
            warped = cv2.warpAffine(img, m, (g, g), flags=cv2.INTER_LINEAR,
                                    borderValue=0.0)
            outX.append(warped.ravel())
            outY.append(label)
    return outX, outY


def _fit(X, y, seed: int, verbose: bool):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X = np.array(X, np.float32)
    y = np.array(y)
    if len(X) < 500:
        raise RuntimeError(f"only {len(X)} glyphs available - check the segmenter")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.12, random_state=seed, stratify=y)
    if verbose:
        print(f"[glyph] training MLP on {len(Xtr)} glyphs ({len(set(y))} classes) ...")
    clf = MLPClassifier(hidden_layer_sizes=(384, 192), activation="relu", alpha=1e-4,
                        batch_size=256, learning_rate_init=2e-3, max_iter=110,
                        early_stopping=True, n_iter_no_change=8, random_state=seed)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xva, yva))
    if verbose:
        print(f"[glyph] held-out glyph accuracy {acc:.4f}")
    return clf, acc, len(Xtr)


def train(n_plates: int = 4000, seed: int = 7, self_train: bool = True,
          harvest_plates: int = 3000, verbose: bool = True) -> GlyphModel:
    from anpr.ocr import ANPREngine

    t0 = time.time()
    if verbose:
        print(f"[glyph] stage 1: {n_plates} bootstrap captures (good site, good day) ...")
    X, y = build_stage1(n_plates, seed, verbose)
    clf, acc, n = _fit(X, y, seed, verbose)
    model = GlyphModel(clf=clf, classes=np.array(clf.classes_), trained_on=n,
                       val_accuracy=acc, stages=1)
    if not self_train:
        return model

    if verbose:
        print(f"[glyph] stage 2: harvesting from {harvest_plates} ordinary captures ...")
    X2, y2 = build_stage2(ANPREngine(model), harvest_plates, seed + 14, verbose,
                          pool=ORDINARY_POOL, stage=2)
    if len(X2) < 500:
        print("[glyph] stage 2 harvested too little - keeping the stage 1 model")
        return model
    X2j, y2j = _jitter(X2, y2, copies=2, seed=seed)
    clf, acc, n = _fit(X + X2j, y + y2j, seed, verbose)
    model = GlyphModel(clf=clf, classes=np.array(clf.classes_), trained_on=n,
                       val_accuracy=acc, stages=2)

    # Stage 3: the improved engine can read captures stage 2 could not, so a
    # second harvest reaches further into the hard scenarios.
    if verbose:
        print(f"[glyph] stage 3: harvesting from {harvest_plates} hard captures ...")
    X3, y3 = build_stage2(ANPREngine(model), harvest_plates, seed + 28, verbose,
                          pool=HARD_POOL, stage=3)
    if len(X3) < 400:
        if verbose:
            print("[glyph] stage 3 harvested too little - keeping the stage 2 model")
        return model
    X3j, y3j = _jitter(X3, y3, copies=3, seed=seed + 1)
    clf, acc, n = _fit(X + X2j + X3j, y + y2j + y3j, seed, verbose)
    if verbose:
        print(f"[glyph] done in {time.time()-t0:.1f}s ({len(X)} bootstrap + "
              f"{len(X2j)} ordinary + {len(X3j)} hard glyphs after jitter; "
              f"{len(X2)} + {len(X3)} harvested)")
    return GlyphModel(clf=clf, classes=np.array(clf.classes_), trained_on=n,
                      val_accuracy=acc, stages=3)


def save(model: GlyphModel, path=None) -> None:
    path = path or config.GLYPH_MODEL
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_or_train(force: bool = False, **kw) -> GlyphModel:
    path = config.GLYPH_MODEL
    if path.exists() and not force:
        try:
            return joblib.load(path)
        except Exception as exc:       # corrupt or version-skewed cache
            print(f"[glyph] could not load {path} ({exc}); retraining")
    model = train(**kw)
    save(model)
    return model


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train the GodsEye glyph classifier")
    ap.add_argument("--plates", type=int, default=4000)
    ap.add_argument("--harvest", type=int, default=3000)
    ap.add_argument("--no-self-train", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    m = train(args.plates, args.seed, not args.no_self_train, args.harvest)
    save(m)
    print(f"saved -> {config.GLYPH_MODEL}  (stages {m.stages}, glyph val acc {m.val_accuracy:.4f})")


if __name__ == "__main__":
    # Run through the canonical module, not this file's __main__ copy of it.
    # Pickle stores a class by module path, so a GlyphModel built here would be
    # saved as __main__.GlyphModel and would fail to load in every other
    # process - which silently costs a five-minute retrain each time.
    import anpr.model

    anpr.model.main()
