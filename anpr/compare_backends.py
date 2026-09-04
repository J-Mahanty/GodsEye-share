"""Head-to-head: classical vs CRNN vs EasyOCR vs anpr_ocr.

    python -m anpr.compare_backends --samples 100          # synthetic
    python -m anpr.compare_backends --real-all             # every real crop
    python -m anpr.compare_backends --real --seeds 8       # vehicle-split, real crops
    python -m anpr.compare_backends --real-frames --seeds 8  # whole photos, via YOLO

The default mode reads synthetic captures (`plates.capture()`) and is the
measurement that decides which backend the platform should ship with - it is
the one to re-run after any change to the camera model, since a corpus change
moves every backend's number and only the gap between them is meaningful.

The `--real*` modes are a different question: they read the real photo set
assembled by build_real_crnn_dataset.py's loaders, to check whether a backend
pretrained on real plates generalises somewhere our synthetic-only CRNN does
not. They are evaluation-only - they do not touch which backend the platform
ships with.

Which real mode to use:

  --real-all     every labelled crop, scored once, with a per-source
                 breakdown. No split, so no split noise: this is the mode for
                 comparing two decoders against each other.
  --real         held-out vehicles only, so it answers "plates it has not been
                 tuned against". Use --seeds 8: a single seed on ~118 vehicles
                 is not a stable number, and run_real_multiseed() reports
                 mean/min/max instead of one arbitrary draw.
  --real-frames  whole photographs rather than pre-made crops, so the
                 YOLOv8-OBB detector is in the loop too.

EasyOCR and anpr_ocr are opt-in everywhere: each is skipped automatically if
its package isn't installed. anpr_ocr needs paddle, which has no Python 3.14
wheels - run it under .venv-paddle.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import config
from anpr import anpr_ocr_reader
from anpr import crnn as crnn_mod
from anpr import easyocr_reader
from anpr import ocr, plates


def _build_engines() -> dict[str, ocr.ANPREngine]:
    engines = {"classical": ocr.ANPREngine(use_crnn=False)}
    reader = crnn_mod.load()
    if reader is None:
        raise SystemExit("no CRNN weights at models/crnn.pt - run "
                         "python -m anpr.train_crnn first")
    engines["crnn"] = ocr.ANPREngine(crnn_reader=reader)
    eo = easyocr_reader.load()
    if eo is not None:
        engines["easyocr"] = ocr.ANPREngine(easyocr_reader=eo)
    else:
        print("[compare_backends] easyocr not installed - skipping that backend "
              "(pip install easyocr)", file=sys.stderr)
    ao = anpr_ocr_reader.load()
    if ao is not None:
        engines["anpr_ocr"] = ocr.ANPREngine(anpr_ocr_reader=ao)
    else:
        print("[compare_backends] anpr-ocr (paddle) not available - skipping that backend "
              "(pip install paddlepaddle huggingface_hub)", file=sys.stderr)
    return engines


def run(samples: int = 100, seed: int = 4242, verbose: bool = True) -> dict:
    engines = _build_engines()
    names = list(engines)

    out: dict[str, dict] = {}
    if verbose:
        header = "".join(f"{n:>22}" for n in names)
        print(f"{'scenario':14}{header}")
    for cond in plates.CONDITIONS:
        rng = random.Random(seed)
        caps = [plates.capture(None, cond, rng) for _ in range(samples)]
        stats = {}
        for name in names:
            eng = engines[name]
            t0 = time.time()
            ok = kept = kept_ok = 0
            for c in caps:
                r = eng.read(c.image)
                hit = r.text == c.text
                ok += hit
                if r.accepted:
                    kept += 1
                    kept_ok += hit
            stats[name] = {
                "plate_accuracy": ok / samples,
                "stored": kept,
                "stored_accuracy": (kept_ok / kept) if kept else 0.0,
                "ms_per_read": (time.time() - t0) / samples * 1000,
            }
        out[cond] = stats
        if verbose:
            row = "".join(f"{stats[n]['plate_accuracy']:6.0%} "
                          f"(st {stats[n]['stored']:3}, {stats[n]['stored_accuracy']:4.0%}) "
                          for n in names)
            print(f"{cond:14} {row}", flush=True)

    # Snapshot the per-scenario rows first: adding OVERALL to `out` while
    # iterating over it is what made a backend disappear once before.
    rows = dict(out)
    overall = {}
    for name in names:
        acc = sum(v[name]["plate_accuracy"] for v in rows.values()) / len(rows)
        stored = sum(v[name]["stored"] for v in rows.values())
        stored_ok = sum(v[name]["stored"] * v[name]["stored_accuracy"]
                        for v in rows.values())
        overall[name] = {
            "plate_accuracy": acc,
            "stored": stored,
            "stored_accuracy": (stored_ok / stored) if stored else 0.0,
            "ms_per_read": sum(v[name]["ms_per_read"] for v in rows.values()) / len(rows),
        }
    out["OVERALL"] = overall
    if verbose:
        o = out["OVERALL"]
        row = "".join(f"{o[n]['plate_accuracy']:6.1%} "
                      f"(st {o[n]['stored']}, {o[n]['stored_accuracy']:.0%})       "
                      for n in names)
        print(f"\n{'OVERALL':14} {row}")
        speed = "".join(f"{o[n]['ms_per_read']:6.0f} ms/read       " for n in names)
        print(f"{'speed':14} {speed}")
    return out


def _load_real_dataset():
    """(crop, text, vehicle) triples for every real, labelled plate photo in
    the repo, using the same loaders and vehicle-identity grouping as
    build_real_crnn_dataset.py - so a crop and its near-duplicate frames
    (same vehicle crossing) never split across train/val here either.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from build_real_crnn_dataset import CAR_TEXTS, crop_bbox_image, frame_labels
    from finetune_crnn_real import load_real_crops as load_xml_crops

    crops, texts, vehicles = [], [], []

    xml_crops, xml_texts = load_xml_crops()
    for c, t in zip(xml_crops, xml_texts):
        crops.append(c); texts.append(t); vehicles.append(f"xml:{t}")

    for prefix, text in CAR_TEXTS.items():
        matches = [f for f in os.listdir("dataset_v1_bbox/train/images")
                   if f.startswith(prefix)]
        for fn in matches:
            crop = crop_bbox_image(fn)
            if crop is not None:
                crops.append(crop); texts.append(text); vehicles.append(f"car:{text}")

    for fn, text in sorted(frame_labels().items()):
        crop = crop_bbox_image(fn)
        if crop is not None:
            crops.append(crop); texts.append(text); vehicles.append(f"video:{text}")

    return crops, texts, vehicles


def _load_real_frames_dataset():
    """(full photograph, text, vehicle) triples from the same three real-label
    sources as _load_real_dataset()/build_real_crnn_dataset.py, but loading
    the whole image instead of the bbox-cropped plate region - this is what
    exercises detect_plate_candidates() (YOLOv8-OBB) rather than feeding a
    pre-made crop straight to the recogniser."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import xml.etree.ElementTree as ET

    import cv2

    from build_real_crnn_dataset import CAR_TEXTS, find_split_dir, frame_labels
    from finetune_crnn_real import REAL_DATASET_DIR

    frames, texts, vehicles = [], [], []

    # 1. XML-labelled real plates - whole image, not the <bndbox> crop.
    ann_dir = os.path.join(REAL_DATASET_DIR, "Annotations")
    img_dir = os.path.join(REAL_DATASET_DIR, "images")
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".xml"):
            continue
        root = ET.parse(os.path.join(ann_dir, fn)).getroot()
        img = cv2.imread(os.path.join(img_dir, root.find("filename").text))
        if img is None:
            continue
        for obj in root.findall("object"):
            text = None
            for attr in obj.findall(".//attribute"):
                name_el = attr.find("name")
                if name_el is not None and name_el.text == "number_plate_text":
                    val_el = attr.find("value")
                    text = val_el.text if val_el is not None else None
            if not text:
                continue
            text = text.strip().upper().replace(" ", "")
            if not text:
                continue
            frames.append(img); texts.append(text); vehicles.append(f"xml:{text}")

    # 2. filename-embedded plates - whole image, not the label-box crop.
    for prefix, text in CAR_TEXTS.items():
        matches = [f for f in os.listdir("dataset_v1_bbox/train/images")
                   if f.startswith(prefix)]
        for fn in matches:
            split = find_split_dir(fn)
            if split is None:
                continue
            img = cv2.imread(f"dataset_v1_bbox/{split}/images/{fn}")
            if img is not None:
                frames.append(img); texts.append(text); vehicles.append(f"car:{text}")

    # 3. dashcam video frames, one label per frame - whole image.
    for fn, text in sorted(frame_labels().items()):
        split = find_split_dir(fn)
        if split is None:
            continue
        img = cv2.imread(f"dataset_v1_bbox/{split}/images/{fn}")
        if img is not None:
            frames.append(img); texts.append(text); vehicles.append(f"video:{text}")

    return frames, texts, vehicles


def run_real_frames(val_vehicle_frac: float = 0.15, seed: int = 11, max_images: int = 0,
                    verbose: bool = True) -> dict:
    """Head-to-head on whole real photographs: detect_plate_candidates()
    (YOLOv8-OBB, same detector for every backend) finds the plate region,
    then each backend's read() decodes it via ANPREngine.read_frame(). Same
    vehicle-identity split as run_real() so membership is consistent between
    the two eval modes."""
    from anpr.ocr import _load_yolo
    if verbose:
        yolo_loaded = _load_yolo() is not None
        print(f"[real-frames] detector: {'YOLOv8-OBB (models/yolov8n-obb-plate.pt)' if yolo_loaded else 'classical fallback - YOLO weights not found'}")

    engines = _build_engines()
    frames, texts, vehicles = _load_real_frames_dataset()

    is_val = _vehicle_split(vehicles, seed, val_vehicle_frac)

    if verbose:
        print(f"[real-frames] {len(frames)} full photos across {len(set(vehicles))} vehicles "
              f"({sum(is_val)} val frames, {len(frames) - sum(is_val)} train frames)")

    out: dict[str, dict] = {}
    for split, mask in (("train", [not v for v in is_val]), ("val", is_val)):
        idx = [i for i, m in enumerate(mask) if m]
        if max_images:
            idx = idx[:max_images]
        stats = {}
        for name, eng in engines.items():
            t0 = time.time()
            ok = 0
            for i in idx:
                r = eng.read_frame(frames[i])
                ok += r.text == texts[i]
            stats[name] = {"plate_accuracy": (ok / len(idx)) if idx else 0.0, "n": len(idx),
                           "s_per_read": ((time.time() - t0) / len(idx)) if idx else 0.0}
        out[split] = stats
        if verbose:
            row = "  ".join(f"{n}: {stats[n]['plate_accuracy']:.0%} "
                            f"({stats[n]['s_per_read']:.2f}s/read)" for n in engines)
            print(f"{split:6} (n={len(idx):3})  {row}", flush=True)
    return out


def _vehicle_split(vehicles: list[str], seed: int, val_vehicle_frac: float = 0.15):
    """The exact split logic run_real()/run_real_frames() each reimplemented
    separately - centralised here after finding they'd silently drifted out
    of sync with each other and with the CLI (see run_real_multiseed())."""
    uniq_vehicles = sorted(set(vehicles))
    rng = random.Random(seed)
    rng.shuffle(uniq_vehicles)
    n_val = max(1, int(len(uniq_vehicles) * val_vehicle_frac))
    val_vehicles = set(uniq_vehicles[:n_val])
    return [v in val_vehicles for v in vehicles]


def run_real_multiseed(seeds: list[int], val_vehicle_frac: float = 0.15,
                       verbose: bool = True) -> dict:
    """run_real(), repeated across several vehicle-split seeds, reporting
    mean/min/max val accuracy per backend.

    Exists because a single seed on this dataset is not a stable measurement:
    only 83 unique vehicles means ~12 land in val, and which specific 12
    swings a backend's val accuracy by tens of points (found by accident -
    the CLI's --real mode was silently using --seed's synthetic-benchmark
    default of 4242 instead of run_real()'s own default of 11, and the two
    seeds gave anpr_ocr 69% vs ~50% on the identical model and code). Report
    a spread, not a single number, so that instability is visible rather than
    hidden behind whichever seed happened to be run last.
    """
    engines = _build_engines()
    crops, texts, vehicles = _load_real_dataset()
    per_seed: dict[int, dict[str, float]] = {}
    for seed in seeds:
        is_val = _vehicle_split(vehicles, seed, val_vehicle_frac)
        idx = [i for i, v in enumerate(is_val) if v]
        stats = {}
        for name, eng in engines.items():
            ok = sum(eng.read(crops[i]).text == texts[i] for i in idx)
            stats[name] = ok / len(idx) if idx else 0.0
        per_seed[seed] = stats
        if verbose:
            row = "  ".join(f"{n}: {stats[n]:.0%}" for n in engines)
            print(f"seed={seed:5} (n_val={len(idx):3})  {row}", flush=True)
    if verbose:
        print()
        for name in engines:
            vals = [per_seed[s][name] for s in seeds]
            print(f"{name:10} mean={sum(vals)/len(vals):.1%}  "
                  f"min={min(vals):.1%}  max={max(vals):.1%}  "
                  f"spread={max(vals)-min(vals):.1%}")
    return per_seed


def run_real_frames_multiseed(seeds: list[int], val_vehicle_frac: float = 0.15,
                              max_images: int = 0, verbose: bool = True) -> dict:
    """run_real_frames(), repeated across several vehicle-split seeds. See
    run_real_multiseed() for why this matters more than a single number."""
    from anpr.ocr import _load_yolo
    if verbose:
        yolo_loaded = _load_yolo() is not None
        print(f"[real-frames] detector: {'YOLOv8-OBB (models/yolov8n-obb-plate.pt)' if yolo_loaded else 'classical fallback - YOLO weights not found'}")
    engines = _build_engines()
    frames, texts, vehicles = _load_real_frames_dataset()
    per_seed: dict[int, dict[str, float]] = {}
    for seed in seeds:
        is_val = _vehicle_split(vehicles, seed, val_vehicle_frac)
        idx = [i for i, v in enumerate(is_val) if v]
        if max_images:
            idx = idx[:max_images]
        stats = {}
        for name, eng in engines.items():
            ok = sum(eng.read_frame(frames[i]).text == texts[i] for i in idx)
            stats[name] = ok / len(idx) if idx else 0.0
        per_seed[seed] = stats
        if verbose:
            row = "  ".join(f"{n}: {stats[n]:.0%}" for n in engines)
            print(f"seed={seed:5} (n_val={len(idx):3})  {row}", flush=True)
    if verbose:
        print()
        for name in engines:
            vals = [per_seed[s][name] for s in seeds]
            print(f"{name:10} mean={sum(vals)/len(vals):.1%}  "
                  f"min={min(vals):.1%}  max={max(vals):.1%}  "
                  f"spread={max(vals)-min(vals):.1%}")
    return per_seed


def run_real_all(verbose: bool = True) -> dict:
    """Per-frame exact-match over EVERY labelled real crop, with no split.

    The vehicle-split modes answer "how does this do on plates it has not been
    tuned against", which matters for a fine-tune. None of these backends is
    trained here, so for a zero-shot recogniser the split only adds noise: with
    ~118 vehicles, *which* ones land in val moves the number by more than any
    code change does. This mode removes that variable - every frame is scored,
    once - which makes it the right measurement for comparing two decoders, and
    the per-source rows show whether a gain is real or just one easy source.
    """
    engines = _build_engines()
    crops, texts, vehicles = _load_real_dataset()
    sources = ["xml", "car", "video"]

    out: dict[str, dict] = {}
    for key in sources + ["ALL"]:
        idx = [i for i, v in enumerate(vehicles)
               if key == "ALL" or v.split(":")[0] == key]
        if not idx:
            continue
        stats = {}
        for name, eng in engines.items():
            ok = sum(eng.read(crops[i]).text == texts[i] for i in idx)
            stats[name] = {"plate_accuracy": ok / len(idx), "n": len(idx)}
        out[key] = stats
        if verbose:
            row = "  ".join(f"{n}: {stats[n]['plate_accuracy']:.1%}" for n in engines)
            print(f"{key:6} (n={len(idx):4})  {row}", flush=True)
    return out


def run_real(val_vehicle_frac: float = 0.15, seed: int = 11, verbose: bool = True) -> dict:
    """Head-to-head on real photos, vehicle-split so no backend is graded on
    a near-duplicate frame of a vehicle it's also scored on elsewhere."""
    engines = _build_engines()
    crops, texts, vehicles = _load_real_dataset()

    is_val = _vehicle_split(vehicles, seed, val_vehicle_frac)

    if verbose:
        print(f"[real] {len(crops)} crops across {len(set(vehicles))} vehicles "
              f"({sum(is_val)} val frames, {len(crops) - sum(is_val)} train frames)")

    out: dict[str, dict] = {}
    for split, mask in (("train", [not v for v in is_val]), ("val", is_val)):
        idx = [i for i, m in enumerate(mask) if m]
        stats = {}
        for name, eng in engines.items():
            ok = 0
            for i in idx:
                r = eng.read(crops[i])
                ok += r.text == texts[i]
            stats[name] = {"plate_accuracy": (ok / len(idx)) if idx else 0.0, "n": len(idx)}
        out[split] = stats
        if verbose:
            row = "  ".join(f"{n}: {stats[n]['plate_accuracy']:.0%}" for n in engines)
            print(f"{split:6} (n={len(idx):3})  {row}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the GodsEye recognisers")
    ap.add_argument("--samples", type=int, default=100,
                    help="synthetic mode only (no --real/--real-frames)")
    ap.add_argument("--seed", type=int, default=4242,
                    help="synthetic mode only - --real/--real-frames use their own "
                         "default (11) unless --real-seed is given")
    ap.add_argument("--real-seed", type=int, default=None,
                    help="vehicle-split seed for --real/--real-frames (default 11 - "
                         "NOT --seed, which is a separate default for synthetic mode; "
                         "the two used to be silently conflated here)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="--real/--real-frames only: run this many consecutive vehicle-split "
                         "seeds (starting at --real-seed, default 11) and report mean/min/max - "
                         "a single seed on this small dataset is not a stable measurement")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--real", action="store_true",
                    help="compare on the real, vehicle-split plate crops instead of synthetic captures")
    ap.add_argument("--real-all", action="store_true",
                    help="per-frame exact-match over every labelled real crop, no "
                         "vehicle split - the stable way to compare two decoders")
    ap.add_argument("--real-frames", action="store_true",
                    help="compare on whole real photographs via read_frame() (exercises YOLOv8-OBB detection)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap per-split image count for --real-frames (0 = no cap)")
    args = ap.parse_args()
    real_seed = args.real_seed if args.real_seed is not None else 11
    if args.real_all:
        out = run_real_all()
    elif args.seeds > 1 and args.real_frames:
        out = run_real_frames_multiseed(list(range(real_seed, real_seed + args.seeds)),
                                        max_images=args.max_images)
    elif args.seeds > 1 and args.real:
        out = run_real_multiseed(list(range(real_seed, real_seed + args.seeds)))
    elif args.real_frames:
        out = run_real_frames(seed=real_seed, max_images=args.max_images)
    elif args.real:
        out = run_real(seed=real_seed)
    else:
        out = run(args.samples, args.seed)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
