"""Build the full real (crop, text) training set from every labelled source.

    python build_real_crnn_dataset.py

Sources:
  1. indian-number-plates-dataset/Annotations/*.xml -- 25 crops with real
     ground-truth text (see finetune_crnn_real.load_real_crops).
  2. dataset_v1_bbox/*/images/car-*  -- 13 crops whose filename embeds the
     plate (verified against one image visually: car-ybs-MH43BP8173... ->
     the plate in the photo really is MH43BP8173).
  3. dataset_v1_bbox/*/images/video*  -- 548 dashcam frames with NO text
     label, transcribed by hand into real_plate_frame_labels.json: 494
     labelled frames covering 84 distinct plates, 8 left out as unreadable.
     This replaced the earlier cluster-propagated labelling
     (cluster_manifest.json + real_plate_transcriptions.json, one hand-typed
     plate per cluster of adjacent frames). Adjacency is not vehicle identity:
     cluster 6 held six different vehicles under one label, cluster 39 held
     eight, and 37% of all frames carried a plate that was not the one in the
     picture. Those two files are kept for provenance; frame_labels() is the
     ground truth.

Splits by *vehicle identity* (plate text), not by image -- otherwise two
frames of the same crossing could land on both sides and inflate validation
the same way the YOLO OBB split did.

Writes real_crnn_dataset.npz: X (N,1,32,160) float32, y (N,) plate strings,
vehicle (N,) grouping key, split ('train'/'val').
"""
from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np

from anpr import crnn
from finetune_crnn_real import load_real_crops as load_xml_crops, prepared

CAR_TEXTS = {
    "car-wbs-MH01DE2780": "MH01DE2780",
    "car-wbs-MH03AR5549": "MH03AR5549",
    "car-wbs-MH12FU1014": "MH12FU1014",
    "car-wbs-MH43AF5037": "MH43AF5037",
    "car-wbs-MH43BU2401": "MH43BU2401",
    "car-wbs-MH46BV0688": "MH46BV0688",
    "car-ybs-MH43BP8173": "MH43BP8173",
    "car-ybs-MH46AD5258": "MH46AD5258",
    "car-ybs-MH46BF2342": "MH46BF2342",
    "car-ybs-MH47N4570": "MH47N4570",
}


FRAME_LABELS_PATH = "real_plate_frame_labels.json"


def frame_labels() -> dict[str, str]:
    """{filename: plate} for the dashcam frames, or {} if the file is absent.

    See the module docstring: this supersedes the cluster-propagated labels in
    real_plate_transcriptions.json, which mislabelled 37% of frames because a
    cluster is a run of adjacent frames, not one vehicle.
    """
    if not os.path.exists(FRAME_LABELS_PATH):
        return {}
    with open(FRAME_LABELS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def find_split_dir(fn: str) -> str | None:
    for split in ("train", "valid", "test"):
        if os.path.exists(f"dataset_v1_bbox/{split}/images/{fn}"):
            return split
    return None


def crop_bbox_image(fn: str) -> np.ndarray | None:
    split = find_split_dir(fn)
    if split is None:
        return None
    stem = os.path.splitext(fn)[0]
    img = cv2.imread(f"dataset_v1_bbox/{split}/images/{fn}")
    if img is None:
        return None
    H, W = img.shape[:2]
    with open(f"dataset_v1_bbox/{split}/labels/{stem}.txt") as f:
        line = f.readline().split()
    if len(line) != 5:
        return None
    _, cx, cy, w, h = map(float, line)
    x1 = max(0, int((cx - w / 2) * W)); y1 = max(0, int((cy - h / 2) * H))
    x2 = min(W, int((cx + w / 2) * W)); y2 = min(H, int((cy + h / 2) * H))
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else None


def load_all(val_vehicle_frac: float = 0.15, seed: int = 11):
    crops, texts, vehicles = [], [], []

    # 1. XML-labelled real plates.
    xml_crops, xml_texts = load_xml_crops()
    for c, t in zip(xml_crops, xml_texts):
        crops.append(c)
        texts.append(t)
        vehicles.append(f"xml:{t}")

    # 2. filename-embedded plates.
    for prefix, text in CAR_TEXTS.items():
        matches = [f for f in os.listdir("dataset_v1_bbox/train/images")
                   if f.startswith(prefix)]
        for fn in matches:
            crop = crop_bbox_image(fn)
            if crop is not None:
                crops.append(crop)
                texts.append(text)
                vehicles.append(f"car:{text}")

    # 3. dashcam video frames, one label per frame.
    for fn, text in sorted(frame_labels().items()):
        crop = crop_bbox_image(fn)
        if crop is not None:
            crops.append(crop)
            texts.append(text)
            vehicles.append(f"video:{text}")

    print(f"[data] {len(crops)} total real crops across "
          f"{len(set(vehicles))} vehicle instances")

    # Split by vehicle, not by frame.
    rng = random.Random(seed)
    uniq_vehicles = sorted(set(vehicles))
    rng.shuffle(uniq_vehicles)
    n_val_vehicles = max(1, int(len(uniq_vehicles) * val_vehicle_frac))
    val_vehicles = set(uniq_vehicles[:n_val_vehicles])

    X = prepared(crops)
    y = np.array(texts)
    is_val = np.array([v in val_vehicles for v in vehicles])

    print(f"[data] {len(uniq_vehicles)} unique vehicles -> "
          f"{n_val_vehicles} held out for val ({is_val.sum()} frames), "
          f"{(~is_val).sum()} train frames")

    np.savez("real_crnn_dataset.npz", X=X, y=y, vehicle=np.array(vehicles),
              is_val=is_val)
    print("wrote real_crnn_dataset.npz")


if __name__ == "__main__":
    load_all()
