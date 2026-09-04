"""Fine-tune the CRNN plate reader on real photographs.

    python finetune_crnn_real.py

`anpr/train_crnn.py` only ever trains on synthetic captures (`plates.capture()`);
it has no path for real images at all. This script adds one, for the only local
dataset that carries real plate-text ground truth:
`indian-number-plates-dataset/Annotations/*.xml`, VOC-style boxes with a
`number_plate_text` attribute. Of its 47 images / 52 annotated boxes, only 25
actually have that attribute filled in -- the rest are unusable for CTC
training (no label).

25 real images is far too few to train a CRNN from scratch, so this:
  1. starts from the existing synthetic-trained checkpoint (models/crnn.pt),
  2. holds out a handful of the 25 real crops for validation, oversamples the
     rest so they carry real weight against a batch,
  3. mixes them into a batch of fresh synthetic captures every epoch, so the
     synthetic-distribution accuracy the model already has isn't wiped out by
     25 examples,
  4. writes a SEPARATE checkpoint (models/crnn_real_finetune.pt) rather than
     overwriting models/crnn.pt. Compare the two (this script prints both real
     and synthetic-style validation) before promoting it.
"""
from __future__ import annotations

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

import config
from anpr import crnn, plates

REAL_DATASET_DIR = "indian-number-plates-dataset"


def load_real_crops(dataset_dir: str = REAL_DATASET_DIR):
    """(image crop, plate text) pairs from the VOC XML annotations."""
    ann_dir = os.path.join(dataset_dir, "Annotations")
    img_dir = os.path.join(dataset_dir, "images")
    crops, texts = [], []
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".xml"):
            continue
        root = ET.parse(os.path.join(ann_dir, fn)).getroot()
        image_path = os.path.join(img_dir, root.find("filename").text)
        img = cv2.imread(image_path)
        if img is None:
            continue
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            text = None
            for attr in obj.findall(".//attribute"):
                name_el = attr.find("name")
                if name_el is not None and name_el.text == "number_plate_text":
                    val_el = attr.find("value")
                    text = val_el.text if val_el is not None else None
            if not text:
                continue
            text = text.strip().upper().replace(" ", "")
            if not text or any(c not in crnn.IDX for c in text):
                continue
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))
            crop = img[max(0, ymin):ymax, max(0, xmin):xmax]
            if crop.size == 0:
                continue
            crops.append(crop)
            texts.append(text)
    return crops, texts


def prepared(crops) -> np.ndarray:
    return np.stack([crnn.prepare(c) for c in crops])[:, None]


def synthetic_batch(rng: random.Random, n: int):
    xs, texts = [], []
    for _ in range(n):
        cap = plates.capture(rng=rng)
        xs.append(crnn.prepare(cap.image))
        texts.append(cap.text)
    return np.stack(xs)[:, None], texts


def evaluate(model, X, labels, batch: int = 32) -> tuple[float, float]:
    import torch

    model.eval()
    ok = greedy_ok = 0
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = torch.from_numpy(X[start:start + batch])
            out = model(xb).cpu().numpy()
            for j in range(out.shape[0]):
                truth = labels[start + j]
                text, _, _ = crnn.constrained_decode(out[j])
                greedy, _ = crnn.greedy_decode(out[j])
                ok += text == truth
                greedy_ok += greedy == truth
    return ok / max(len(X), 1), greedy_ok / max(len(X), 1)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune the CRNN on real plate photos")
    ap.add_argument("--val-count", type=int, default=6,
                    help="how many of the 25 real examples to hold out for validation")
    ap.add_argument("--oversample", type=int, default=20,
                    help="how many times each real training crop is repeated per epoch")
    ap.add_argument("--synthetic-per-epoch", type=int, default=2000,
                    help="fresh synthetic captures mixed in each epoch")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base", type=str, default=str(config.MODEL_DIR / "crnn.pt"))
    ap.add_argument("--out", type=str, default=str(config.MODEL_DIR / "crnn_real_finetune.pt"))
    args = ap.parse_args()

    if not crnn.available():
        raise SystemExit("torch is not installed")

    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    crops, texts = load_real_crops()
    print(f"[real] {len(crops)} labelled real crops loaded")
    idx = list(range(len(crops)))
    rng.shuffle(idx)
    val_idx = idx[:args.val_count]
    train_idx = idx[args.val_count:]
    print(f"[real] {len(train_idx)} train / {len(val_idx)} val")

    Xreal_tr = prepared([crops[i] for i in train_idx])
    yreal_tr = [texts[i] for i in train_idx]
    Xreal_va = prepared([crops[i] for i in val_idx])
    yreal_va = [texts[i] for i in val_idx]

    model = crnn.build_model()
    state = torch.load(args.base, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    print(f"[crnn] starting from {args.base}")

    pre_real_acc, pre_real_greedy = evaluate(model, Xreal_va, yreal_va, args.batch)
    print(f"[before] real-val plate accuracy {pre_real_acc:.1%} (greedy {pre_real_greedy:.1%})")

    ctc = nn.CTCLoss(blank=crnn.BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, (len(train_idx) * args.oversample + args.synthetic_per_epoch) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs)

    best = pre_real_acc
    torch.save({"model": model.state_dict(), "val_accuracy": best}, args.out)

    for epoch in range(1, args.epochs + 1):
        Xsyn, ysyn = synthetic_batch(rng, args.synthetic_per_epoch)
        Xreal_rep = np.repeat(Xreal_tr, args.oversample, axis=0)
        yreal_rep = yreal_tr * args.oversample

        X = np.concatenate([Xsyn, Xreal_rep], axis=0)
        y = ysyn + yreal_rep
        order = np.arange(len(X))
        np.random.default_rng(args.seed + epoch).shuffle(order)

        model.train()
        total, seen = 0.0, 0
        for start in range(0, len(order) - args.batch + 1, args.batch):
            bidx = order[start:start + args.batch]
            xb = torch.from_numpy(X[bidx])
            targets = [crnn.encode(y[i]) for i in bidx]
            flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
            tgt_len = torch.tensor([len(t) for t in targets], dtype=torch.long)

            logits = model(xb)
            logp = logits.log_softmax(2).permute(1, 0, 2)
            inp_len = torch.full((len(bidx),), logits.shape[1], dtype=torch.long)
            loss = ctc(logp, flat, inp_len, tgt_len)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss) * len(bidx)
            seen += len(bidx)

        real_acc, real_greedy = evaluate(model, Xreal_va, yreal_va, args.batch)
        print(f"[crnn] epoch {epoch:2}  loss {total/max(seen,1):.3f}  "
              f"real-val {real_acc:.1%} (greedy {real_greedy:.1%})", flush=True)
        if real_acc >= best:
            best = real_acc
            torch.save({"model": model.state_dict(), "val_accuracy": best}, args.out)
            print(f"[crnn] saved -> {args.out}", flush=True)

    print(f"\nbest real-val plate accuracy: {best:.1%} (started at {pre_real_acc:.1%})")


if __name__ == "__main__":
    main()
