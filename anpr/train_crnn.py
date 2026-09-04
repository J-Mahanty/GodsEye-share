"""Train the CRNN on the camera corpus.

    python -m anpr.train_crnn --samples 60000 --epochs 12

Data is generated, not collected: `plates.capture()` produces an image and its
ground-truth string, so the corpus is unbounded and perfectly labelled. There is
no segmentation step and therefore no alignment problem — CTC handles alignment
itself, which is exactly why the curriculum the classical model needed is not
needed here.

Runs on CPU. A 60k-sample run takes roughly 40-70 minutes on a laptop; the
checkpoint is written after every epoch so an interrupted run is still usable.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

import config
from anpr import crnn, plates


def make_batch(rng: random.Random, n: int):
    """n captures and their labels, drawn across every scenario and fault."""
    xs, texts = [], []
    for _ in range(n):
        cap = plates.capture(rng=rng)
        xs.append(crnn.prepare(cap.image))
        texts.append(cap.text)
    return np.stack(xs)[:, None], texts


TWO_ROW_TRAIN_FRAC = 0.32
"""Share of two-row plates in the *training* corpus.

On the road they are about 18% of traffic, and the benchmark keeps that. For
training they are oversampled, because a two-row plate is a visually distinct
sub-problem - it is unwrapped into a line before the reader sees it - and at 18%
the network learned it far more slowly than the single-row case it saw five
times as often.
"""


def _chunk(args):
    """One worker's share of the corpus. Runs in its own process."""
    count, seed = args
    rng = random.Random(seed)
    xs = np.zeros((count, crnn.IMG_H, crnn.IMG_W), np.float32)
    texts = []
    for i in range(count):
        cap = plates.capture(rng=rng, two_row=rng.random() < TWO_ROW_TRAIN_FRAC)
        xs[i] = crnn.prepare(cap.image)
        texts.append(cap.text)
    return xs, texts


def generate(n: int, seed: int, verbose: bool = True, workers: int | None = None):
    """Build a fixed dataset up front so epochs are comparable.

    A single capture costs about 40 ms - the camera model is doing real work -
    so generating tens of thousands of them serially would take longer than the
    training itself. Each worker owns a slice and its own seeded generator, so
    the corpus is reproducible regardless of how many cores run it.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 1))
    per = [n // workers] * workers
    for i in range(n - sum(per)):
        per[i] += 1
    t0 = time.time()
    X = np.zeros((n, 1, crnn.IMG_H, crnn.IMG_W), np.float32)
    labels: list[str] = []
    at = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for xs, texts in pool.map(_chunk, [(c, seed + 1000 * i)
                                           for i, c in enumerate(per)]):
            X[at:at + len(xs), 0] = xs
            labels.extend(texts)
            at += len(xs)
            if verbose:
                print(f"  generated {at}/{n} captures "
                      f"({at / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)
    return X, labels


def train(samples: int = 60000, epochs: int = 12, batch: int = 64, lr: float = 3e-3,
          seed: int = 5, val_frac: float = 0.04, out: Path | None = None,
          verbose: bool = True, resume: bool = False):
    """Train, or continue training on a fresh corpus.

    Resuming matters here because the corpus is generated: a second run draws
    entirely new captures, so warm-starting from the checkpoint is strictly more
    data rather than another pass over the same data. It is the cheapest way to
    buy accuracy once the loss curve flattens.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    out = Path(out or (config.MODEL_DIR / "crnn.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[crnn] generating {samples} camera captures ...")
    X, labels = generate(samples, seed, verbose)
    n_val = max(512, int(samples * val_frac))
    Xtr, ytr = X[:-n_val], labels[:-n_val]
    Xva, yva = X[-n_val:], labels[-n_val:]

    model = crnn.build_model()
    started_from = "scratch"
    if resume and out.exists():
        state = torch.load(out, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        started_from = f"{out.name} (val {state.get('val_accuracy', 0):.1%})"
    params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[crnn] {params/1e6:.2f}M parameters, {len(Xtr)} train / {len(Xva)} val, "
              f"starting from {started_from}")

    ctc = nn.CTCLoss(blank=crnn.BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // batch) * epochs
    peak = lr * (0.35 if resume else 1.0)      # a warm start does not want a hot restart
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=peak, total_steps=steps)

    # When resuming, the incumbent has to be beaten before it is overwritten.
    best = 0.0
    if resume and out.exists():
        try:
            best = float(torch.load(out, map_location="cpu").get("val_accuracy", 0.0))
        except Exception:
            best = 0.0
    order = np.arange(len(Xtr))
    for epoch in range(1, epochs + 1):
        model.train()
        np.random.default_rng(seed + epoch).shuffle(order)
        total, seen, t0 = 0.0, 0, time.time()
        for start in range(0, len(order) - batch + 1, batch):
            idx = order[start:start + batch]
            xb = torch.from_numpy(Xtr[idx])
            targets = [crnn.encode(ytr[i]) for i in idx]
            flat = torch.tensor([c for t in targets for c in t], dtype=torch.long)
            tgt_len = torch.tensor([len(t) for t in targets], dtype=torch.long)

            logits = model(xb)                                  # (B, T, C)
            logp = logits.log_softmax(2).permute(1, 0, 2)       # CTC wants (T, B, C)
            inp_len = torch.full((len(idx),), logits.shape[1], dtype=torch.long)
            loss = ctc(logp, flat, inp_len, tgt_len)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
            seen += len(idx)
            if verbose and seen % (batch * 100) == 0:
                print(f"    epoch {epoch}  {seen}/{len(order)}  loss {total/seen:.3f}",
                      flush=True)

        acc, greedy_acc = evaluate(model, Xva, yva, batch)
        if verbose:
            print(f"[crnn] epoch {epoch:2}  loss {total/max(seen,1):.3f}  "
                  f"val plate {acc:.1%} (greedy {greedy_acc:.1%})  "
                  f"{time.time()-t0:.0f}s", flush=True)
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "val_accuracy": acc,
                        "samples": samples, "epochs": epoch}, out)
            if verbose:
                print(f"[crnn] saved -> {out}", flush=True)
    return best


def evaluate(model, X, labels, batch: int = 64) -> tuple[float, float]:
    """Plate accuracy under the grammar-constrained decode and the greedy one."""
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
    return ok / len(X), greedy_ok / len(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the GodsEye CRNN plate reader")
    ap.add_argument("--samples", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the saved checkpoint on a fresh corpus")
    args = ap.parse_args()
    if not crnn.available():
        raise SystemExit("torch is not installed; the platform runs on the classical "
                         "engine without it (pip install torch --index-url "
                         "https://download.pytorch.org/whl/cpu)")
    best = train(args.samples, args.epochs, args.batch, args.lr, args.seed,
                 out=Path(args.out) if args.out else None, resume=args.resume)
    print(f"best validation plate accuracy {best:.1%}")


if __name__ == "__main__":
    main()
