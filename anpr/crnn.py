"""CRNN + CTC plate reader — the recogniser that does not segment.

Why replace the classical engine
--------------------------------
The segment-and-classify pipeline in `ocr.py` reached 87% on the old
filter-based corpus and 13% on camera-realistic captures. The bottleneck was
measured, not guessed: perspective rectification, decoder merge depth, atom caps
and morphology kernels all moved it by a point or less, while held-out glyph
accuracy sat at 97% on a training mix dominated by easy crops. Ten characters at
~91% each is 40% of plates, which is exactly what daylight scored.

The structural problem is that **segmentation is a decision made too early**.
Once a gantry capture has been through JPEG at 90 px wide and a 3x digital zoom,
the gaps between characters are not reliably there to find. Any pipeline that
must commit to character boundaries before recognising anything is betting on
information the image no longer contains.

A CRNN never makes that decision. It slides a convolutional stack across the
plate, produces a sequence of per-column class distributions, and CTC marginalises
over *every* alignment between those columns and the output string. Character
boundaries are latent, not chosen.

What it keeps from the old engine
---------------------------------
The plate grammar, which was the best idea in it. CTC's own greedy decode is
unconstrained, so this uses a prefix beam search restricted to the Bharat
registration shapes: a digit can still never win a letter slot. That is worth
several points on degraded plates and costs microseconds.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config

# CTC needs a blank symbol; index 0 is reserved for it.
BLANK = 0
ALPHABET = config.ALPHABET                      # 36 characters
IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
REV = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1

IMG_H = 32                                      # every plate is resampled to this height
IMG_W = 160                                     # ... and this width
MIN_TIMESTEPS = 40                              # width // 4 after the conv stack


def available() -> bool:
    """Is torch installed? The platform runs without it, on the classical engine."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --- preprocessing ------------------------------------------------------
def text_bands(gray: np.ndarray, min_frac: float = 0.06) -> list[tuple[int, int]]:
    """Rows of the crop that contain characters, as (start, end) bands.

    Counting bands is how you tell a two-row plate from a one-row plate. The
    canvas aspect ratio cannot: a camera crop carries margin and perspective, so
    a perfectly ordinary single-row capture arrives at 2.4:1 and would be cut in
    half by an aspect test.
    """
    h, w = gray.shape[:2]
    if h < 8:
        return [(0, h)]
    ink = 255.0 - cv2.GaussianBlur(gray, (0, 0), 1.5).mean(axis=1)
    ink -= ink.min()
    if ink.max() < 1e-6:
        return [(0, h)]
    ink /= ink.max()
    on = ink > 0.45
    bands, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if y - start >= max(3, int(h * min_frac)):
                bands.append((start, y))
            start = None
    if start is not None and h - start >= max(3, int(h * min_frac)):
        bands.append((start, h))
    return bands or [(0, h)]


def _trim_columns(row: np.ndarray) -> np.ndarray:
    """Trim a row to the columns that actually carry characters.

    The blue IND band runs the full height of a High Security plate, so both
    halves of a split inherit it - joining them puts a dark bar in the middle of
    the line, where the reader expects a character. Trimming each half to its
    own ink extent removes the band from the second half and tightens both.
    """
    h, w = row.shape[:2]
    if w < 12:
        return row
    ink = 255.0 - cv2.GaussianBlur(row, (0, 0), 1.0).mean(axis=0)
    ink -= ink.min()
    if ink.max() < 1e-6:
        return row
    on = np.where(ink / ink.max() > 0.35)[0]
    if on.size < 4:
        return row
    pad = max(1, int(0.02 * w))
    return row[:, max(0, int(on[0]) - pad):min(w, int(on[-1]) + pad + 1)]


def unwrap_two_row(gray: np.ndarray) -> np.ndarray:
    """A two-row plate, laid out as one line for a left-to-right reader.

    Motorcycles, autos and trucks in India carry two-row plates, and squashing
    one into a 32-pixel-high strip leaves each row about fourteen pixels tall -
    unreadable, and measured at 2% plate accuracy before this existed. The two
    character bands are placed side by side, which is the layout the recogniser
    was built for.
    """
    h, w = gray.shape[:2]
    bands = text_bands(gray)
    if len(bands) != 2:
        return gray
    (t0, t1), (b0, b1) = bands
    # both bands should be a similar height, and neither trivially thin
    ht, hb = t1 - t0, b1 - b0
    if min(ht, hb) < 0.30 * max(ht, hb) or max(ht, hb) > 0.6 * h:
        return gray
    pad = max(1, int(0.12 * max(ht, hb)))
    top = _trim_columns(gray[max(0, t0 - pad):min(h, t1 + pad)])
    bottom = _trim_columns(gray[max(0, b0 - pad):min(h, b1 + pad)])
    if min(top.shape[0], bottom.shape[0]) < 4:
        return gray
    height = max(top.shape[0], bottom.shape[0])
    top = cv2.resize(top, (max(8, int(top.shape[1] * height / top.shape[0])), height))
    bottom = cv2.resize(bottom, (max(8, int(bottom.shape[1] * height / bottom.shape[0])), height))
    return np.hstack([top, bottom])


def forced_row_split(gray: np.ndarray) -> np.ndarray | None:
    """Split a crop at the clearest horizontal gap and lay the halves side by side.

    `unwrap_two_row` only fires when the ink profile resolves into exactly two
    bands. On a genuine two-row plate that test fails about three times in ten -
    glare bridges the rows, or grime breaks one of them into three - and the
    plate is then squashed into a 32-pixel strip and read at 3% accuracy. This
    is the hypothesis of last resort: assume two rows, cut at the quietest row
    in the middle third, and let the decoder decide whether that was right.
    """
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return None
    ink = 255.0 - cv2.GaussianBlur(gray, (0, 0), 1.5).mean(axis=1)
    ink -= ink.min()
    if ink.max() < 1e-6:
        return None
    ink /= ink.max()
    lo, hi = int(h * 0.33), int(h * 0.67)
    if hi - lo < 2:
        return None
    cut = lo + int(np.argmin(ink[lo:hi]))
    top, bottom = gray[:cut], gray[cut:]
    if min(top.shape[0], bottom.shape[0]) < 6:
        return None
    top, bottom = _trim_columns(top), _trim_columns(bottom)
    if min(top.shape[1], bottom.shape[1]) < 8:
        return None
    height = max(top.shape[0], bottom.shape[0])
    top = cv2.resize(top, (max(8, int(top.shape[1] * height / top.shape[0])), height))
    bottom = cv2.resize(bottom, (max(8, int(bottom.shape[1] * height / bottom.shape[0])), height))
    return np.hstack([top, bottom])


def row_hypotheses(gray: np.ndarray) -> list[np.ndarray]:
    """The layouts this crop might be, cheapest interpretation first.

    The one-row/two-row call used to be made once, by a threshold, before the
    recogniser saw anything - and it was wrong both ways. It skipped 30% of real
    two-row plates (which then read at 3.3%) and misfired on 6.7% of single-row
    ones, and *every* misfire read at 0%: splitting a single line in half puts
    the second half of the registration in front of the first.

    A threshold cannot be made reliable here, so the decision is deferred. Each
    layout is decoded and the exact CTC score picks the winner, which is the
    same principle that made the recogniser itself work: do not commit to a
    segmentation before you have tried to read it.
    """
    hyps = [gray]
    seen = {gray.shape}
    for cand in (unwrap_two_row(gray), forced_row_split(gray)):
        if cand is not None and cand.shape not in seen and cand.size:
            hyps.append(cand)
            seen.add(cand.shape)
    return hyps


def prepare(gray: np.ndarray, unwrap: bool = True) -> np.ndarray:
    """One plate crop -> the fixed-size, contrast-normalised tensor input.

    Deliberately minimal: no binarisation, no deskew, no segmentation. Those
    steps each throw away information on the assumption that what remains is
    enough, and on a 90 px JPEG plate that assumption is what fails. The network
    is given the greyscale and left to decide what matters. The one structural
    step is unwrapping a two-row plate, because no amount of training makes a
    single-line reader see around a line break.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if unwrap:
        gray = unwrap_two_row(gray)
    h, w = gray.shape[:2]
    if h < 4 or w < 8:
        gray = cv2.resize(gray, (max(w, 16), max(h, 8)))
    # CLAHE recovers local contrast on fogged and under-exposed captures without
    # committing to a threshold.
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    resized = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return (x - x.mean()) / (x.std() + 1e-5)


def encode(text: str) -> list[int]:
    return [IDX[c] for c in text if c in IDX]


# --- the network --------------------------------------------------------
def build_model():
    """A compact CRNN sized for CPU training.

    Four convolutional blocks downsample height 32 -> 1 while keeping width
    resolution at W/4, giving 40 timesteps for a 10-11 character plate: enough
    for CTC to place a blank between every pair of characters. A single
    bidirectional GRU carries context along the plate, which is what lets the
    network use the fact that character four is a digit when character four is
    smeared.
    """
    import torch
    from torch import nn

    class CRNN(nn.Module):
        def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 128):
            super().__init__()
            def block(cin, cout, pool):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                    nn.MaxPool2d(pool))
            self.cnn = nn.Sequential(
                block(1, 32, (2, 2)),      # 32x160 -> 16x80
                block(32, 64, (2, 2)),     # -> 8x40
                block(64, 128, (2, 1)),    # -> 4x40   (keep width resolution)
                block(128, 128, (4, 1)),   # -> 1x40
            )
            self.rnn = nn.GRU(128, hidden, num_layers=2, bidirectional=True,
                              batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden * 2, num_classes)

        def forward(self, x):                       # x: (B, 1, 32, 160)
            f = self.cnn(x)                         # (B, 128, 1, 40)
            f = f.squeeze(2).permute(0, 2, 1)       # (B, 40, 128)
            f, _ = self.rnn(f)
            return self.head(f)                     # (B, 40, classes)

    return CRNN()


# --- grammar-constrained CTC decoding -----------------------------------
def _patterns() -> list[str]:
    from anpr.ocr import PATTERNS
    return PATTERNS


LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")


def matches_grammar(text: str) -> str:
    """The registration shape this string fits, or "" if it fits none."""
    if not text:
        return ""
    for pattern in _patterns():
        if len(pattern) != len(text):
            continue
        if all((c.isalpha() if slot == "L" else c.isdigit())
               for slot, c in zip(pattern, text)):
            return pattern
    return ""


def greedy_decode(logits: np.ndarray) -> tuple[str, float]:
    """Standard CTC collapse: argmax per timestep, drop repeats and blanks."""
    probs = _softmax(logits)
    best = probs.argmax(axis=1)
    out, confs, prev = [], [], -1
    for t, k in enumerate(best):
        if k != prev and k != BLANK:
            out.append(REV[int(k)])
            confs.append(float(probs[t, k]))
        prev = int(k)
    text = "".join(out)
    return text, float(np.min(confs)) if confs else 0.0


def constrained_decode(logits: np.ndarray, beam: int = 16) -> tuple[str, float, str]:
    """CTC prefix beam search restricted to the plate grammar.

    Proper prefix beam search, which means tracking two probabilities per
    prefix: the mass of paths ending in a blank and the mass ending in the
    prefix's own last character. Collapsing them - as a naive beam does -
    mishandles repeated characters and double-counts merging paths. Measured
    against plain greedy decoding, the naive version *lost* three points, which
    is the tell: a grammar can only ever remove illegal strings, so if it is not
    winning, the search is wrong.

    One pass per registration shape; the best complete string wins.
    """
    probs = _softmax(logits)
    T = probs.shape[0]
    best_overall = ("", -1e9, "")

    for pattern in _patterns():
        K = len(pattern)
        if K > T:
            continue
        allowed = [np.array([IDX[c] for c in (LETTERS if slot == "L" else DIGITS)])
                   for slot in pattern]
        # prefix -> (log p_blank, log p_nonblank)
        beams: dict[str, list[float]] = {"": [0.0, NEG_INF]}
        for t in range(T):
            p = np.log(np.maximum(probs[t], 1e-12))
            nxt: dict[str, list[float]] = {}

            def add(prefix: str, pb: float, pnb: float) -> None:
                cur = nxt.setdefault(prefix, [NEG_INF, NEG_INF])
                cur[0] = _logaddexp(cur[0], pb)
                cur[1] = _logaddexp(cur[1], pnb)

            for prefix, (pb, pnb) in beams.items():
                k = len(prefix)
                total = _logaddexp(pb, pnb)
                # a blank leaves the prefix unchanged
                add(prefix, total + float(p[BLANK]), NEG_INF)
                # so does repeating the character just emitted
                if prefix:
                    add(prefix, NEG_INF, pnb + float(p[IDX[prefix[-1]]]))
                if k >= K or T - t < K - k:
                    continue
                # extend by one character that is legal in this slot
                idx = allowed[k]
                for j in idx[np.argsort(p[idx])[-5:]]:
                    ch = REV[int(j)]
                    # a genuine repeat needs a blank between the two, so it may
                    # only extend the blank-ending mass
                    src = pb if (prefix and ch == prefix[-1]) else total
                    add(prefix + ch, NEG_INF, src + float(p[j]))

            beams = {pre: v for pre, v in
                     sorted(nxt.items(), key=lambda kv: -_logaddexp(*kv[1]))[:beam]}

        done = [(_logaddexp(*v), pre) for pre, v in beams.items() if len(pre) == K]
        if not done:
            continue
        lp, text = max(done)
        score = lp / K                                   # per-character, comparable
        if score > best_overall[1]:
            best_overall = (text, score, pattern)
    if not best_overall[0]:
        text, _ = greedy_decode(logits)
        return text, -9.0, ""
    return best_overall


NEG_INF = -1e30


def _logaddexp(a: float, b: float) -> float:
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _push(table: dict, key, value: float) -> None:
    if key not in table or value > table[key]:
        table[key] = value


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ctc_score(logits: np.ndarray, text: str, idx: dict | None = None,
              blank: int = BLANK, is_probs: bool = False) -> tuple[float, list[float]]:
    """Exact CTC score of `text`, with a per-character posterior.

    `idx`/`blank`/`is_probs` exist so a *different* CTC model can borrow this
    recursion instead of duplicating it. anpr/anpr_ocr_reader.py passes
    PaddleOCR's own 64-class charset, and passes `is_probs=True` because
    ppocr's CTCHead already softmaxes its output at inference. Defaults keep
    the CRNN's own alphabet and raw-logit behaviour, so existing callers are
    unaffected.

    The forward-backward recursions marginalise over *every* alignment between
    the network's output columns and the string - the quantity CTC was trained
    to maximise. Two numbers come out: the sequence log-probability, and the
    peak posterior of each character's own state, which is what the state-code
    repair needs in order to know which character to doubt.

    This replaced a heuristic that walked the argmax path looking for each
    character's highest posterior in turn, advancing a cursor past each peak it
    found. When that scan mis-advanced - which it did whenever a character
    peaked later than the one after it - every subsequent character was searched
    in a window that no longer contained it, and the minimum came back near zero
    *on correct reads*. Measured over 700 captures it ranked correct reads below
    wrong ones: AUROC 0.37, worse than a coin toss, with a median confidence of
    0.000 on correct reads against 0.070 on wrong ones. The exact computation
    scores 0.98 on the same reads, which is what makes the confidence floor mean
    something: at a floor holding 90% stored accuracy it keeps 45% of reads
    where the heuristic kept 13.6%.
    """
    if not text:
        return NEG_INF, []
    idx = IDX if idx is None else idx
    probs = logits if is_probs else _softmax(logits)
    lp = np.log(np.maximum(probs, 1e-12))
    T = lp.shape[0]
    seq = np.empty(2 * len(text) + 1, dtype=np.int64)
    seq[0::2] = blank
    try:
        seq[1::2] = [idx[c] for c in text]
    except KeyError:
        return NEG_INF, [0.0] * len(text)
    S = seq.size
    if T < S - (S // 2):                 # not enough columns to emit the string
        return NEG_INF, [0.0] * len(text)

    em = lp[:, seq]                      # (T, S) emission of each extended state
    # a state may be reached from s-2 only if it is a real character and not a
    # repeat of the one two back - the blank between a doubled letter is required
    skip = np.zeros(S, bool)
    skip[2:] = (seq[2:] != blank) & (seq[2:] != seq[:-2])

    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = em[0, 0]
    if S > 1:
        alpha[0, 1] = em[0, 1]
    for t in range(1, T):
        prev = alpha[t - 1]
        a = prev.copy()
        a[1:] = np.logaddexp(a[1:], prev[:-1])
        a[2:] = np.where(skip[2:], np.logaddexp(a[2:], prev[:-2]), a[2:])
        alpha[t] = a + em[t]

    beta = np.full((T, S), -np.inf)
    beta[T - 1, S - 1] = em[T - 1, S - 1]
    if S > 1:
        beta[T - 1, S - 2] = em[T - 1, S - 2]
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1]
        b = nxt.copy()
        b[:-1] = np.logaddexp(b[:-1], nxt[1:])
        b[:-2] = np.where(skip[2:], np.logaddexp(b[:-2], nxt[2:]), b[:-2])
        beta[t] = b + em[t]

    total = np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2]) if S > 1 else alpha[T - 1, 0]
    if not np.isfinite(total):
        return NEG_INF, [0.0] * len(text)
    # occupancy posterior of each state at each column; a character's confidence
    # is the highest that its own state ever reaches
    gamma = alpha + beta - em - total
    peaks = np.exp(np.clip(gamma[:, 1::2].max(axis=0), -60.0, 0.0))
    return float(total), [float(min(max(p, 0.0), 1.0)) for p in peaks]


def sequence_confidence(logits: np.ndarray, text: str, **kw) -> float:
    """The CTC score of `text`, normalised per character so lengths compare."""
    if not text:
        return 0.0
    total, _ = ctc_score(logits, text, **kw)
    if total <= NEG_INF / 2:
        return 0.0
    return float(np.exp(np.clip(total / len(text), -60.0, 0.0)))


def character_confidences(logits: np.ndarray, text: str) -> list[float]:
    """Per-character posteriors under the exact CTC alignment marginal."""
    return ctc_score(logits, text)[1]


# --- inference wrapper --------------------------------------------------
@dataclass
class CRNNReader:
    model: object
    device: str = "cpu"
    beam: int = 12

    def logits(self, gray: np.ndarray, unwrap: bool = True) -> np.ndarray:
        import torch

        x = prepare(gray, unwrap)[None, None]
        with torch.no_grad():
            out = self.model(torch.from_numpy(x).to(self.device))
        return out[0].cpu().numpy()

    def logits_batch(self, grays: list[np.ndarray], unwrap: bool = True) -> np.ndarray:
        import torch

        x = np.stack([prepare(g, unwrap) for g in grays])[:, None]
        with torch.no_grad():
            out = self.model(torch.from_numpy(x).to(self.device))
        return out.cpu().numpy()

    def _decode(self, lg: np.ndarray):
        """One set of logits -> (text, char confidences, pattern, greedy, score)."""
        greedy, _ = greedy_decode(lg)
        text, pattern = greedy, matches_grammar(greedy)
        if not pattern:
            # Greedy produced something that is not a registration, so spend the
            # beam. Measured on 400 captures: greedy alone 37.2% at 0.1 ms, the
            # beam always-on 38.0% at 53 ms. The network has learned the plate
            # structure from a hundred thousand examples, so the explicit
            # grammar now only earns its cost on the minority of reads where the
            # network disagrees with itself.
            text, _, pattern = constrained_decode(lg, self.beam)
        if not text:
            return None
        total, confs = ctc_score(lg, text)
        if total <= NEG_INF / 2:
            return None
        score = float(np.exp(np.clip(total / len(text), -60.0, 0.0)))
        return text, confs, pattern, greedy, score

    def read(self, gray: np.ndarray):
        """-> (text, char confidences, pattern, greedy text, confidence)

        Every plausible row layout is decoded and the exact CTC score picks the
        winner. The layout used to be decided by a threshold on the ink profile
        before the recogniser ran, and that threshold was wrong both ways: it
        skipped 30% of genuine two-row plates, which then read at 3.3%, and it
        misfired on 6.7% of single-row plates, every one of which read at 0%
        because the second half of the registration ended up in front of the
        first. A grammatical read is preferred over an ungrammatical one, and
        the CTC score separates the rest.

        The hypotheses go through the network in one batch, so the cost is one
        forward pass over two or three stacked crops rather than two or three
        passes.
        """
        return self.read_with_logits(gray)[0]

    def read_with_logits(self, gray: np.ndarray):
        """read(), plus the CTC lattices it decided from.

        The lattices are what the network-constrained repair pass needs: they
        still hold the plate even when the decode did not find it, so a
        candidate string from a neighbouring camera can be scored against the
        original evidence rather than against the string that came out of it.
        """
        hyps = row_hypotheses(gray)
        lg = self.logits_batch(hyps, unwrap=False)
        lattices = [lg[i] for i in range(len(hyps))]
        best = None
        for i in range(len(hyps)):
            cand = self._decode(lg[i])
            if cand is None:
                continue
            # legality first, then the score: an illegal string is not a
            # registration whatever the network thinks of it
            rank = (bool(cand[2]), cand[4])
            if best is None or rank > best[0]:
                best = (rank, cand)
        if best is None:
            return ("", [], "", "", 0.0), lattices
        return best[1], lattices


def load(path=None, device: str = "cpu") -> CRNNReader | None:
    """Load the trained CRNN, or None if torch or the weights are missing."""
    import pathlib

    if not available():
        return None
    path = pathlib.Path(path or (config.MODEL_DIR / "crnn.pt"))
    if not path.exists():
        return None
    import torch

    model = build_model()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return CRNNReader(model=model, device=device)
