"""The GodsEye ANPR engine.

Pipeline
--------
    plate crop
      -> normalise (deskew, CLAHE, fixed height)      segment.normalize
      -> flatten illumination + Otsu -> ink mask       segment.binarize
      -> over-segment into atoms                       segment.segment
      -> format-constrained decode  <-- this module
      -> plate string + per-character confidence

The decoder is what lifts accuracy on bad captures. Classical ANPR pipelines
commit to one segmentation and then classify; a smeared "KA" that fused into one
blob, or a "0" that broke into two fragments, is then unrecoverable. Here the
segmenter is deliberately allowed to over-cut, and a dynamic program searches
every way of grouping 1-3 adjacent atoms into characters, scoring each grouping
against the Bharat-series plate grammar (LL DD LLL DDDD). The grammar rules out
whole families of classifier errors - a digit can never win a letter slot - and
the DP recovers merges and splits that a fixed segmentation cannot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from anpr import crnn as crnn_mod
from anpr import model as glyph_model
from anpr import segment

def union_box(atoms, i, j):
    """Bounding box of atoms[i:j] — one character may span several atoms."""
    x = min(atoms[k][0] for k in range(i, j))
    y = min(atoms[k][1] for k in range(i, j))
    x2 = max(atoms[k][0] + atoms[k][2] for k in range(i, j))
    y2 = max(atoms[k][1] + atoms[k][3] for k in range(i, j))
    return (x, y, x2 - x, y2 - y)


LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

# Valid Bharat-series shapes: 2 state letters, 1-2 district digits,
# 0-3 series letters, 4 registration digits.
PATTERNS: list[str] = [
    "LL" + "D" * d + "L" * s + "DDDD"
    for d in (1, 2)
    for s in (3, 2, 1, 0)
]

# Real Indian state / UT codes, used to repair the first two characters.
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB", "BH",
}


def plate_shape(text: str) -> str:
    """The grammar class of a string: "MH02DW8021" -> "LLDDLLDDDD"."""
    return "".join("L" if c in LETTERS else "D" if c in DIGITS else "?" for c in text)


_PATTERN_SET = frozenset(PATTERNS)


def matches_plate_grammar(text: str) -> bool:
    """Is `text` a well-formed registration with a real state code?

    Measured against every ground-truth label in the repo's real-photo corpus:
    76/76 carry a valid state code, and 74/76 match one of PATTERNS - the two
    exceptions ("KL34F", "KL34A465") are truncated transcriptions, not plate
    formats. Tight enough to steer a decoder with; callers must still fall back
    to an ungrammatical read rather than returning nothing, or those two - and
    any format this list has not met yet - become unreadable instead of merely
    wrong.
    """
    return bool(text) and text[:2] in STATE_CODES and plate_shape(text) in _PATTERN_SET


def repair_state_code(text: str) -> tuple[str, bool]:
    """Snap the first two characters onto a real state code when they are not
    one and the fix is a single confusable substitution.

    Module-level twin of ANPREngine._repair_state, so a backend that decodes
    without an engine instance (anpr/anpr_ocr_reader.py) can use it too. With
    39 codes in the lexicon "W8" is not ambiguous - it is WB.
    """
    if len(text) < 4 or text[:2] in STATE_CODES:
        return text, False
    best, best_cost = None, 3
    for code in STATE_CODES:
        cost = sum(a != b for a, b in zip(code, text[:2]))
        if cost < best_cost and all(
            a == b or _confusable(a, b) for a, b in zip(code, text[:2])
        ):
            best, best_cost = code, cost
    if best is None:
        return text, False
    return best + text[2:], True


# Each backend's confidence means a different thing, so each gets its own
# storage floor. Anything not listed falls back to the classical floor.
_BACKEND_FLOORS = {
    "crnn": "MIN_PLATE_CONFIDENCE_CRNN",
    "anpr_ocr": "MIN_PLATE_CONFIDENCE_ANPR_OCR",
}


@dataclass
class PlateRead:
    text: str                       # decoded plate, no spaces ("" if unreadable)
    confidence: float               # 0-1; what the floor is applied to. CRNN: the
                                    # exact CTC probability of the string, per
                                    # character. Classical: weakest character x
                                    # agreement between binarisations.
    char_confidence: float = 0.0    # the weakest character alone
    agreement: float = 0.0          # classical: share of binarisations that agreed;
                                    # burst reads: share of frames that agreed
    mean_confidence: float = 0.0    # geometric mean over the characters
    char_confidences: list[float] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    pattern: str = ""
    raw: str = ""                   # greedy per-atom read, before the decoder
    repaired: bool = False          # True if the state code was corrected
    plate_found: bool = True
    variant: str = ""               # which binarisation hypothesis won ("crnn" for the network)
    backend: str = "classical"      # which recogniser produced this read

    @property
    def floor(self) -> float:
        """The confidence floor this read is judged against."""
        name = _BACKEND_FLOORS.get(self.backend)
        return getattr(config, name) if name else config.MIN_PLATE_CONFIDENCE

    @property
    def accepted(self) -> bool:
        """Would the platform store this read? Ask the read, not a constant:
        the two backends' confidences are calibrated separately."""
        return bool(self.text) and self.confidence >= self.floor
    candidates: list[tuple[int, int, int, int]] = field(default_factory=list)
    reason: str = ""                # why nothing was read, in operator language

    @property
    def pretty(self) -> str:
        from anpr.plates import pretty
        return pretty(self.text)

    def as_dict(self) -> dict:
        return {
            "text": self.text, "pretty": self.pretty, "confidence": round(self.confidence, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "pattern": self.pattern, "raw": self.raw, "repaired": self.repaired,
            "char_confidence": round(self.char_confidence, 4),
            "agreement": round(self.agreement, 3),
            "char_confidences": [round(c, 3) for c in self.char_confidences],
            "boxes": self.boxes, "plate_found": self.plate_found, "variant": self.variant,
            "backend": self.backend, "accepted": self.accepted,
            "floor": round(self.floor, 3),
            "candidates": self.candidates, "reason": self.reason,
        }


class ANPREngine:
    """Loads (or trains) the glyph model once and reads plates."""

    def __init__(self, model: glyph_model.GlyphModel | None = None, max_merge: int = 3,
                 char_bonus: float = 0.25, skip_penalty: float = 1.6,
                 crnn_reader=None, use_crnn: bool = True, easyocr_reader=None,
                 anpr_ocr_reader=None):
        self.model = model or glyph_model.load_or_train()
        # The CRNN reads the plate without segmenting it, which is what the
        # camera-realistic corpus demands. If it is not available - no torch, no
        # weights - everything still works on the classical pipeline.
        self.crnn = crnn_reader if crnn_reader is not None else (
            crnn_mod.load() if use_crnn else None)
        # Evaluation-only extra backends (anpr/easyocr_reader.py,
        # anpr/anpr_ocr_reader.py). Only used when explicitly passed in -
        # never picked up implicitly, unlike crnn/classical above - so the
        # default engine construction used by sim/api/dashboard is
        # unaffected. See CLAUDE.md for why.
        self.easyocr = easyocr_reader
        self.anpr_ocr = anpr_ocr_reader
        self.max_merge = max_merge
        # Reward for explaining one more character (an insertion bonus, as in
        # speech decoding). Without it the DP is free to drop a hard character
        # to keep its average confidence high; a plate is not allowed to be
        # partially read, so paying for coverage is the right trade.
        self.char_bonus = char_bonus
        # Cost of discarding an atom as noise. Mud specks, rivets and frame
        # fragments segment like characters; without an escape hatch the DP has
        # to fold them into a neighbouring glyph and corrupts it.
        self.skip_penalty = skip_penalty
        self._last_debug: dict = {}
        self._class_index = {c: i for i, c in enumerate(self.model.classes)}
        self._letter_idx = np.array([self._class_index[c] for c in sorted(LETTERS)
                                     if c in self._class_index])
        self._digit_idx = np.array([self._class_index[c] for c in sorted(DIGITS)
                                    if c in self._class_index])

    # --- public API ----------------------------------------------------
    def read(self, image: np.ndarray) -> PlateRead:
        """Read a plate crop (grayscale or BGR)."""
        if self.anpr_ocr is not None:
            return self.anpr_ocr.read(image)
        if self.easyocr is not None:
            return self.easyocr.read(image)
        if self.crnn is not None:
            return self._read_crnn(image)
        return self._read_classical(image)

    def _read_crnn(self, image: np.ndarray) -> PlateRead:
        """CRNN + grammar-constrained CTC decode.

        No binarisation, no segmentation: the network sees the greyscale crop
        and CTC marginalises over every alignment between its output columns and
        the string.

        Confidence is that same marginal: the exact probability CTC assigns to
        the decoded string, normalised per character. It is not the weakest
        character any more, because the weakest character was being estimated by
        a heuristic scan that ranked correct reads *below* wrong ones (AUROC
        0.37). The weakest character is still reported, as `char_confidence`,
        and still drives the state-code repair - it is now computed from the
        alignment marginal rather than guessed at.
        """
        gray = segment.to_gray(image)
        self._last_debug = {"norm": crnn_mod.prepare(gray), "ink": None, "atoms": []}
        return self._crnn_plate_read(self.crnn.read(gray))

    def _crnn_plate_read(self, decoded) -> PlateRead:
        """Turn one CRNN decode into a PlateRead."""
        text, confs, pattern, greedy, score = decoded
        if not text:
            return PlateRead(text="", confidence=0.0, plate_found=False, backend="crnn",
                             reason="the recogniser produced no registration for this crop")
        text, repaired = self._repair_state(text, confs)
        weakest = float(np.min(np.clip(confs, 1e-6, 1.0))) if confs else 0.0
        mean = float(np.exp(np.mean(np.log(np.clip(confs, 1e-6, 1.0))))) if confs else 0.0
        return PlateRead(text=text, confidence=score, char_confidence=weakest,
                         agreement=1.0, mean_confidence=mean,
                         char_confidences=list(confs), pattern=pattern,
                         raw=greedy, repaired=repaired, variant="crnn", backend="crnn")

    def read_evidence(self, images: list[np.ndarray]):
        """Read a burst and keep the CTC lattices -> (fused read, evidence).

        The evidence is what makes a failed capture repairable later: the plate
        is often still in the lattice when the decode did not find it, so a
        candidate from a neighbouring camera can be scored against the original
        image rather than against the wrong string it produced. Only the CRNN
        backend has a lattice; on the classical fallback this degrades to an
        ordinary burst read with no evidence, and those captures stay
        unrepairable.
        """
        frames = [im for im in images if im is not None and im.size]
        if self.crnn is None:
            return self.read_burst(frames), []
        reads, evidence = [], []
        for im in frames:
            decoded, lattices = self.crnn.read_with_logits(segment.to_gray(im))
            evidence.append(lattices)
            r = self._crnn_plate_read(decoded)
            if r.text:
                reads.append(r)
        return self.fuse_reads(reads), evidence

    def _read_classical(self, image: np.ndarray) -> PlateRead:
        """Segment-and-classify: kept as the fallback when torch is absent.

        Every binarisation hypothesis is decoded and the highest-scoring read
        wins, so one bad threshold no longer costs the plate.
        """
        norm = segment.normalize(image)
        self._last_debug = {"norm": norm, "ink": None, "atoms": []}
        best = None
        atoms_seen = []
        decoded_strings: list[str] = []
        for variant, ink in segment.ink_variants(norm):
            atoms = segment.segment(ink)
            atoms_seen.append(atoms)
            if len(atoms) < 4:
                continue
            units, probs = self._score_units(ink, atoms)
            decoded = self._decode(atoms, units, probs)
            if decoded is None:
                continue
            score, text, confs, pattern, spans = decoded
            decoded_strings.append(text)
            if best is None or score > best[0]:
                best = (score, text, confs, pattern, spans, atoms, ink, variant,
                        self._greedy(atoms, units, probs))
        if best is None:
            widest = max(atoms_seen, key=len) if atoms_seen else []
            self._last_debug["atoms"] = widest
            return PlateRead(
                text="", confidence=0.0, boxes=widest, plate_found=False,
                reason=(f"segmented {len(widest)} marks but no arrangement of them spells a "
                        "valid Indian registration (two letters, district digits, series "
                        "letters, four digits)"))

        _, text, confs, pattern, spans, atoms, ink, variant, raw = best
        self._last_debug = {"norm": norm, "ink": ink, "atoms": atoms, "spans": spans}
        agreement_text = text
        text, repaired = self._repair_state(text, confs)
        boxes = [self._union(atoms, a, b) for a, b in spans]
        # Confidence combines two things the engine actually knows.
        #
        # First, the weakest character: one wrong glyph makes the whole
        # registration wrong, so the minimum is the honest summary, not the mean.
        #
        # Second, and far more informative, how many of the binarisation
        # hypotheses decoded the same string. The winner is chosen as the
        # highest-scoring of eight, so its own confidence is inflated by
        # selection - measured over 500 reads it is 0.99 on correct reads and
        # 0.89 on wrong ones, which barely separates them. Agreement between
        # independent hypotheses is not selected for in the same way and scores
        # 0.80 against 0.26. Their product is what the confidence floor uses.
        clipped = np.clip(np.asarray(confs, dtype=float), 1e-6, 1.0)
        char_conf = float(clipped.min())
        agreement = (decoded_strings.count(agreement_text) / len(decoded_strings)
                     if decoded_strings else 0.0)
        return PlateRead(text=text, confidence=char_conf * agreement,
                         char_confidence=char_conf, agreement=agreement,
                         char_confidences=list(confs),
                         mean_confidence=float(np.exp(np.mean(np.log(clipped)))),
                         boxes=boxes, pattern=pattern, raw=raw, repaired=repaired,
                         variant=variant)

    def read_burst(self, images: list[np.ndarray]) -> PlateRead:
        """Fuse the frames a camera captures of one vehicle into one read.

        A real ANPR node does not get one photograph of a car. It is triggered
        by a loop or a tripwire and takes a burst as the vehicle crosses the
        zone - five to fifteen frames at different distances, exposures and
        motion blurs, of the same plate. Reading only the middle one throws
        away every other look.

        The frames are read independently and their decodes voted, weighted by
        the exact CTC score, which is what makes this work at all: a vote is
        only as good as its ability to tell a confident frame from a lucky one.
        Measured over 250 vehicles across all ten conditions, one frame reads
        40.4% of plates and eight read 77.2%, with the same weights - the frames
        fail in different ways and agreement is strong evidence.

        The reported confidence stays the best single frame's, so the storage
        floor keeps exactly the meaning it was calibrated with; how many frames
        agreed is reported separately, in `agreement`.
        """
        return self.fuse_reads([self.read(im) for im in images
                                if im is not None and im.size])

    def fuse_reads(self, reads: list[PlateRead]) -> PlateRead:
        """Combine already-decoded frames of one vehicle into a single read.

        Split out from `read_burst` so a caller that wants to show its working -
        the ANPR Lab draws every frame and what it individually read - does not
        have to decode each frame twice.
        """
        reads = [r for r in reads if r is not None and r.text]
        if not reads:
            return PlateRead(text="", confidence=0.0, plate_found=False,
                             backend="crnn" if self.crnn is not None else "classical",
                             reason="no frame in the burst held a readable registration")
        tally: dict[str, float] = {}
        for r in reads:
            tally[r.text] = tally.get(r.text, 0.0) + max(r.confidence, 1e-9)
        winner = max(tally.items(), key=lambda kv: kv[1])[0]
        agreeing = [r for r in reads if r.text == winner]
        best = max(agreeing, key=lambda r: r.confidence)
        best.agreement = len(agreeing) / len(reads)
        best.variant = f"{best.variant}:burst{len(reads)}" if best.variant else f"burst{len(reads)}"
        return best

    def read_detailed(self, image: np.ndarray):
        """read() plus the intermediate images, for the ANPR Lab view and for
        the self-training harvester."""
        read = self.read(image)
        return read, self._last_debug

    def read_frame(self, frame: np.ndarray, max_candidates: int = 12) -> PlateRead:
        """Read from a full photograph rather than a tight plate crop.

        Every candidate region the localiser proposes is decoded, plus the whole
        frame in case the image already *is* a crop, and the most confident
        legal read wins. When nothing wins, the returned read carries the
        candidate boxes and a plain-language reason, because "no plate found" on
        its own tells an operator nothing about what to try next.
        """
        cands = detect_plate_candidates(frame, max_candidates)
        best = self.read(frame)
        best.candidates = cands
        tried = 1
        for box in cands:
            # A localiser box is approximate, and a crop that carries a strip of
            # bumper into the binariser reads worse than one cut a little tight.
            # Decoding a few insets of the same region costs milliseconds and
            # recovers most of what a loose box loses.
            for crop in _crop_variants(frame, box):
                if crop.size == 0:
                    continue
                tried += 1
                r = self.read(crop)
                if r.text and r.confidence > best.confidence:
                    r.candidates = cands
                    best = r
        note = ("This engine reads vehicle number plates - two letters, district digits, a "
                "series and four digits - and refuses anything that is not one. It is not a "
                "general text or handwriting reader.")
        if not best.text:
            best.plate_found = False
            best.candidates = cands
            best.reason = (f"examined {tried} plate-shaped region(s) and none of them held a "
                           f"readable registration. {note}")
        elif not best.accepted:
            # Something decoded, but not well enough to act on. Saying "found"
            # here would be the dishonest answer: noise and lettering that is
            # not a plate will always produce *some* grammatical string, and the
            # confidence floor is what separates that from a real read.
            best.plate_found = False
            best.reason = (f"best candidate was {best.pretty} at {best.confidence:.0%} "
                           f"confidence, below the {best.floor:.0%} floor, so "
                           f"the platform would discard it rather than store it. {note}")
        return best

    # --- decoding ------------------------------------------------------
    def _score_units(self, ink, atoms):
        """Classify every grouping of 1..max_merge adjacent atoms."""
        units: list[tuple[int, int]] = []
        feats = []
        n = len(atoms)
        for i in range(n):
            for m in range(1, min(self.max_merge, n - i) + 1):
                box = self._union(atoms, i, i + m)
                if box[2] > 0.45 * ink.shape[1]:
                    continue
                units.append((i, i + m))
                feats.append(segment.glyph(ink, box))
        proba = self.model.clf.predict_proba(np.array(feats, np.float32))
        return units, {u: proba[k] for k, u in enumerate(units)}

    _union = staticmethod(lambda atoms, i, j: union_box(atoms, i, j))

    def _slot_best(self, p: np.ndarray, slot: str):
        idx = self._letter_idx if slot == "L" else self._digit_idx
        k = int(idx[int(np.argmax(p[idx]))])
        return self.model.classes[k], float(p[k])

    def _skip_costs(self, atoms) -> list[float]:
        """Per-atom cost of dropping it as noise: cheap for specks, dear for
        anything the size of a real character."""
        heights = sorted(a[3] for a in atoms)
        areas = sorted(a[2] * a[3] for a in atoms)
        hmed = heights[len(heights) // 2] or 1
        amed = areas[len(areas) // 2] or 1
        costs = []
        for (_, _, w, h) in atoms:
            small = (h < 0.62 * hmed) or (w * h < 0.42 * amed)
            costs.append(0.35 * self.skip_penalty if small else self.skip_penalty)
        return costs

    def _decode(self, atoms, units, probs):
        """DP over atom groupings, one pass per grammar pattern.

        dp[a][k] = best score for explaining atoms[0:a] as the first k characters
        of the pattern. Three moves are available at each step: take the next
        1..max_merge atoms as one character, or drop one atom as noise.
        """
        n = len(atoms)
        skip = self._skip_costs(atoms)
        best_overall = None
        for pattern in PATTERNS:
            K = len(pattern)
            if K > n or n > self.max_merge * K + 4:   # a pattern the atoms cannot support
                continue
            NEG = -1e9
            dp = np.full((n + 1, K + 1), NEG)
            back: dict = {}
            dp[0][0] = 0.0
            for a in range(n):
                for k in range(K + 1):
                    if dp[a][k] == NEG:
                        continue
                    drop = dp[a][k] - skip[a]
                    if drop > dp[a + 1][k]:
                        dp[a + 1][k] = drop
                        back[(a + 1, k)] = (a, k, None, None)
                    if k == K:
                        continue
                    for m in range(1, self.max_merge + 1):
                        u = (a, a + m)
                        if u not in probs:
                            continue
                        ch, p = self._slot_best(probs[u], pattern[k])
                        score = dp[a][k] + math.log(max(p, 1e-6)) + self.char_bonus
                        if score > dp[a + m][k + 1]:
                            dp[a + m][k + 1] = score
                            back[(a + m, k + 1)] = (a, k, ch, p)
            total = dp[n][K]
            if total == NEG:
                continue
            if best_overall is None or total > best_overall[0]:
                chars, confs, spans = [], [], []
                a, k = n, K
                while a > 0:
                    pa, pk, ch, p = back[(a, k)]
                    if ch is not None:
                        chars.append(ch)
                        confs.append(p)
                        spans.append((pa, a))
                    a, k = pa, pk
                best_overall = (total, "".join(reversed(chars)), list(reversed(confs)),
                                pattern, list(reversed(spans)))
        if best_overall is None:
            return None
        return best_overall   # (score, text, confs, pattern, spans)

    def _greedy(self, atoms, units, probs) -> str:
        """Unconstrained per-atom read — kept for the dashboard so an operator can
        see what the grammar actually fixed."""
        out = []
        for i in range(len(atoms)):
            p = probs.get((i, i + 1))
            if p is None:
                continue
            out.append(self.model.classes[int(np.argmax(p))])
        return "".join(out)

    def _repair_state(self, text: str, confs: list[float]) -> tuple[str, bool]:
        """Snap the first two characters to a real state code when they are not
        one and the fix is a single confusable substitution."""
        return repair_state_code(text)


# --- confusion --------------------------------------------------------
_CONF: dict[str, set[str]] = {}
for _a, _b in config.CONFUSION_PAIRS:
    _CONF.setdefault(_a, set()).add(_b)
    _CONF.setdefault(_b, set()).add(_a)


def _confusable(a: str, b: str) -> bool:
    return b in _CONF.get(a, ())


def confusion_distance(a: str, b: str) -> float:
    """Edit distance where a confusable substitution costs 0.4 instead of 1.

    Used by trajectory search: an operator typing KA05MJ1234 should still find
    the sighting a rain-blurred camera stored as KA05NJ1234.
    """
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0.0
            elif _confusable(a[i - 1], b[j - 1]):
                sub = 0.4
            else:
                sub = 1.0
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
    return prev[m]


# --- plate localisation ------------------------------------------------
import os

_yolo_model = None
def _load_yolo():
    global _yolo_model
    model_path = str(config.MODEL_DIR / "yolov8n-obb-plate.pt")
    if _yolo_model is None and os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(model_path)
        except ImportError:
            pass
    return _yolo_model

def detect_plate_candidates(frame: np.ndarray, max_candidates: int = 12) -> list:
    """Locate plate-shaped regions in a wider photograph (YOLO or Classical)."""
    yolo = _load_yolo()
    if yolo is not None:
        return _detect_plate_candidates_yolo(yolo, frame, max_candidates)
    return _detect_plate_candidates_classical(frame, max_candidates)

# The OBB polygon behind each box from the most recent
# _detect_plate_candidates_yolo() call, keyed by the axis-aligned box tuple
# detect_plate_candidates() actually returns. detect_plate_candidates()'s own
# public contract stays a plain box list (benchmark.py, compare_backends.py
# both depend on that) - this side-channel is how _crop_variants() recovers
# the rotation that collapsing an OBB to its axis-aligned min/max throws
# away, without changing that contract. Single most-recent-call cache: safe
# because read_frame() consumes detect_plate_candidates()'s result
# synchronously, within the same call, before the next frame is processed.
_last_obb_polygons: dict[tuple, np.ndarray] = {}


def _detect_plate_candidates_yolo(yolo_model, frame: np.ndarray, max_candidates: int = 12) -> list:
    """YOLOv8-based plate localiser (Supports OBB)."""
    # Run inference, confidence threshold 0.25 (standard)
    results = yolo_model(frame, verbose=False)[0]
    boxes = []

    H, W = frame.shape[:2]
    _last_obb_polygons.clear()

    if hasattr(results, 'obb') and results.obb is not None:
        for obb in results.obb:
            conf = float(obb.conf[0])
            coords = obb.xyxyxyxy[0].cpu().numpy()  # shape (4, 2)
            x1, y1 = np.min(coords, axis=0).astype(int)
            x2, y2 = np.max(coords, axis=0).astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append((conf * w * h, (x1, y1, w, h)))
                _last_obb_polygons[(x1, y1, w, h)] = coords
    else:
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                boxes.append((conf * w * h, (x1, y1, w, h)))
            
    boxes.sort(key=lambda t: -t[0])
    return [b for _, b in boxes[:max_candidates]]


def _detect_plate_candidates_classical(frame: np.ndarray, max_candidates: int = 12) -> list:
    """Locate plate-shaped regions in a wider photograph.

    Classical localiser, run at two scales: a plate is a horizontal band of
    tightly-spaced vertical strokes, so a morphological gradient followed by a
    wide closing joins the characters of a plate into one blob while leaving
    most of the scene apart. Candidates are filtered on aspect and fill,
    de-duplicated by overlap, and passed to the reader, which is the real
    arbiter - a region that does not decode is not a plate.

    This is the seam for a production upgrade: return boxes from a trained
    detector (YOLO, RetinaNet) here and nothing downstream changes.
    """
    gray = to_gray_frame(frame)
    if gray is None or gray.shape[0] < 30 or gray.shape[1] < 60:
        return []
    H, W = gray.shape
    scale = min(1.0, 900.0 / max(W, 1))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray
    inv = 1.0 / scale
    small = cv2.bilateralFilter(small, 5, 40, 40)
    h, w = small.shape

    boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    for kx, ky in ((21, 5), (35, 9), (13, 5)):
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 0.05 * w or bh < 8 or bw * bh < 700:
                continue
            ar = bw / max(bh, 1)
            if not (1.6 <= ar <= 7.5):                 # one-row and two-row plates
                continue
            fill = cv2.contourArea(c) / max(bw * bh, 1)
            if fill < 0.35:
                continue
            # edge density inside the box: characters are busy, a wall is not
            density = float((th[y:y + bh, x:x + bw] > 0).mean())
            if density < 0.12:
                continue
            pad_x, pad_y = int(0.03 * bw) + 2, int(0.12 * bh) + 2
            box = (max(0, int((x - pad_x) * inv)), max(0, int((y - pad_y) * inv)),
                   min(W, int((bw + 2 * pad_x) * inv)), min(H, int((bh + 2 * pad_y) * inv)))
            boxes.append((density * bw * bh, box))

    boxes.sort(key=lambda t: -t[0])
    kept: list[tuple[int, int, int, int]] = []
    for _, b in boxes:
        if all(_iou(b, k) < 0.35 for k in kept):
            kept.append(b)
        if len(kept) >= max_candidates:
            break
    return kept


def _crop_variants(frame: np.ndarray, box) -> list:
    """The candidate box, a tightened version, the bright plate face inside
    it, and - when the box came from a YOLOv8-OBB detection - a
    perspective-rectified crop that undoes the plate's rotation/skew."""
    x, y, w, h = box
    H, W = frame.shape[:2]
    out = [frame[y:y + h, x:x + w]]
    ix, iy = int(0.06 * w), int(0.12 * h)
    if w - 2 * ix > 24 and h - 2 * iy > 10:
        out.append(frame[y + iy:y + h - iy, x + ix:x + w - ix])
    face = _plate_face(frame, box)
    if face is not None:
        fx, fy, fw, fh = face
        if fw > 24 and fh > 10:
            out.append(frame[fy:fy + fh, fx:fx + fw])
    warped = _obb_rectified_crop(frame, box)
    if warped is not None:
        out.append(warped)
    return out


def _obb_rectified_crop(frame: np.ndarray, box):
    """Perspective-warp the OBB polygon behind `box` (if any, from the most
    recent YOLOv8-OBB detection) onto a flat rectangle, undoing rotation an
    axis-aligned crop keeps. None when the box wasn't from an OBB detection,
    or the polygon is close enough to axis-aligned that warping buys nothing.
    """
    poly = _last_obb_polygons.get(tuple(box))
    if poly is None:
        return None
    pts = np.asarray(poly, dtype=np.float32)
    # Order corners by angle around the centroid so opposite edge lengths
    # below actually measure width/height rather than a diagonal.
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    pts = pts[order]

    def edge(i, j):
        return float(np.linalg.norm(pts[i] - pts[j]))

    w = max(edge(0, 1), edge(2, 3))
    h = max(edge(1, 2), edge(3, 0))
    if w < 24 or h < 10:
        return None
    # Skip near-axis-aligned boxes: the plain crop already covers them, and
    # warping a near-rectangle just adds interpolation noise for nothing.
    axis_aligned = np.array([[pts[:, 0].min(), pts[:, 1].min()],
                             [pts[:, 0].max(), pts[:, 1].min()],
                             [pts[:, 0].max(), pts[:, 1].max()],
                             [pts[:, 0].min(), pts[:, 1].max()]], dtype=np.float32)
    if float(np.abs(pts - axis_aligned).max()) < 0.03 * max(w, h):
        return None
    if w < h:  # keep the wide side horizontal, matching a plate's aspect
        pts = np.roll(pts, 1, axis=0)
        w, h = h, w
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (int(w), int(h)))


def _plate_face(frame: np.ndarray, box):
    """Tighten a box onto the plate's own bright rectangle, if there is one."""
    x, y, w, h = box
    crop = segment.to_gray(frame[y:y + h, x:x + w])
    if crop.size == 0 or min(crop.shape) < 12:
        return None
    _, th = cv2.threshold(cv2.GaussianBlur(crop, (5, 5), 0), 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw < 0.45 * w or ch < 0.35 * h:
            continue
        ar = cw / max(ch, 1)
        if not (1.4 <= ar <= 8.0):
            continue
        if best is None or cw * ch > best[2] * best[3]:
            best = (cx, cy, cw, ch)
    if best is None:
        return None
    cx, cy, cw, ch = best
    pad = 2
    return (max(0, x + cx - pad), max(0, y + cy - pad), cw + 2 * pad, ch + 2 * pad)


def to_gray_frame(frame: np.ndarray):
    if frame is None or frame.size == 0:
        return None
    return segment.to_gray(frame)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --- module-level singleton -------------------------------------------
_engine: ANPREngine | None = None


def engine() -> ANPREngine:
    global _engine
    if _engine is None:
        _engine = ANPREngine()
    return _engine
