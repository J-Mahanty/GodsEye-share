"""Awiros/anpr-ocr backend — pretrained specifically on 558,767 real Indian
license plate samples (Apache 2.0, huggingface.co/Awiros/anpr-ocr; PP-OCRv5,
PPHGNetV2_B4 backbone, 37.3M params), paired with our existing YOLOv8-OBB
plate detector. Evaluation-only, same opt-in pattern as
anpr/easyocr_reader.py — not wired into any default engine path.

Preprocessing, architecture config, and decode logic mirror the model's own
reference `test.py` (fetched from the HF repo) as closely as possible - this
is a straight port, not a reinterpretation, since the weights were trained
against that exact preprocessing.

Unlike EasyOCR, this model does no text detection of its own: it expects a
pre-cropped plate image and returns exactly one decoded string, so
read_frame()'s YOLOv8 candidate crops feed it directly with no adaptation.

`ppocr` (PaddleOCR's internal package, needed to construct this architecture)
isn't published standalone on PyPI - the model's own test.py bootstraps it by
git-cloning github.com/PaddlePaddle/PaddleOCR (Apache 2.0) into a sibling
folder, and `_ensure_paddleocr()` below does the same, cached under
models/anpr_ocr/PaddleOCR/ alongside the downloaded weights.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys

import cv2
import numpy as np

import config
from anpr import crnn as crnn_mod
from anpr.ocr import PlateRead, matches_plate_grammar, plate_shape, repair_state_code

_REPO_ID = "Awiros/anpr-ocr"
_CACHE_DIR = str(config.MODEL_DIR / "anpr_ocr")
_PADDLEOCR_DIR = os.path.join(_CACHE_DIR, "PaddleOCR")

# PP-OCRv5 server rec, SVTR_HGNet. CTC head: 64 classes (63 dict chars +
# blank). NRTR head: 67 -> 68 internally (NRTRHead adds +1) to match the
# shipped weights. Copied verbatim from the model's own test.py.
CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr", "dims": 120, "depth": 2,
                            "hidden_dims": 120, "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-05},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    },
}
IMAGE_SHAPE = [3, 48, 320]

#try to install paddle ocr
def available() -> bool:
    """Is paddle importable? The platform runs fine without it - this
    backend just isn't offered."""
    try:
        import paddle  # pyright: ignore[reportMissingImports] # noqa: F401
        return True
    except Exception:
        return False

#if not available, install from github
def _ensure_paddleocr() -> None:
    """Make `ppocr` importable, cloning PaddlePaddle/PaddleOCR (Apache 2.0)
    the same way the model's own test.py bootstraps itself on first run."""
    if os.path.isfile(os.path.join(_PADDLEOCR_DIR, "ppocr", "__init__.py")):
        if _PADDLEOCR_DIR not in sys.path:
            sys.path.insert(0, _PADDLEOCR_DIR)
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"[anpr_ocr_reader] cloning PaddleOCR into {_PADDLEOCR_DIR} (first run only) ...",
          file=sys.stderr)
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/PaddlePaddle/PaddleOCR.git", _PADDLEOCR_DIR,
    ])
    sys.path.insert(0, _PADDLEOCR_DIR)

#Import huggingface dataset
def _ensure_weights() -> tuple[str, str]:
    """Download model.safetensors + en_dict.txt from the HF repo, cached
    under models/anpr_ocr/ - untracked, like the platform's other large
    model files (see .gitignore)."""
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(_REPO_ID, "model.safetensors", local_dir=_CACHE_DIR)
    dict_path = hf_hub_download(_REPO_ID, "en_dict.txt", local_dir=_CACHE_DIR)
    return weights, dict_path

#Resize images for required yolo dimenions
def _resize_for_rec(img_bgr: np.ndarray, target_shape) -> np.ndarray:
    _, h, w = target_shape
    img_h, img_w = img_bgr.shape[:2]
    ratio = h / img_h
    new_w = min(int(img_w * ratio), w)
    resized = cv2.resize(img_bgr, (new_w, h))
    if new_w < w:
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized
        resized = padded
    return resized

#Preprocessing image for model : image to array of pixels
def _preprocess(img_bgr: np.ndarray, target_shape) -> np.ndarray:
    img = _resize_for_rec(img_bgr, target_shape)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose((2, 0, 1))


"""--- grammar-constrained decoding --------------------------------------

The shipped model reports 98.42% exact-match on its own held-out set, but
ppocr's CTCLabelDecode is a plain argmax over 64 classes with no notion of
what a registration looks like. On this repo's real corpus that argmax
routinely emits the state sticker as extra characters ("MH02ER9194JH", six
frames out of thirteen in one cluster) or drops a trailing digit
("MH01AX113"). Both are recoverable without touching the weights: the
network's own posterior already ranks the correct string highly, it just
does not win the greedy path.

So: fold the 64 classes onto the 36-character plate alphabet, run a CTC
prefix beam search, propose every grammatical reading each beam admits, and
rank those by the *exact* CTC score of the whole string - the same
forward-backward recursion anpr/crnn.py uses, borrowed rather than
reimplemented (crnn.ctc_score takes the charset as a parameter for exactly
this). An ungrammatical read is still returned when nothing grammatical is
on the beam, so an unusual-format plate degrades to today's behaviour
instead of vanishing."""

PLATE_ALPHABET = config.ALPHABET                 # "0123456789A..Z"
PLATE_IDX = {c: i + 1 for i, c in enumerate(PLATE_ALPHABET)}   # 0 is blank
PLATE_BLANK = 0
NEG = -1e30

BEAM_WIDTH = 16
BEAM_TOP_K = 6          # classes considered per column
MAX_PREFIX = 14         # longest plate is 11; the slack is what substrings eat

# Probability of upper and lower case letters normalised to uppercase letters
def _fold_probs(probs: np.ndarray, characters: list[str]) -> np.ndarray:
    """Collapse ppocr's 64 classes onto blank + the 36 plate characters.

    `characters` is ['blank', '0'..'9', 'A'..'Z', 'a'..'z', ' ']. A plate is
    uppercase, so probability the model splits between 'B' and 'b' is the same
    evidence and belongs in one column; space carries no plate information and
    behaves exactly like a separator, so it folds into blank. Doing this before
    the search rather than upper()-ing the result afterwards is what stops a
    case-split from pushing the wrong character to the top of a column.
    """
    T = probs.shape[0]
    out = np.zeros((T, len(PLATE_ALPHABET) + 1), dtype=np.float64)
    for j, ch in enumerate(characters):
        if j >= probs.shape[1]:
            break
        col = probs[:, j]
        up = ch.upper()
        if up in PLATE_IDX:
            out[:, PLATE_IDX[up]] += col
        else:                            # 'blank', ' ', anything unexpected
            out[:, PLATE_BLANK] += col
    total = out.sum(axis=1, keepdims=True)
    return out / np.maximum(total, 1e-12)

#Removing characters with extremely low probability (Greedy Alogrithm)
def _greedy(folded: np.ndarray) -> str:
    """Collapsed argmax path - what CTCLabelDecode would have returned."""
    path = folded.argmax(axis=1)
    out, prev = [], -1
    for k in path:
        if k != prev and k != PLATE_BLANK:
            out.append(PLATE_ALPHABET[k - 1])
        prev = k
    return "".join(out)

#Does not immediately decide, keeps tracks of mulitple possibilities using their probabilities
def _prefix_beam(folded: np.ndarray, width: int = BEAM_WIDTH,
                 top_k: int = BEAM_TOP_K) -> list[str]:
    """CTC prefix beam search. Returns the surviving strings, best first."""
    lp = np.log(np.maximum(folded, 1e-12))
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG)}   # (log p_blank, log p_nonblank)
    for t in range(lp.shape[0]):
        row = lp[t]
        cands = np.argpartition(row, -top_k)[-top_k:]
        nxt: dict[tuple, tuple[float, float]] = {}

        def bump(pref, pb=NEG, pnb=NEG):
            b, nb = nxt.get(pref, (NEG, NEG))
            nxt[pref] = (np.logaddexp(b, pb), np.logaddexp(nb, pnb))

        for prefix, (p_b, p_nb) in beams.items():
            p_tot = np.logaddexp(p_b, p_nb)
            for k in cands:
                if k == PLATE_BLANK:
                    bump(prefix, pb=p_tot + row[k])
                    continue
                ch = PLATE_ALPHABET[k - 1]
                if prefix and prefix[-1] == ch:
                    # a repeat only becomes a second character across a blank;
                    # without one it collapses back onto the same prefix
                    bump(prefix, pnb=p_nb + row[k])
                    if len(prefix) < MAX_PREFIX:
                        bump(prefix + (ch,), pnb=p_b + row[k])
                elif len(prefix) < MAX_PREFIX:
                    bump(prefix + (ch,), pnb=p_tot + row[k])
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -np.logaddexp(*kv[1]))[:width])
    ranked = sorted(beams.items(), key=lambda kv: -np.logaddexp(*kv[1]))
    return ["".join(p) for p, _ in ranked if p]

#Used by _prefix_beam to decide if valid
def _grammatical_readings(text: str) -> set[str]:
    """Every legal registration this string could be hiding.

    The string itself when it is already legal; otherwise its substrings -
    which is what strips a trailing state sticker off "MH02ER9194JH" - and a
    single confusable substitution in the state-code slot, which is what turns
    "W8..." into "WB...".
    """
    out: set[str] = set()
    seen = {text}
    repaired, did = repair_state_code(text)
    if did:
        seen.add(repaired)
    for base in seen:
        if matches_plate_grammar(base):
            out.add(base)
        n = len(base)
        for i in range(n):
            for j in range(i + 7, min(n, i + 11) + 1):
                sub = base[i:j]
                if matches_plate_grammar(sub):
                    out.add(sub)
                else:
                    fixed, ok = repair_state_code(sub)
                    if ok and matches_plate_grammar(fixed):
                        out.add(fixed)
    return out


DUAL_ROW_MAX_ASPECT = 2.6      # wider than this and it is a single-row plate

#Cuts multiple rows and lays them side by side to join them into one
def _split_rows(image: np.ndarray) -> np.ndarray | None:
    """A two-row plate, cut at its gutter and laid out as one row.

    The model card says dual-row is handled end-to-end, and on its own corpus
    it is (96.91%). On this repo's real photos it is not: every two-row plate
    inspected reads short by exactly the characters that a squashed row loses -
    MH04DW9020 -> "MH04DWM020", MH01BU1852 -> "MH01BU852", MH47AV6753 ->
    "MH47AV753", MH02EZ2599 -> "MH022599". The cause is geometric, not
    semantic: _resize_for_rec() preserves aspect at a fixed 48px height, so a
    square plate arrives 48x48 inside a 320-wide canvas and each row of
    characters is left about 20px tall.

    Cutting the plate and laying the rows side by side hands the same model a
    single-row-shaped input at full character height. This is an ADDITIONAL
    candidate, never a replacement - read() scores both and keeps whichever the
    network itself likes better, so a single-row plate that happens to be cut
    badly cannot be made worse.
    """
    h, w = image.shape[:2]
    if h < 20 or w < 20 or w / h > DUAL_ROW_MAX_ASPECT:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = ink.sum(axis=1).astype(np.float64)
    lo, hi = int(0.30 * h), int(0.70 * h)
    if hi - lo < 3:
        return None
    cut = lo + int(np.argmin(proj[lo:hi]))
    top, bottom = image[:cut], image[cut:]
    if top.shape[0] < 8 or bottom.shape[0] < 8:
        return None
    height = max(top.shape[0], bottom.shape[0])

    def to_height(part):
        ph, pw = part.shape[:2]
        return cv2.resize(part, (max(1, int(pw * height / ph)), height),
                          interpolation=cv2.INTER_CUBIC)

    return np.hstack([to_height(top), to_height(bottom)])

#Calculating score to give each prediction
def _score(folded: np.ndarray, text: str) -> tuple[float, list[float]]:
    """Exact CTC log-probability of `text` and its per-character posteriors."""
    return crnn_mod.ctc_score(folded, text, idx=PLATE_IDX, blank=PLATE_BLANK,
                              is_probs=True)

#Choosing which prediction to keep based on grammer
def _prefer(a: PlateRead, b: PlateRead) -> bool:
    """Is read `a` better than read `b`?

    Legality first, confidence second. A well-formed registration at 0.64 is
    better evidence than an illegal string at 0.99 - the illegal one cannot be
    a plate whatever the network thinks of it. Comparing on confidence alone
    let "MH47AV753" (a dropped digit, scored 0.99 off the squashed two-row
    crop) beat the row-split read "MH47AV6753" that was actually correct.
    """
    a_ok = ":ungrammatical" not in a.variant
    b_ok = ":ungrammatical" not in b.variant
    if a_ok != b_ok:
        return a_ok
    return a.confidence > b.confidence



#Actual OCR
class AnprOcrReader:


    def __init__(self, model, post_process, paddle_mod):
        self._model = model
        self._post_process = post_process
        self._paddle = paddle_mod
        # ['blank', '0'..'9', 'A'..'Z', 'a'..'z', ' '] - the decode side of the
        # dictionary the weights were trained against. _fold_probs() needs it
        # to know which of the 64 columns is which character.
        self._characters = list(getattr(post_process, "character", []))

    #Convert image to greyscale, match pixels then call _fold_probs(to make alphabets case insensitive)
    def probabilities(self, image: np.ndarray) -> np.ndarray:
        """One forward pass -> the CTC head's (T, 36+1) folded posterior."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tensor = self._paddle.to_tensor(
            np.expand_dims(_preprocess(image, IMAGE_SHAPE), axis=0))
        with self._paddle.no_grad():
            preds = self._model(tensor)
        if isinstance(preds, dict):
            pred_tensor = preds.get("ctc", next(iter(preds.values())))
        elif isinstance(preds, (list, tuple)):
            pred_tensor = preds[0]
        else:
            pred_tensor = preds
        # ppocr's CTCHead softmaxes its own output at inference, so this is
        # already a distribution - do not softmax it again.
        probs = np.asarray(pred_tensor.numpy(), dtype=np.float64)[0]
        return _fold_probs(probs, self._characters)

    #Read a pre-cropped, preprocessed image(_split_row)
    def read(self, image: np.ndarray) -> PlateRead:
        """Read a pre-cropped plate image (grayscale or BGR).

        A plate too square to be a single row is read twice - as-is, and with
        its two rows laid side by side - and the network's own score picks the
        winner. See _split_rows().
        """
        best = self.decode(self.probabilities(image))
        stacked = _split_rows(image)
        if stacked is not None and stacked.size:
            alt = self.decode(self.probabilities(stacked))
            alt.variant += ":rowsplit"
            if alt.text and _prefer(alt, best):
                best = alt
        return best

    #Choosing the final plate as being read
    def decode(self, folded: np.ndarray) -> PlateRead:
        """Turn a folded posterior into a PlateRead.

        Split out from read() so the same posterior can be re-decoded (and so
        the decode can be tested without a paddle forward pass).
        """
        
        greedy = _greedy(folded) #one greedy best choice
        beams = _prefix_beam(folded) #different alternative choices
        if greedy and greedy not in beams:
            beams.append(greedy)

        #Set of all possible values
        candidates: set[str] = set() 
        for b in beams:
            candidates |= _grammatical_readings(b)

        #scoring each candidate and keepin the best candidate
        best, best_conf, best_confs = "", 0.0, []
        for cand in candidates:
            total, per = _score(folded, cand)
            conf = float(np.exp(np.clip(total / len(cand), -60.0, 0.0)))
            if conf > best_conf:
                best, best_conf, best_confs = cand, conf, per

        #Fallback
        grammatical = bool(best)
        if not grammatical:
            # Nothing legal survived. Return the beam's own answer anyway: a
            # format this grammar has not met (KL34A465 in the corpus today)
            # should read wrong, as it does now, not become unreadable.
            fallback = beams[0] if beams else greedy
            if not fallback:
                return PlateRead(text="", confidence=0.0, plate_found=False,
                                 backend="anpr_ocr",
                                 reason="anpr-ocr found no readable text in this crop")
            total, best_confs = _score(folded, fallback)
            best = fallback
            best_conf = float(np.exp(np.clip(total / len(fallback), -60.0, 0.0)))

        weakest = float(np.min(np.clip(best_confs, 1e-6, 1.0))) if best_confs else 0.0
        mean = (float(np.exp(np.mean(np.log(np.clip(best_confs, 1e-6, 1.0)))))
                if best_confs else 0.0)
        return PlateRead(
            text=best, confidence=best_conf, char_confidence=weakest,
            agreement=1.0, mean_confidence=mean, char_confidences=list(best_confs),
            pattern=plate_shape(best), raw=greedy, repaired=best != greedy,
            backend="anpr_ocr", plate_found=True,
            variant="anpr_ocr" if grammatical else "anpr_ocr:ungrammatical")

#Safety net
def load(checkpoint_path: str | None = None) -> AnprOcrReader | None:
    """Construct an AnprOcrReader, or None if unavailable (no paddle, no
    network for the first-run git clone / weight download, etc.).

    `checkpoint_path`: load a fine-tuned checkpoint (a PaddleOCR .pdparams
    file, e.g. models/anpr_ocr/finetune_data/output/best_accuracy.pdparams)
    instead of the base weights from Hugging Face - the base model.safetensors
    is never touched. Used to benchmark a fine-tune against the base model
    with the exact same wrapper/preprocessing (see
    anpr/finetune_anpr_ocr_compare.py).

    Loads torch/ultralytics (if importable) before paddle, on purpose:
    loading paddle first and only later triggering a torch model load (e.g.
    the YOLOv8-OBB detector via anpr.ocr._load_yolo(), or anpr.crnn.load())
    in the same process reproducibly corrupts torch's tensors with a paddle
    C++ error ("PreconditionNotMet - Tensor holds no memory") - a binary
    symbol clash between the two frameworks' compiled extensions. Loading
    torch's model(s) first, then paddle, avoids it; the reverse order does
    not seem to (untested whether it's fully commutative beyond that).
    """
    try:
        from anpr.ocr import _load_yolo
        _load_yolo()
        from anpr import crnn as _crnn_mod
        _crnn_mod.load()
    except Exception:
        pass
    if not available():
        return None
    try:
        _ensure_paddleocr()
        import paddle # pyright: ignore[reportMissingImports]
        from ppocr.modeling.architectures import build_model as ppocr_build_model # pyright: ignore[reportMissingImports]
        from ppocr.postprocess import build_post_process # pyright: ignore[reportMissingImports]
        from safetensors.numpy import load_file # pyright: ignore[reportMissingImports]

        paddle.set_device("cpu")  # no Metal/GPU backend for paddle on macOS
        _, dict_path = _ensure_weights()  # en_dict.txt always comes from the base repo

        post_process = build_post_process({
            "name": "CTCLabelDecode", "character_dict_path": dict_path,
            "use_space_char": True,
        })
        cfg = copy.deepcopy(MODEL_CONFIG)
        model = ppocr_build_model(cfg["Architecture"])
        model.eval()
        if checkpoint_path is not None:
            state = paddle.load(checkpoint_path)
            model.set_state_dict(state)
        else:
            weights_path, _ = _ensure_weights()
            np_state = load_file(weights_path)
            model.set_state_dict({k: paddle.to_tensor(v) for k, v in np_state.items()})
        return AnprOcrReader(model, post_process, paddle)
    except Exception as exc:
        print(f"[anpr_ocr_reader] failed to load: {exc}", file=sys.stderr)
        return None
