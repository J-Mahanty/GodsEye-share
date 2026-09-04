"""GodsEye — central configuration.

Every path and tunable the platform needs, resolved relative to this file so the
modules work regardless of the directory uvicorn/streamlit was launched from.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CAMERA_FILE = DATA_DIR / "cameras.json"
DB_PATH = Path(os.getenv("GODSEYE_DB", DATA_DIR / "godseye.db"))
GLYPH_MODEL = MODEL_DIR / "glyph_mlp.joblib"
FIGURE_DIR = ROOT / "charts"

CITY_NAME = "Kolkata"
CITY_CENTER = (22.5726, 88.3639)

# --- ANPR ---------------------------------------------------------------
# Characters that can appear on an Indian civilian plate.
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GLYPH_SIZE = 32                 # classifier input is GLYPH_SIZE x GLYPH_SIZE
PLATE_NORM_HEIGHT = 64          # plate crops are normalised to this height
# Below this a read is dropped rather than stored. Confidence is the weakest
# character times the share of binarisation hypotheses that agreed, so this is
# not a probability - it is an operating point, chosen on the benchmark: it
# discards about a quarter of reads and lifts the accuracy of what is stored
# from 83% to roughly 96%.
MIN_PLATE_CONFIDENCE = 0.40
# The CRNN's confidence is the exact CTC probability of the decoded string,
# marginalised over every alignment and normalised per character (see
# anpr.crnn.ctc_score). It is a different quantity from the classical engine's
# (weakest character x agreement between binarisation hypotheses), so each
# backend is judged against its own floor, exposed as PlateRead.accepted.
#
# Measured over 800 mixed captures, this floor keeps 43.6% of reads at 92.0%
# exact-string accuracy; 0.98 keeps 38.8% at 96.5% if a site wants to be
# stricter. The previous floor of 0.70 belonged to a heuristic confidence that
# ranked correct reads *below* wrong ones (AUROC 0.37); on the exact score the
# same 90% operating point keeps three times as many reads.
MIN_PLATE_CONFIDENCE_CRNN = 0.96

# anpr_ocr (Awiros PP-OCRv5) reports the same quantity as the CRNN - the exact
# per-character-normalised CTC probability of the decoded string - but over a
# different model, so it needs its own floor. Calibrated over the 532-crop real
# corpus: this floor keeps 89% of reads at 95.1% stored accuracy (0.90 would
# keep 86% at 96.0%, 0.50 keeps 97% at 92.3%).
MIN_PLATE_CONFIDENCE_ANPR_OCR = 0.80

# How many frames a camera contributes per vehicle. A real ANPR node is
# triggered as the vehicle enters the zone and takes a burst, not a photograph:
# the frames differ in distance, exposure and motion blur, they fail in
# different ways, and agreement between them is strong evidence. Measured over
# 250 vehicles, one frame reads 40.4% of plates and eight read 77.2% with
# identical weights. Set to 1 to go back to reading a single frame.
BURST_FRAMES = int(os.getenv("GODSEYE_BURST", "5"))

# Pairs the classifier genuinely confuses on degraded plates. Used both for
# format-aware correction and for fuzzy plate search.
CONFUSION_PAIRS = [
    ("0", "O"), ("0", "D"), ("0", "Q"), ("1", "I"), ("1", "L"), ("1", "T"),
    ("2", "Z"), ("5", "S"), ("6", "G"), ("8", "B"), ("4", "A"), ("7", "T"),
    ("9", "G"), ("U", "V"), ("M", "N"), ("C", "G"), ("K", "X"), ("3", "8"),
]

# --- Network-constrained repair -----------------------------------------
# A capture below the floor is not stored as a sighting, but its CTC lattice is
# kept briefly: once the neighbouring cameras have reported, the plate can be
# recovered by ranking what *they* saw against this camera's own evidence.
# See core/repair.py for why this is a ranking problem and not a fill-in-the-
# blanks one.
REPAIR_ENABLED = os.getenv("GODSEYE_REPAIR", "1") not in ("0", "false", "False")
# How long a vehicle may take between two adjacent cameras, as multiples of the
# free-flow time. Generous on purpose: too tight a window loses the true plate,
# too wide only adds candidates, and the likelihood ranking is barely affected
# by them - a twentyfold larger set costs nine points.
REPAIR_WINDOW_MIN_FACTOR = 0.6      # faster than free flow
REPAIR_WINDOW_MAX_FACTOR = 4.0      # jammed, or stopped for a chai
REPAIR_WINDOW_SLACK_S = 90.0
# The repair runs behind the live feed: half its evidence is the *downstream*
# camera, which has not seen the vehicle yet at the moment the capture fails.
REPAIR_LAG_S = 15 * 60.0
REPAIR_EVIDENCE_TTL_S = 60 * 60.0   # then the lattices are dropped
REPAIR_CLONE_LOOKBACK_S = 6 * 3600.0
REPAIR_MISSING_FRAME_PENALTY = -30.0
# Acceptance. Both conditions together stand in for a "none of these"
# hypothesis: when the vehicle entered off a road no camera covers, the true
# plate is simply absent from the candidate set and the best wrong candidate
# still looks plausible.
#
# The margin is what actually buys the precision, and by a wide margin. Swept
# over 60 refused captures from a simulated 90 minutes: at a score floor of
# -3.0, requiring no margin accepts 59 of them at 79.7% precision, while
# requiring 0.15 accepts 46 at 100%. The floor alone is a much blunter
# instrument. The floor is kept tighter than the sweep strictly needs because
# that sweep under-represents the case the floor exists for: the simulator's
# camera coverage is dense, so the true plate was in the candidate set 93.3% of
# the time. On a network with real gaps that share is lower and the floor is
# what stops a confident-looking wrong candidate getting through.
#
# 100% on 46 samples is not 100%. The upper bound of the 95% interval is about
# 6%, so treat these as "roughly 95% precision" until measured on more data.
REPAIR_MIN_SCORE = -2.00            # per-character CTC log-likelihood
REPAIR_MIN_MARGIN = 0.15            # ... and it must beat the runner-up by this

# --- Simulation ---------------------------------------------------------
FLEET_SIZE = int(os.getenv("GODSEYE_FLEET", "260"))
SIM_TICK_SECONDS = float(os.getenv("GODSEYE_TICK", "1.0"))
# Run every simulated sighting through the real OCR pipeline (renders a plate
# image, degrades it, reads it back). Truthful end-to-end, but CPU-hungry.
INLINE_OCR = os.getenv("GODSEYE_INLINE_OCR", "1") not in ("0", "false", "False")
INLINE_OCR_MAX_PER_TICK = int(os.getenv("GODSEYE_INLINE_OCR_MAX", "12"))

# --- Analytics ----------------------------------------------------------
DEFAULT_WINDOW_MIN = 30
CONGESTION_THRESHOLDS = {"free": 1.25, "moderate": 1.6, "heavy": 2.2}  # ratio vs free-flow

# --- Alerts -------------------------------------------------------------
IMPOSSIBLE_SPEED_KMPH = 160.0   # above this between two cameras => cloned plate

# --- Clone detection (core/clones.py) -----------------------------------
# A clone verdict accuses a real motorist, so each of these exists to rule out a
# cheaper explanation than "this registration is on two vehicles".
# Two cameras a few hundred metres apart turn ordinary clock skew into a huge
# implied speed, so a conflict has to span real distance.
CLONE_MIN_KM = 3.0
# Evidence from a read the engine barely stood behind is not evidence. Both ends
# of a conflicting pair must clear this.
CLONE_MIN_CONFIDENCE = 0.70
# One impossible leg is a lead; several independent ones are a finding. Below
# this a verdict is reported as "suspected" and stays off the alert queue.
CLONE_CONFIRM_PAIRS = 2
# How far either side of a sighting to look for the plate it might really have
# been, and how close in OCR confusion distance that plate has to be. 1.0 is
# about one substitution the engine is known to make.
CLONE_MISREAD_WINDOW_S = 15 * 60.0
CLONE_MISREAD_MAX_DISTANCE = 1.0
CLONE_ALERT_DEDUP_S = 30 * 60.0
LOITER_REVISITS = 4             # same camera N times ...
LOITER_WINDOW_MIN = 45          # ... within this window
ODD_HOUR_RANGE = (1, 5)         # 01:00-05:00 local
API_BASE = os.getenv("GODSEYE_API", "http://127.0.0.1:8000")
