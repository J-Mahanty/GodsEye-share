"""Network-constrained repair of captures the engine could not stand behind.

The idea
--------
A camera fails to read a plate. Its neighbours, minutes earlier and minutes
later, did read plates - and whatever drove past this camera almost certainly
appears in one of those lists. That turns an open-vocabulary recognition problem
(36^10 strings) into a closed-set retrieval problem over a few hundred
candidates, which is a far easier problem.

What this is not
----------------
It is *not* "detect which characters are missing and fill them in". That does
not work, and the reason is measurable: per-character CTC posteriors are peaked
almost everywhere, so 99.8% of characters come back above 0.9 confidence
including the wrong ones, and per-character confidence predicts per-character
correctness at AUROC 0.692. The engine knows which *strings* it is unsure of
(AUROC 0.974); it does not know which *character* let it down. There is no
reliable mask to build, so there are no blanks to fill.

What works instead is to keep the evidence and score candidates against it. For
each plate a neighbour saw, `crnn.ctc_score` gives the exact probability that
this camera's own image evidence would have produced *that* string, marginalised
over every alignment. Ranking candidates by that likelihood recovers the true
plate as the top hit in about 70% of failed reads, and it holds up as the
candidate set grows: 78% at ten candidates, 69% at two hundred.

The guards, and why each one is there
-------------------------------------
*Uniqueness is not correctness.* When the vehicle came in off a road no camera
covers, the true plate is simply absent from the candidate set - and the best
of the wrong candidates still looks plausible. Accepting the top hit whenever it
is unique false-fills 5.3% of those cases. So acceptance needs an absolute
likelihood floor *and* a margin over the runner-up, which together encode a
"none of these" hypothesis.

*Inferences must never compound.* Candidate sets are built from `sightings`, and
inferences are written to `inferences`. An inferred plate can therefore never
become evidence for another inference. Without that rule a single mistake
propagates along a corridor and the platform manufactures a journey nobody made
- which, for a product whose headline feature is trajectory reconstruction, is
the worst failure available.

*Clones are exempt.* This pass works by assuming plausible travel between
cameras. The clone rule fires on *implausible* travel. Run the repair over a
cloned plate and it will happily reconstruct one coherent path from two real
vehicles and erase the anomaly. Plates carrying a clone alert are skipped.

*An inference is not a read.* Nothing here writes to `sightings`. What comes out
is a hypothesis with its evidence chain attached, for an operator to act on
knowingly.
"""
from __future__ import annotations

import io
import json
import time

import numpy as np

import config
from core import db
from core.network import CameraNetwork, network


# --- evidence retention -------------------------------------------------
def pack_evidence(frames: list[list[np.ndarray]]) -> bytes:
    """CTC lattices for one capture -> a compressed blob.

    One entry per frame, each holding that frame's row-layout hypotheses. Stored
    at float16: the lattice is a log-probability surface that gets softmaxed
    again on the way out, and half precision costs nothing measurable while
    halving a store that is already the bulkiest thing the platform keeps.
    """
    flat, index = [], []
    for hyps in frames:
        index.append(len(hyps))
        flat.extend(np.asarray(h, dtype=np.float16) for h in hyps)
    if not flat:
        return b""
    buf = io.BytesIO()
    np.savez_compressed(buf, index=np.asarray(index, np.int16),
                        **{f"h{i}": a for i, a in enumerate(flat)})
    return buf.getvalue()


def unpack_evidence(blob: bytes) -> list[list[np.ndarray]]:
    if not blob:
        return []
    with np.load(io.BytesIO(blob)) as z:
        index = z["index"].tolist()
        flat = [z[f"h{i}"].astype(np.float32) for i in range(sum(index))]
    out, at = [], 0
    for n in index:
        out.append(flat[at:at + n])
        at += n
    return out


# --- candidate generation ----------------------------------------------
def _window(net: CameraNetwork, a: str, b: str, ts: float, upstream: bool) -> tuple[float, float]:
    """When a vehicle seen at `a` could have been at `b` (or the reverse).

    The lower bound allows for a vehicle moving faster than free flow, the upper
    for congestion and for a short stop. Both are deliberately generous: a
    window that is too tight loses the true plate entirely, while one that is
    too wide only adds candidates, and the likelihood ranking barely notices
    them - measured, a twentyfold larger candidate set costs nine points.
    """
    free = max(net.free_flow_minutes(a, b), 0.1) * 60.0
    lo = free * config.REPAIR_WINDOW_MIN_FACTOR
    hi = free * config.REPAIR_WINDOW_MAX_FACTOR + config.REPAIR_WINDOW_SLACK_S
    return (ts - hi, ts - lo) if upstream else (ts + lo, ts + hi)


def _cloned_plates(since: float, conn) -> set[str]:
    return {r["plate"] for r in db.rows(
        "SELECT DISTINCT plate FROM alerts WHERE kind = 'clone' AND ts >= ?",
        (since,), conn=conn) if r["plate"]}


def candidate_plates(camera_id: str, ts: float, net: CameraNetwork | None = None,
                     conn=None) -> dict[str, list[dict]]:
    """Plates the neighbouring cameras saw, with the sightings that support them.

    Both directions count. A vehicle that passed this camera was upstream of it
    a few minutes ago and will be downstream of it a few minutes hence, and
    which of those two a given neighbour represents depends on the direction of
    travel, which the failed capture does not tell us.
    """
    net = net or network()
    conn = conn or db.connect()
    out: dict[str, list[dict]] = {}
    banned = _cloned_plates(ts - config.REPAIR_CLONE_LOOKBACK_S, conn)
    for nb in net.neighbours(camera_id):
        for upstream in (True, False):
            a, b = (nb, camera_id) if upstream else (camera_id, nb)
            lo, hi = _window(net, a, b, ts, upstream)
            for r in db.rows(
                    "SELECT plate, ts, camera_id, confidence FROM sightings"
                    " WHERE camera_id = ? AND ts BETWEEN ? AND ?", (nb, lo, hi), conn=conn):
                if r["plate"] in banned:
                    continue
                out.setdefault(r["plate"], []).append({
                    "camera_id": nb, "ts": r["ts"],
                    "position": "upstream" if upstream else "downstream",
                    "minutes": round(abs(ts - r["ts"]) / 60.0, 2),
                })
    return out


# --- scoring ------------------------------------------------------------
def score_candidate(evidence: list[list[np.ndarray]], plate: str) -> float:
    """Mean per-character CTC log-likelihood of `plate` over every frame.

    Each frame contributes its best row-layout hypothesis, and the frames are
    averaged rather than summed so that captures with different burst lengths
    compare on the same scale.
    """
    from anpr import crnn

    if not evidence or not plate:
        return -1e30
    total = 0.0
    for hyps in evidence:
        best = -1e30
        for lg in hyps:
            s, _ = crnn.ctc_score(lg, plate)
            if s > -1e29:
                best = max(best, s / len(plate))
        total += best if best > -1e29 else config.REPAIR_MISSING_FRAME_PENALTY
    return total / len(evidence)


def rank(evidence: list[list[np.ndarray]], plates: list[str]) -> list[tuple[str, float]]:
    """Every candidate, best first."""
    return sorted(((p, score_candidate(evidence, p)) for p in plates),
                  key=lambda kv: -kv[1])


# --- the pass -----------------------------------------------------------
def repair(since: float | None = None, until: float | None = None, limit: int = 200,
           net: CameraNetwork | None = None, conn=None,
           verbose: bool = False) -> list[dict]:
    """Reconcile unresolved captures against what the neighbours saw.

    This runs *behind* the live feed, not inside it. Half the evidence is the
    downstream camera, which by definition has not seen the vehicle yet at the
    moment the capture fails, so `until` should trail the present by at least
    one link's travel time. `config.REPAIR_LAG_S` is that trailing distance.
    """
    net = net or network()
    conn = db.ensure_schema(conn)
    until = until if until is not None else time.time() - config.REPAIR_LAG_S
    pending = db.open_unresolved(since, until, limit, conn=conn)
    made, seen = [], []
    for row in pending:
        seen.append(row["id"])
        evidence = unpack_evidence(row["evidence"])
        if not evidence:
            continue
        support = candidate_plates(row["camera_id"], row["ts"], net, conn)
        if not support:
            continue
        ranked = rank(evidence, list(support))
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else config.REPAIR_MISSING_FRAME_PENALTY
        margin = best_score - runner_up
        if best_score < config.REPAIR_MIN_SCORE or margin < config.REPAIR_MIN_MARGIN:
            continue                       # "none of these" wins
        detail = {"support": support[best], "candidates": len(ranked),
                  "runner_up": round(runner_up, 3),
                  "engine_read": row["best_text"], "engine_confidence": row["confidence"]}
        inf_id = db.add_inference(row["id"], row["ts"], row["camera_id"], best,
                                  round(best_score, 4), round(margin, 4), len(ranked),
                                  detail, true_plate=row["true_plate"], conn=conn)
        made.append({"id": inf_id, "unresolved_id": row["id"], "plate": best,
                     "camera_id": row["camera_id"], "ts": row["ts"],
                     "score": best_score, "margin": margin, "candidates": len(ranked),
                     "true_plate": row["true_plate"], "detail": detail})
        if verbose:
            mark = "" if row["true_plate"] is None else (
                "  ok" if best == row["true_plate"] else f"  WRONG (truth {row['true_plate']})")
            print(f"  {row['camera_id']} {best} from {len(ranked)} candidates, "
                  f"score {best_score:.2f} margin {margin:.2f}{mark}")
    db.resolve_unresolved(seen, conn=conn)
    conn.commit()
    return made


def summary(since: float | None = None, conn=None) -> dict:
    """How much the repair pass is adding, and how far it can be trusted.

    `accuracy` is only populated where ground truth exists, which in deployment
    means never - it is the simulator's audit, not an operational metric.
    """
    conn = db.ensure_schema(conn)
    rows_ = db.inferences(since=since, limit=100000, conn=conn)
    audited = [r for r in rows_ if r["true_plate"]]
    correct = sum(1 for r in audited if r["plate"] == r["true_plate"])
    open_n = db.rows("SELECT COUNT(*) c FROM unresolved WHERE resolved = 0", conn=conn)
    total_unres = db.rows("SELECT COUNT(*) c FROM unresolved", conn=conn)
    return {
        "inferences": len(rows_),
        "unresolved_total": total_unres[0]["c"] if total_unres else 0,
        "unresolved_open": open_n[0]["c"] if open_n else 0,
        "audited": len(audited),
        "accuracy": (correct / len(audited)) if audited else None,
        "mean_candidates": (float(np.mean([r["candidates"] for r in rows_]))
                            if rows_ else 0.0),
    }


# --- command line -------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Reconcile refused captures against what the neighbouring cameras saw")
    ap.add_argument("--hours", type=float, default=6.0,
                    help="how far back to look for unresolved captures")
    ap.add_argument("--lag-minutes", type=float, default=config.REPAIR_LAG_S / 60.0,
                    help="how far behind the present to stop; the downstream camera "
                         "has not seen the vehicle yet inside this window")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--prune", action="store_true",
                    help="also drop retained lattices past REPAIR_EVIDENCE_TTL_S")
    args = ap.parse_args()

    now = time.time()
    made = repair(since=now - args.hours * 3600.0,
                  until=now - args.lag_minutes * 60.0,
                  limit=args.limit, verbose=True)
    s = summary(since=now - args.hours * 3600.0)
    print(f"\n{len(made)} inference(s) from {s['unresolved_total']} refused captures; "
          f"{s['unresolved_open']} still open, {s['mean_candidates']:.0f} candidates each")
    if s["accuracy"] is not None:
        print(f"audited against simulator ground truth: {s['accuracy']:.1%} "
              f"over {s['audited']} inferences")
    if args.prune:
        dropped = prune_evidence(now - config.REPAIR_EVIDENCE_TTL_S)
        print(f"pruned the lattices of {dropped} capture(s) past the retention window")


if __name__ == "__main__":
    main()
