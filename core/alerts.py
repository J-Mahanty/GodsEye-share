"""Real-time alerting.

Every sighting is checked against the watchlist and a set of behavioural rules
as it lands. The rules only ever look at a short history of the same plate, so
the cost per sighting is a couple of indexed lookups and the same code runs
unchanged whether it is fed a simulator stream or a live camera feed.

Rules
-----
watchlist   plate is on the control-room watchlist                    critical
clone       the same plate seen too far apart, too fast               critical
loitering   same camera revisited N times in a short window           medium
detour      returns to a camera it already passed, after covering ground  low
odd_hour    roaming the network during the small hours                low
speeding    implied link speed far above the free-flow limit          medium
"""
from __future__ import annotations

import time
from datetime import datetime

import config
from core import db
from core.network import CameraNetwork, network

DEDUP_MINUTES = {"watchlist": 5.0, "clone": 30.0, "loitering": 30.0,
                 "detour": 45.0, "odd_hour": 120.0, "speeding": 10.0}

SEVERITY = {"watchlist": "critical", "clone": "critical", "loitering": "medium",
            "detour": "low", "odd_hour": "low", "speeding": "medium"}


def _recently_alerted(kind: str, plate: str, now: float, conn) -> bool:
    window = DEDUP_MINUTES.get(kind, 15.0) * 60.0
    r = db.rows("SELECT 1 FROM alerts WHERE kind = ? AND plate = ? AND ts >= ? LIMIT 1",
                (kind, plate, now - window), conn=conn)
    return bool(r)


def _emit(kind: str, message: str, sighting: dict, detail: dict, conn) -> dict | None:
    now = float(sighting["ts"])
    plate = sighting["plate"]
    if _recently_alerted(kind, plate, now, conn):
        return None
    severity = SEVERITY.get(kind, "low")
    alert_id = db.add_alert(kind, severity, message, plate=plate,
                            camera_id=sighting["camera_id"], detail=detail, ts=now, conn=conn)
    return {"id": alert_id, "ts": now, "kind": kind, "severity": severity,
            "plate": plate, "camera_id": sighting["camera_id"], "message": message,
            "detail": detail, "acked": 0}


def evaluate(sighting: dict, conn=None, net: CameraNetwork | None = None) -> list[dict]:
    """Run every rule against one freshly-ingested sighting."""
    conn = conn or db.connect()
    net = net or network()
    out: list[dict] = []
    plate = sighting["plate"]
    now = float(sighting["ts"])
    cam = sighting["camera_id"]
    cam_name = net.name(cam)

    # 1. watchlist ------------------------------------------------------
    watch = db.is_watched(plate, conn=conn)
    if watch:
        a = _emit("watchlist",
                  f"WATCHLIST {plate} seen at {cam_name} - {watch['reason']}",
                  sighting, {"reason": watch["reason"], "listed_severity": watch["severity"],
                             "camera": cam_name, "confidence": sighting.get("confidence")}, conn)
        if a:
            out.append(a)

    history = db.rows(
        "SELECT * FROM sightings WHERE plate = ? AND ts < ? ORDER BY ts DESC LIMIT 8",
        (plate, now), conn=conn)

    if history:
        prev = history[0]
        dt_min = (now - prev["ts"]) / 60.0
        if prev["camera_id"] != cam and dt_min > 0:
            km = net.route_km(prev["camera_id"], cam)
            kmph = km / (dt_min / 60.0) if dt_min > 0 else float("inf")
            free = net.link_free_kmph(prev["camera_id"], cam) or 40.0

            # 2. cloned plate -------------------------------------------
            if (kmph > config.IMPOSSIBLE_SPEED_KMPH and km >= 3.0
                    and float(sighting.get("confidence") or 0) >= 0.70
                    and float(prev["confidence"] or 0) >= 0.70):
                a = _emit("clone",
                          f"{plate} implies {kmph:.0f} km/h between {net.name(prev['camera_id'])} "
                          f"and {cam_name} - probable cloned plate",
                          sighting, {"from": prev["camera_id"], "to": cam, "km": round(km, 2),
                                     "minutes": round(dt_min, 2), "implied_kmph": round(kmph, 1)},
                          conn)
                if a:
                    out.append(a)

            # 6. speeding -------------------------------------------------
            # Only on a real road link: between non-adjacent cameras the route
            # is inferred, so "implied speed" is not evidence of anything.
            elif (net.graph.has_edge(prev["camera_id"], cam)
                  and kmph > 1.6 * free and km >= 1.0):
                a = _emit("speeding",
                          f"{plate} averaged {kmph:.0f} km/h on a {free:.0f} km/h link "
                          f"({net.name(prev['camera_id'])} to {cam_name})",
                          sighting, {"from": prev["camera_id"], "to": cam,
                                     "implied_kmph": round(kmph, 1), "free_kmph": free}, conn)
                if a:
                    out.append(a)

            # 4. doubling back ---------------------------------------------
            # "More than one hop" fires on every ordinary journey, and "drove
            # further than the direct distance" fires whenever a vehicle simply
            # makes two trips. What is actually anomalous is covering real
            # ground and then returning to a camera already passed.
            recent = list(reversed(history[:5])) + [sighting]
            seen_before = [x["camera_id"] for x in recent[:-1]]
            if len(recent) >= 4 and cam in seen_before[:-1]:
                travelled = sum(net.route_km(x["camera_id"], y["camera_id"])
                                for x, y in zip(recent, recent[1:]))
                direct = net.route_km(recent[0]["camera_id"], cam)
                span_min = (now - recent[0]["ts"]) / 60.0
                # Thresholds set by measurement, not taste: at 1.6x over an hour
                # this fired on 28% of all plates, which is a commute, not an
                # anomaly. At 4x within 40 minutes it picks out a few percent.
                if (travelled >= 10.0 and travelled >= 4.0 * max(direct, 1.0)
                        and span_min <= 40 and len(set(seen_before)) >= 3):
                    a = _emit("detour",
                              f"{plate} returned to {cam_name} after {travelled:.1f} km "
                              f"across {len(set(seen_before))} cameras in {span_min:.0f} min",
                              sighting, {"travelled_km": round(travelled, 1),
                                         "direct_km": round(direct, 1),
                                         "cameras": [x["camera_id"] for x in recent],
                                         "minutes": round(span_min, 1)}, conn)
                    if a:
                        out.append(a)

        # 3. loitering ---------------------------------------------------
        window = config.LOITER_WINDOW_MIN * 60.0
        revisits = sum(1 for h in history if h["camera_id"] == cam and h["ts"] >= now - window) + 1
        if revisits >= config.LOITER_REVISITS:
            a = _emit("loitering",
                      f"{plate} passed {cam_name} {revisits} times in "
                      f"{config.LOITER_WINDOW_MIN} minutes",
                      sighting, {"camera": cam, "passes": revisits,
                                 "window_min": config.LOITER_WINDOW_MIN}, conn)
            if a:
                out.append(a)

    # 5. odd hour --------------------------------------------------------
    hour = datetime.fromtimestamp(now).hour
    lo, hi = config.ODD_HOUR_RANGE
    if lo <= hour < hi:
        window = 60.0 * 60.0
        seen = db.rows("SELECT DISTINCT camera_id FROM sightings WHERE plate = ? AND ts >= ?",
                       (plate, now - window), conn=conn)
        if len(seen) >= 3:
            a = _emit("odd_hour",
                      f"{plate} crossed {len(seen)} cameras between {lo:02d}:00 and {hi:02d}:00",
                      sighting, {"cameras": [s["camera_id"] for s in seen], "hour": hour}, conn)
            if a:
                out.append(a)

    return out


def summary(hours: float = 6.0, conn=None) -> dict:
    since = time.time() - hours * 3600
    rows = db.alerts(limit=1000, since=since, conn=conn)
    by_kind: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    for a in rows:
        by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1
        by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1
    return {"hours": hours, "total": len(rows), "by_kind": by_kind, "by_severity": by_sev,
            "open": sum(1 for a in rows if not a["acked"])}


def backfill(hours: float = 6.0, conn=None, net: CameraNetwork | None = None) -> int:
    """Replay recent history through the rules — used after seeding, so a fresh
    demo database already has a populated alert queue."""
    conn = conn or db.connect()
    net = net or network()
    since = (db.rows("SELECT MAX(ts) t FROM sightings", conn=conn)[0]["t"] or time.time()) - hours * 3600
    made = 0
    for s in db.sightings_between(since, conn=conn):
        made += len(evaluate(s, conn=conn, net=net))
    return made
