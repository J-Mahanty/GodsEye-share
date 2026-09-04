"""Single-plate trajectory reconstruction.

Turns the scattered sightings of one plate into an ordered journey: which road
it took between two cameras, how long that leg took, how fast it must have been
travelling, which way it was heading, and where the reconstruction is not
trustworthy (an impossible speed, or a gap where the vehicle left the network).

Plate search is confusion-aware. An operator who types KA05MJ1234 must still
find the sighting a rain-blurred camera stored as KA05NJ1234, so candidates are
ranked by an edit distance that charges only 0.4 for a substitution the OCR
engine is known to make.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import config
from anpr.ocr import confusion_distance
from core import db
from core.network import CameraNetwork, network


@dataclass
class Leg:
    from_camera: str
    to_camera: str
    from_name: str
    to_name: str
    departed_ts: float
    arrived_ts: float
    minutes: float
    road_km: float
    implied_kmph: float
    free_flow_kmph: float
    direction: str
    bearing: float
    via: list[str]
    polyline: list[tuple[float, float]]
    plausible: bool
    note: str = ""


@dataclass
class Trajectory:
    plate: str
    sightings: list[dict] = field(default_factory=list)
    legs: list[Leg] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"plate": self.plate, "sightings": self.sightings,
                "legs": [asdict(l) for l in self.legs], "summary": self.summary}


def _enrich(sightings: list[dict], net: CameraNetwork) -> list[dict]:
    out = []
    for s in sightings:
        cam = net.cameras.get(s["camera_id"], {})
        out.append({**s,
                    "camera_name": cam.get("name", s["camera_id"]),
                    "lat": cam.get("lat"), "lon": cam.get("lon"),
                    "sector": cam.get("sector"), "road": cam.get("road")})
    return out


def reconstruct(plate: str, since: float | None = None, until: float | None = None,
                conn=None, net: CameraNetwork | None = None) -> Trajectory:
    """Full spatial-temporal path of one plate, oldest sighting first."""
    net = net or network()
    plate = plate.upper().replace(" ", "")
    raw = db.sightings_for_plate(plate, since, until, conn=conn)
    sightings = _enrich(raw, net)
    legs: list[Leg] = []

    for a, b in zip(sightings, sightings[1:]):
        dt_min = (b["ts"] - a["ts"]) / 60.0
        if a["camera_id"] == b["camera_id"]:
            continue
        km = net.route_km(a["camera_id"], b["camera_id"])
        kmph = (km / (dt_min / 60.0)) if dt_min > 0 else float("inf")
        free = net.link_free_kmph(a["camera_id"], b["camera_id"]) or 40.0
        note, plausible = "", True
        if kmph > config.IMPOSSIBLE_SPEED_KMPH:
            note = (f"implied {kmph:.0f} km/h over {km:.1f} km - physically impossible, "
                    "same plate probably running on two vehicles")
            plausible = False
        elif dt_min > 45 and km < 6:
            note = f"{dt_min:.0f} min for {km:.1f} km - vehicle stopped or parked in between"
        elif len(net.route(a["camera_id"], b["camera_id"])) > 3:
            note = "no direct link - path inferred through intermediate cameras"
        legs.append(Leg(
            from_camera=a["camera_id"], to_camera=b["camera_id"],
            from_name=a["camera_name"], to_name=b["camera_name"],
            departed_ts=a["ts"], arrived_ts=b["ts"], minutes=dt_min,
            road_km=km, implied_kmph=kmph if kmph != float("inf") else 0.0,
            free_flow_kmph=free,
            direction=net.direction(a["camera_id"], b["camera_id"]),
            bearing=net.heading(a["camera_id"], b["camera_id"]),
            via=net.route(a["camera_id"], b["camera_id"]),
            polyline=net.polyline(a["camera_id"], b["camera_id"]),
            plausible=plausible, note=note))

    return Trajectory(plate=plate, sightings=sightings, legs=legs,
                      summary=_summarise(plate, sightings, legs))


def _summarise(plate: str, sightings: list[dict], legs: list[Leg]) -> dict:
    if not sightings:
        return {"plate": plate, "sightings": 0, "found": False}
    total_km = sum(l.road_km for l in legs)
    moving = [l for l in legs if l.plausible and l.minutes > 0]
    duration_min = (sightings[-1]["ts"] - sightings[0]["ts"]) / 60.0
    return {
        "plate": plate,
        "found": True,
        "sightings": len(sightings),
        "cameras_visited": len({s["camera_id"] for s in sightings}),
        "first_seen": sightings[0]["ts"],
        "last_seen": sightings[-1]["ts"],
        "first_camera": sightings[0]["camera_id"],
        "last_camera": sightings[-1]["camera_id"],
        "first_camera_name": sightings[0]["camera_name"],
        "last_camera_name": sightings[-1]["camera_name"],
        "duration_min": duration_min,
        "distance_km": total_km,
        "avg_kmph": (total_km / (sum(l.minutes for l in moving) / 60.0)
                     if sum(l.minutes for l in moving) > 0 else 0.0),
        "sectors": sorted({s.get("sector") for s in sightings if s.get("sector")}),
        "mean_ocr_confidence": sum(s["confidence"] for s in sightings) / len(sightings),
        "implausible_legs": sum(1 for l in legs if not l.plausible),
        "vehicle_class": sightings[-1].get("vehicle_class"),
    }


# --- search -------------------------------------------------------------
def search(query: str, limit: int = 12, since: float | None = None,
           max_distance: float = 1.7, conn=None, check_clones: bool = True) -> list[dict]:
    """Find plates matching a possibly-misread or partial query.

    Exact match ranks first, then confusion-aware near matches, then substring
    matches for operators who only caught part of the plate.

    Every candidate is screened for a duplicate registration on the way out.
    Search is the moment an operator meets a plate, and it is the wrong moment to
    stay quiet about one being on two vehicles: everything they do next - a
    trajectory, a watchlist entry, a challan - assumes it belongs to one. The
    screen is cheap because a clean plate costs one indexed query and a walk over
    consecutive pairs; the expensive misread test only runs once a conflict has
    actually been found.
    """
    q = query.upper().replace(" ", "").replace("-", "")
    if not q:
        return []
    conn = conn or db.connect()
    plates = db.distinct_plates(since, conn=conn)
    scored = []
    for p in plates:
        if p == q:
            d = 0.0
        elif len(q) < 6:
            d = 0.5 if q in p else 99.0                 # partial-plate lookup
        else:
            d = confusion_distance(q, p)
        if d <= max_distance:
            scored.append((d, p))
    scored.sort(key=lambda t: (t[0], t[1]))

    out = []
    for d, p in scored[:limit]:
        row = db.rows(
            "SELECT COUNT(*) n, MIN(ts) first_ts, MAX(ts) last_ts, AVG(confidence) conf"
            " FROM sightings WHERE plate = ?", (p,), conn=conn)[0]
        last = db.rows("SELECT camera_id FROM sightings WHERE plate = ?"
                       " ORDER BY ts DESC LIMIT 1", (p,), conn=conn)
        entry = {"plate": p, "distance": round(d, 2), "exact": d == 0.0,
                 "sightings": row["n"], "first_ts": row["first_ts"],
                 "last_ts": row["last_ts"], "mean_confidence": row["conf"],
                 "last_camera": last[0]["camera_id"] if last else None,
                 "watched": bool(db.is_watched(p, conn=conn)),
                 "clone_verdict": "none", "clone_reason": "", "min_vehicles": 1}
        if check_clones:
            from core import clones

            rep = clones.check(p, since=since, conn=conn)
            entry.update(clone_verdict=rep.verdict, clone_reason=rep.reason,
                         min_vehicles=rep.min_vehicles)
        out.append(entry)
    return out


def co_travellers(plate: str, window_s: float = 120.0, min_shared: int = 3,
                  conn=None) -> list[dict]:
    """Plates repeatedly seen at the same camera within `window_s` of this one —
    a convoy, a tail, or two vehicles simply sharing a commute."""
    conn = conn or db.connect()
    mine = db.sightings_for_plate(plate.upper().replace(" ", ""), conn=conn)
    if not mine:
        return []
    counts: dict[str, dict] = {}
    for s in mine:
        near = db.rows(
            "SELECT plate, camera_id, ts FROM sightings"
            " WHERE camera_id = ? AND ts BETWEEN ? AND ? AND plate != ?",
            (s["camera_id"], s["ts"] - window_s, s["ts"] + window_s, s["plate"]), conn=conn)
        for r in near:
            e = counts.setdefault(r["plate"], {"plate": r["plate"], "shared": 0, "cameras": set()})
            e["shared"] += 1
            e["cameras"].add(r["camera_id"])
    out = [{**v, "cameras": sorted(v["cameras"]), "camera_count": len(v["cameras"])}
           for v in counts.values() if v["shared"] >= min_shared]
    out.sort(key=lambda e: (-e["camera_count"], -e["shared"]))
    return out[:10]


def last_known(plate: str, conn=None) -> dict | None:
    rows = db.rows("SELECT * FROM sightings WHERE plate = ? ORDER BY ts DESC LIMIT 1",
                   (plate.upper().replace(" ", ""),), conn=conn)
    if not rows:
        return None
    net = network()
    return _enrich(rows, net)[0]


def age_minutes(ts: float) -> float:
    return (time.time() - ts) / 60.0
