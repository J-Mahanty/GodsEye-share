"""Macro traffic analytics over the whole camera network.

Everything here is derived from one primitive: consecutive sightings of the same
plate at two cameras. That pair gives a road distance and a time, hence a speed,
hence congestion; aggregated over all plates it gives link flows, origin-
destination demand, bottlenecks and the movement heatmap.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from core import db
from core.network import CameraNetwork, network


def _window(minutes: float, end: float | None = None) -> tuple[float, float]:
    end = end if end is not None else _latest_ts()
    return end - minutes * 60.0, end


def _latest_ts() -> float:
    r = db.rows("SELECT MAX(ts) t FROM sightings")
    return (r[0]["t"] or time.time()) if r else time.time()


def frame(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None) -> pd.DataFrame:
    since, until = _window(minutes, end)
    rows = db.sightings_between(since, until)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("ts")


# --- per camera ---------------------------------------------------------
def camera_density(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None,
                   net: CameraNetwork | None = None) -> pd.DataFrame:
    """Vehicles per minute at each camera, with a per-lane load index."""
    net = net or network()
    df = frame(minutes, end)
    base = pd.DataFrame([{
        "camera_id": c["id"], "name": c["name"], "lat": c["lat"], "lon": c["lon"],
        "sector": c["sector"], "road": c["road"], "lanes": c["lanes"], "type": c["type"],
    } for c in net.cameras.values()])
    if df.empty:
        base["count"] = 0
        base["unique_plates"] = 0
        base["per_min"] = 0.0
        base["mean_confidence"] = 0.0
        base["per_lane_per_min"] = 0.0
        base["load"] = 0.0
        return base

    g = df.groupby("camera_id").agg(
        count=("plate", "size"),
        unique_plates=("plate", "nunique"),
        mean_confidence=("confidence", "mean"),
        mean_speed=("speed_kmph", "mean"),
    ).reset_index()
    out = base.merge(g, on="camera_id", how="left").fillna(
        {"count": 0, "unique_plates": 0, "mean_confidence": 0.0, "mean_speed": 0.0})
    out["per_min"] = out["count"] / max(minutes, 1e-9)
    out["per_lane_per_min"] = out["per_min"] / out["lanes"].clip(lower=1)
    peak = out["per_lane_per_min"].max() or 1.0
    out["load"] = (out["per_lane_per_min"] / peak).round(3)
    return out.sort_values("count", ascending=False)


def sector_load(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None) -> pd.DataFrame:
    d = camera_density(minutes, end)
    g = d.groupby("sector").agg(cameras=("camera_id", "size"), count=("count", "sum"),
                                per_min=("per_min", "sum")).reset_index()
    return g.sort_values("count", ascending=False)


# --- link level ---------------------------------------------------------
def link_flows(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None,
               net: CameraNetwork | None = None, max_gap_min: float = 60.0) -> pd.DataFrame:
    """Flow and speed on every camera-to-camera movement seen in the window."""
    net = net or network()
    df = frame(minutes, end)
    cols = ["from_camera", "to_camera", "link", "name", "vehicles", "median_kmph",
            "p15_kmph", "free_kmph", "congestion_ratio", "level", "km",
            "from_lat", "from_lon", "to_lat", "to_lon", "direction"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.sort_values(["plate", "ts"])
    nxt = df.shift(-1)
    same = df["plate"] == nxt["plate"]
    moved = df["camera_id"] != nxt["camera_id"]
    pairs = pd.DataFrame({
        "from_camera": df["camera_id"], "to_camera": nxt["camera_id"],
        "dt_min": (nxt["ts"] - df["ts"]) / 60.0,
    })[same & moved]
    pairs = pairs[(pairs["dt_min"] > 0) & (pairs["dt_min"] <= max_gap_min)]
    # Keep only movements along a real road link. Between two cameras that are
    # not neighbours the vehicle's path is inferred, so neither the distance nor
    # the speed can be trusted as a measurement of that road.
    adjacent = [net.graph.has_edge(a, b)
                for a, b in zip(pairs["from_camera"], pairs["to_camera"])]
    pairs = pairs[adjacent]
    if pairs.empty:
        return pd.DataFrame(columns=cols)

    km = {}
    for a, b in pairs[["from_camera", "to_camera"]].drop_duplicates().itertuples(index=False):
        km[(a, b)] = net.route_km(a, b)
    pairs["km"] = [km[(a, b)] for a, b in zip(pairs["from_camera"], pairs["to_camera"])]
    pairs = pairs[pairs["km"] > 0]
    pairs["kmph"] = pairs["km"] / (pairs["dt_min"] / 60.0)
    pairs = pairs[pairs["kmph"] <= config.IMPOSSIBLE_SPEED_KMPH]
    if pairs.empty:
        return pd.DataFrame(columns=cols)

    g = pairs.groupby(["from_camera", "to_camera"]).agg(
        vehicles=("kmph", "size"),
        median_kmph=("kmph", "median"),
        p15_kmph=("kmph", lambda s: float(np.percentile(s, 15))),
        km=("km", "first"),
    ).reset_index()

    g["free_kmph"] = [net.link_free_kmph(a, b) for a, b in zip(g["from_camera"], g["to_camera"])]
    g["congestion_ratio"] = (g["free_kmph"] / g["median_kmph"].clip(lower=1e-6)).round(2)
    g["level"] = g["congestion_ratio"].apply(congestion_level)
    g["name"] = [_link_name(net, a, b) for a, b in zip(g["from_camera"], g["to_camera"])]
    g["link"] = g["from_camera"] + " -> " + g["to_camera"]
    g["direction"] = [net.direction(a, b) for a, b in zip(g["from_camera"], g["to_camera"])]
    coords = {c: (v["lat"], v["lon"]) for c, v in net.cameras.items()}
    g["from_lat"] = [coords[a][0] for a in g["from_camera"]]
    g["from_lon"] = [coords[a][1] for a in g["from_camera"]]
    g["to_lat"] = [coords[b][0] for b in g["to_camera"]]
    g["to_lon"] = [coords[b][1] for b in g["to_camera"]]
    return g[cols].sort_values("vehicles", ascending=False)


def _link_name(net: CameraNetwork, a: str, b: str) -> str:
    if net.graph.has_edge(a, b):
        return net.graph[a][b]["name"] or f"{net.name(a)} - {net.name(b)}"
    return f"{net.name(a)} - {net.name(b)}"


def congestion_level(ratio: float) -> str:
    t = config.CONGESTION_THRESHOLDS
    if ratio <= t["free"]:
        return "free"
    if ratio <= t["moderate"]:
        return "moderate"
    if ratio <= t["heavy"]:
        return "heavy"
    return "severe"


def bottlenecks(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None,
                min_vehicles: int = 4, top: int = 10) -> pd.DataFrame:
    """Links that are both slow and busy — where intervention actually pays."""
    f = link_flows(minutes, end)
    if f.empty:
        return f
    f = f[f["vehicles"] >= min_vehicles].copy()
    if f.empty:
        return f
    # rank by delay-minutes lost: extra time per vehicle x vehicles on the link
    f["delay_min_per_veh"] = (f["km"] / f["median_kmph"].clip(lower=1e-6)
                              - f["km"] / f["free_kmph"].clip(lower=1e-6)) * 60.0
    f["delay_min_total"] = (f["delay_min_per_veh"] * f["vehicles"]).round(1)
    f["delay_min_per_veh"] = f["delay_min_per_veh"].round(2)
    return f.sort_values("delay_min_total", ascending=False).head(top)


# --- demand -------------------------------------------------------------
def od_matrix(minutes: float = 120.0, end: float | None = None, gateways_only: bool = False,
              net: CameraNetwork | None = None) -> pd.DataFrame:
    """Origin-destination pairs: where each plate entered and left the network."""
    net = net or network()
    df = frame(minutes, end)
    if df.empty:
        return pd.DataFrame(columns=["origin", "destination", "trips", "origin_name",
                                     "destination_name", "median_minutes"])
    if gateways_only:
        df = df[df["camera_id"].isin(net.gateways)]
        if df.empty:
            return pd.DataFrame(columns=["origin", "destination", "trips", "origin_name",
                                         "destination_name", "median_minutes"])
    first = df.groupby("plate").first().reset_index()[["plate", "camera_id", "ts"]]
    last = df.groupby("plate").last().reset_index()[["plate", "camera_id", "ts"]]
    j = first.merge(last, on="plate", suffixes=("_o", "_d"))
    j = j[j["camera_id_o"] != j["camera_id_d"]]
    if j.empty:
        return pd.DataFrame(columns=["origin", "destination", "trips", "origin_name",
                                     "destination_name", "median_minutes"])
    j["minutes"] = (j["ts_d"] - j["ts_o"]) / 60.0
    g = j.groupby(["camera_id_o", "camera_id_d"]).agg(
        trips=("plate", "size"), median_minutes=("minutes", "median")).reset_index()
    g = g.rename(columns={"camera_id_o": "origin", "camera_id_d": "destination"})
    g["origin_name"] = [net.name(c) for c in g["origin"]]
    g["destination_name"] = [net.name(c) for c in g["destination"]]
    g["median_minutes"] = g["median_minutes"].round(1)
    return g.sort_values("trips", ascending=False)


# --- map layers ---------------------------------------------------------
def heatmap(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None,
            net: CameraNetwork | None = None, samples_per_link: int = 6) -> pd.DataFrame:
    """Weighted points for the movement heatmap.

    Camera nodes carry their own count, and each traversed link is sampled along
    its polyline so the heat follows the roads instead of pooling on junctions.
    """
    net = net or network()
    pts = []
    d = camera_density(minutes, end, net)
    for r in d.itertuples(index=False):
        if r.count > 0:
            pts.append({"lat": r.lat, "lon": r.lon, "weight": float(r.count),
                        "kind": "camera", "label": r.name})
    flows = link_flows(minutes, end, net)
    for r in flows.itertuples(index=False):
        line = net.polyline(r.from_camera, r.to_camera)
        if len(line) < 2:
            continue
        # congestion multiplies the heat: slow, busy roads should glow hottest
        w = float(r.vehicles) * min(float(r.congestion_ratio), 3.0) / samples_per_link
        for lat, lon in _sample_along(line, samples_per_link):
            pts.append({"lat": lat, "lon": lon, "weight": w, "kind": "link", "label": r.name})
    return pd.DataFrame(pts)


def _sample_along(line, n: int):
    """n points spaced evenly *by distance* along a polyline.

    Sampling by vertex index instead would bunch the heat wherever a road has
    closely-spaced shape points, which is exactly where the road bends.
    """
    from core.network import haversine_km

    seg = [haversine_km(a[0], a[1], b[0], b[1]) for a, b in zip(line, line[1:])]
    total = sum(seg)
    if total <= 0:
        return [line[0]] * n
    cum, acc = [0.0], 0.0
    for d in seg:
        acc += d
        cum.append(acc)
    out = []
    for i in range(n):
        target = total * (i / max(n - 1, 1))
        k = 0
        while k < len(seg) - 1 and cum[k + 1] < target:
            k += 1
        span = max(cum[k + 1] - cum[k], 1e-9)
        t = (target - cum[k]) / span
        out.append((line[k][0] + t * (line[k + 1][0] - line[k][0]),
                    line[k][1] + t * (line[k + 1][1] - line[k][1])))
    return out


def flow_trend(hours: float = 3.0, bucket_min: float = 5.0, end: float | None = None) -> pd.DataFrame:
    """Network-wide sightings and unique vehicles per time bucket."""
    df = frame(hours * 60.0, end)
    if df.empty:
        return pd.DataFrame(columns=["bucket", "sightings", "unique_plates", "mean_confidence"])
    df = df.copy()
    df["bucket"] = pd.to_datetime(df["ts"], unit="s").dt.floor(f"{int(bucket_min)}min")
    g = df.groupby("bucket").agg(sightings=("plate", "size"),
                                 unique_plates=("plate", "nunique"),
                                 mean_confidence=("confidence", "mean")).reset_index()
    return g


def speed_trend(hours: float = 3.0, bucket_min: float = 10.0, end: float | None = None) -> pd.DataFrame:
    """Median network speed per bucket — the city's pulse in one line."""
    end = end if end is not None else _latest_ts()
    out = []
    steps = int(hours * 60 / bucket_min)
    for i in range(steps):
        t_end = end - i * bucket_min * 60
        f = link_flows(bucket_min, t_end)
        if f.empty:
            continue
        out.append({"bucket": pd.to_datetime(t_end, unit="s"),
                    "median_kmph": float(f["median_kmph"].median()),
                    "congested_links": int((f["congestion_ratio"] > config.CONGESTION_THRESHOLDS["moderate"]).sum()),
                    "vehicles": int(f["vehicles"].sum())})
    return pd.DataFrame(out).sort_values("bucket") if out else pd.DataFrame(
        columns=["bucket", "median_kmph", "congested_links", "vehicles"])


# --- headline -----------------------------------------------------------
def city_summary(minutes: float = config.DEFAULT_WINDOW_MIN, end: float | None = None) -> dict:
    d = camera_density(minutes, end)
    f = link_flows(minutes, end)
    st = db.stats()
    worst = None
    if not f.empty:
        b = bottlenecks(minutes, end, top=1)
        if not b.empty:
            r = b.iloc[0]
            worst = {"link": r["name"], "from": r["from_camera"], "to": r["to_camera"],
                     "median_kmph": round(float(r["median_kmph"]), 1),
                     "free_kmph": float(r["free_kmph"]),
                     "level": r["level"], "delay_min_total": float(r["delay_min_total"])}
    return {
        "window_min": minutes,
        "sightings": int(d["count"].sum()),
        "vehicles_per_min": round(float(d["per_min"].sum()), 1),
        "unique_plates": int(d["unique_plates"].sum()),
        "active_cameras": int((d["count"] > 0).sum()),
        "total_cameras": int(len(d)),
        "median_network_kmph": round(float(f["median_kmph"].median()), 1) if not f.empty else 0.0,
        "congested_links": int((f["congestion_ratio"] > config.CONGESTION_THRESHOLDS["moderate"]).sum()) if not f.empty else 0,
        "monitored_links": int(len(f)),
        "worst_link": worst,
        "busiest_camera": (d.iloc[0]["name"] if not d.empty and d.iloc[0]["count"] > 0 else None),
        "mean_ocr_confidence": round(float(st["mean_confidence"]), 3),
        "open_alerts": st["open_alerts"],
        "db": st,
    }
