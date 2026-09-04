"""Map layer builders for the control room.

Kept apart from the page code because the maps are the dense part: the network
is drawn from real link geometry rather than straight lines between markers, and
every layer has to agree on one tooltip field and one colour language.
"""
from __future__ import annotations

import time

import pandas as pd
import pydeck as pdk

# Carto basemaps need no API token, unlike the Mapbox styles pydeck names by
# default — an unauthenticated Mapbox style renders as a blank grey page.
BASEMAPS = {
    "Light": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Streets": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
}

# pydeck reads a plain Python string as a per-row data accessor: passing
# width_units="pixels" compiles to "@@=pixels", deck.gl looks for a column of
# that name, finds nothing, and the width becomes NaN - which is what turned
# every road into a viewport-sized wedge and made the text layers invisible.
# Enum constants must be wrapped so they survive as literals.
PIXELS = pdk.types.String("pixels")

LEVEL_COLOUR = {"free": [38, 166, 91], "moderate": [232, 186, 40],
                "heavy": [235, 120, 32], "severe": [222, 47, 52]}
UNMEASURED = [150, 158, 168]

CAMERA_COLOUR = {"gateway": [126, 87, 194], "junction": [33, 118, 214]}
LOAD_COLOUR = [[70, 170, 230], [240, 190, 60], [230, 90, 60]]


def _num(value, fmt: str) -> str:
    """Format a number that may be missing — a scripted or partial sighting can
    carry no speed, and a tooltip is not worth an exception."""
    try:
        if value is None or value != value:      # None or NaN
            return "-"
        return fmt.format(value)
    except (TypeError, ValueError):
        return "-"


# --- network ------------------------------------------------------------
def road_frame(net, flows: pd.DataFrame) -> pd.DataFrame:
    """Every road link with the congestion measured on it.

    Flows are directional; a road is drawn once, coloured by its worse
    direction, with both directions in the tooltip — a corridor that crawls
    inbound and runs free outbound is the normal case, and hiding it behind an
    average would be the wrong summary.
    """
    rows = []
    measured = {}
    if flows is not None and not flows.empty:
        for r in flows.itertuples(index=False):
            measured[(r.from_camera, r.to_camera)] = r

    for link in net.links():
        a, b = link["a"], link["b"]
        fwd, rev = measured.get((a, b)), measured.get((b, a))
        seen = [x for x in (fwd, rev) if x is not None]
        worst = max(seen, key=lambda x: x.congestion_ratio) if seen else None
        vehicles = sum(x.vehicles for x in seen)
        tip = f"{link['name']}  ·  {link['km']:.1f} km  ·  free-flow {link['free_kmph']:.0f} km/h"
        for x, arrow in ((fwd, "→"), (rev, "←")):
            if x is not None:
                tip += (f"\n{arrow} {net.name(x.from_camera)} to {net.name(x.to_camera)}: "
                        f"{x.vehicles} veh, {x.median_kmph:.0f} km/h ({x.level})")
        if not seen:
            tip += "\nno through traffic measured in this window"
        rows.append({
            "name": link["name"], "path": link["path"], "km": link["km"],
            "vehicles": int(vehicles),
            "level": worst.level if worst is not None else "unmeasured",
            "ratio": float(worst.congestion_ratio) if worst is not None else 0.0,
            "kmph": float(worst.median_kmph) if worst is not None else 0.0,
            "colour": LEVEL_COLOUR.get(worst.level, UNMEASURED) if worst is not None else UNMEASURED,
            "width": 3.0 + 9.0 * min(vehicles / 40.0, 1.0),
            "tip": tip,
        })
    return pd.DataFrame(rows)


def road_layer(roads: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer("PathLayer", data=roads, get_path="path", get_color="colour",
                     get_width="width", width_units=PIXELS, width_min_pixels=2,
                     cap_rounded=True, joint_rounded=True, pickable=True,
                     auto_highlight=True)


def camera_frame(density: pd.DataFrame) -> pd.DataFrame:
    d = density.copy()
    d["load"] = d["load"].fillna(0.0)
    d["count"] = d["count"].fillna(0).astype(int)
    d["radius"] = 90 + d["load"] * 260
    d["colour"] = [
        (LOAD_COLOUR[2] if r.load > 0.6 else LOAD_COLOUR[1] if r.load > 0.3 else LOAD_COLOUR[0])
        + [225] for r in d.itertuples(index=False)]
    d["ring"] = [CAMERA_COLOUR.get(r.type, CAMERA_COLOUR["junction"]) + [255]
                 for r in d.itertuples(index=False)]
    d["ring_radius"] = d["radius"] + [70 if t == "gateway" else 30 for t in d["type"]]
    d["tip"] = [
        f"{r.name}  ({r.camera_id})\n{r.road} · {r.sector} · {r.lanes} lanes · {r.type}\n"
        f"{int(r.count)} reads · {r.per_min:.1f}/min · {int(r.unique_plates)} vehicles\n"
        f"per-lane load {r.load:.2f} · mean read confidence {r.mean_confidence:.0%}"
        for r in d.itertuples(index=False)]
    return d


def camera_layers(cams: pd.DataFrame, labels: bool = True,
                  max_labels: int = 14) -> list[pdk.Layer]:
    out = [
        # outer ring encodes what kind of site it is, inner disc encodes load
        pdk.Layer("ScatterplotLayer", data=cams, get_position=["lon", "lat"],
                  get_radius="ring_radius", get_fill_color="ring", opacity=0.35,
                  radius_min_pixels=4, pickable=False),
        pdk.Layer("ScatterplotLayer", data=cams, get_position=["lon", "lat"],
                  get_radius="radius", get_fill_color="colour", radius_min_pixels=3,
                  stroked=True, get_line_color=[255, 255, 255, 160], line_width_min_pixels=1,
                  pickable=True, auto_highlight=True),
    ]
    if labels:
        # Thirty-five names at city zoom collide into an unreadable mat, and they
        # collide with the basemap's own labels as well. Name the sites an
        # operator is watching - the busiest ones - and let the rest speak
        # through their tooltips.
        shown = cams.nlargest(max_labels, "count") if "count" in cams else cams
        out.append(pdk.Layer(
            "TextLayer", data=shown, get_position=["lon", "lat"], get_text="name",
            get_size=11, size_units=PIXELS, get_color=[35, 40, 48],
            get_alignment_baseline="'top'", get_text_anchor="'middle'",
            get_pixel_offset=[0, 12], background=True,
            get_background_color=[255, 255, 255, 190], background_padding=[3, 1],
            pickable=False))
    return out


def flow_layer(flows: pd.DataFrame, net) -> pdk.Layer | None:
    """Directional arcs: which way the traffic is actually moving, and how much."""
    if flows is None or flows.empty:
        return None
    f = flows.copy()
    f["colour"] = [LEVEL_COLOUR.get(l, UNMEASURED) for l in f["level"]]
    peak = max(f["vehicles"].max(), 1)
    f["awidth"] = 1.0 + 7.0 * (f["vehicles"] / peak)
    f["tip"] = [f"{net.name(r.from_camera)} → {net.name(r.to_camera)}\n"
                f"{r.vehicles} vehicles · {r.median_kmph:.0f} km/h vs {r.free_kmph:.0f} "
                f"free-flow · {r.level}" for r in f.itertuples(index=False)]
    return pdk.Layer("ArcLayer", data=f,
                     get_source_position=["from_lon", "from_lat"],
                     get_target_position=["to_lon", "to_lat"],
                     get_source_color="colour", get_target_color="colour",
                     get_width="awidth", get_height=0.25, get_tilt=12,
                     pickable=True, auto_highlight=True)


def sighting_layer(recent: pd.DataFrame, net, window_s: float = 300.0) -> pdk.Layer | None:
    """The last few minutes of reads, fading with age — the live pulse."""
    if recent is None or recent.empty:
        return None
    now = time.time()
    r = recent.copy()
    r = r[r["ts"] >= now - window_s]
    if r.empty:
        return None
    coords = {c: (v["lat"], v["lon"]) for c, v in net.cameras.items()}
    r = r[r["camera_id"].isin(coords)]
    if r.empty:
        return None
    r["lat"] = [coords[c][0] for c in r["camera_id"]]
    r["lon"] = [coords[c][1] for c in r["camera_id"]]
    age = (now - r["ts"]).clip(lower=0, upper=window_s) / window_s
    r["alpha"] = (235 * (1 - age)).astype(int)
    r["colour"] = [[255, 214, 64, int(a)] for a in r["alpha"]]
    r["rad"] = 130 + 340 * (1 - age)
    r["tip"] = [f"{x.plate} · {net.name(x.camera_id)}\n"
                f"{time.strftime('%H:%M:%S', time.localtime(x.ts))} · heading {x.direction} · "
                f"{x.speed_kmph:.0f} km/h · confidence {x.confidence:.0%}"
                for x in r.itertuples(index=False)]
    return pdk.Layer("ScatterplotLayer", data=r, get_position=["lon", "lat"],
                     get_radius="rad", get_fill_color="colour", radius_min_pixels=2,
                     pickable=True)


# --- trajectory ---------------------------------------------------------
def trajectory_layers(traj, net) -> tuple[list[pdk.Layer], pd.DataFrame]:
    """Path, direction arrows and numbered stops for one vehicle."""
    legs, pts = [], []
    for i, leg in enumerate(traj.legs):
        line = leg.polyline
        if len(line) < 2:
            continue
        legs.append({
            "path": [[lon, lat] for lat, lon in line],
            "colour": [222, 47, 52] if not leg.plausible else
                      [37, 99, 235] if i % 2 == 0 else [59, 130, 246],
            "tip": (f"{leg.from_name} → {leg.to_name}\n{leg.minutes:.0f} min · "
                    f"{leg.road_km:.1f} km · {leg.implied_kmph:.0f} km/h implied "
                    f"(free-flow {leg.free_flow_kmph:.0f}) · heading {leg.direction}"
                    + (f"\n⚠ {leg.note}" if leg.note else "")),
            "width": 6.0,
        })
        mid = line[len(line) // 2]
        pts.append({"lat": mid[0], "lon": mid[1], "kind": "arrow",
                    "label": _arrow(leg.bearing), "tip": leg.direction})

    stops = pd.DataFrame([{
        "lat": s["lat"], "lon": s["lon"], "order": str(i + 1),
        "camera": s["camera_name"],
        "tip": (f"{i+1}. {s['camera_name']}\n"
                f"{time.strftime('%d %b %H:%M:%S', time.localtime(s['ts']))}\n"
                f"heading {s['direction']} · {s['speed_kmph']:.0f} km/h · "
                f"read confidence {s['confidence']:.0%} · capture {s['condition']}"),
        "colour": [22, 163, 74, 235] if i == 0 else
                  ([220, 38, 38, 235] if i == len(traj.sightings) - 1 else [249, 115, 22, 225]),
        "radius": 200 if i in (0, len(traj.sightings) - 1) else 140,
    } for i, s in enumerate(traj.sightings) if s.get("lat")])

    layers: list[pdk.Layer] = []
    if legs:
        layers.append(pdk.Layer("PathLayer", data=legs, get_path="path", get_color="colour",
                                get_width="width", width_units=PIXELS, width_min_pixels=3,
                                cap_rounded=True, joint_rounded=True, pickable=True,
                                auto_highlight=True))
    if pts:
        layers.append(pdk.Layer("TextLayer", data=pts, get_position=["lon", "lat"],
                                get_text="label", get_size=22, size_units=PIXELS,
                                get_color=[30, 64, 175], pickable=False))
    if not stops.empty:
        layers += [
            pdk.Layer("ScatterplotLayer", data=stops, get_position=["lon", "lat"],
                      get_radius="radius", get_fill_color="colour", radius_min_pixels=6,
                      stroked=True, get_line_color=[255, 255, 255, 220],
                      line_width_min_pixels=2, pickable=True, auto_highlight=True),
            pdk.Layer("TextLayer", data=stops, get_position=["lon", "lat"], get_text="order",
                      get_size=12, size_units=PIXELS, get_color=[255, 255, 255],
                      get_alignment_baseline="'center'", get_text_anchor="'middle'",
                      pickable=False),
        ]
    return layers, stops


def _arrow(bearing: float) -> str:
    return "→↗↑↖←↙↓↘"[int(((bearing + 22.5) % 360) // 45)] if bearing is not None else "•"


# --- chrome -------------------------------------------------------------
def view(net, zoom: float = 11.2, pitch: float = 40.0) -> pdk.ViewState:
    lats = [c["lat"] for c in net.cameras.values()]
    lons = [c["lon"] for c in net.cameras.values()]
    return pdk.ViewState(latitude=sum(lats) / len(lats), longitude=sum(lons) / len(lons),
                         zoom=zoom, pitch=pitch, bearing=0)


def deck(layers, viewstate, basemap: str = "Light") -> pdk.Deck:
    return pdk.Deck(layers=[l for l in layers if l is not None],
                    initial_view_state=viewstate,
                    map_style=BASEMAPS.get(basemap, BASEMAPS["Light"]),
                    tooltip={"text": "{tip}"})


def legend(kind: str = "network") -> str:
    def chip(rgb, label):
        return (f"<span style='display:inline-flex;align-items:center;margin-right:14px'>"
                f"<span style='width:12px;height:12px;border-radius:3px;margin-right:5px;"
                f"background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'></span>{label}</span>")

    if kind == "network":
        parts = [chip(LEVEL_COLOUR[k], k.capitalize()) for k in
                 ("free", "moderate", "heavy", "severe")]
        parts.append(chip(UNMEASURED, "No traffic measured"))
        parts.append(chip(CAMERA_COLOUR["gateway"], "Gateway camera"))
        parts.append(chip(CAMERA_COLOUR["junction"], "Junction camera"))
    else:
        parts = [chip([22, 163, 74], "First sighting"), chip([249, 115, 22], "En route"),
                 chip([220, 38, 38], "Last sighting"), chip([222, 47, 52], "Implausible leg")]
    return "<div style='font-size:0.82rem;opacity:0.85;margin-top:4px'>" + "".join(parts) + "</div>"
