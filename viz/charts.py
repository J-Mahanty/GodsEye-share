"""Render the judge-facing chart set from the platform's own measurements.

    python -m viz.charts              # everything, into charts/
    python -m viz.charts --no-db      # skip the four that need a seeded database
    python -m viz.charts 1 5 9        # only these

Every chart is drawn from a file in the repo. Nothing here is typed in by hand
except the README_* blocks below, which carry a line reference to the table they
were copied from -- those tables have no JSON behind them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from viz import theme

OUT = config.FIGURE_DIR


# --- numbers with no JSON behind them -----------------------------------
# Copied from the README tables named in each comment. Kept in one block, with
# provenance, so a judge's question about any of them traces to a source.

# README.md:462-468 -- exact match on real plate crops, by source.
README_REAL_BACKENDS = {
    "sources": [("xml", 25), ("car", 13), ("video", 494), ("all", 532)],
    "rows": [
        ("classical (synthetic-trained)", [0.160, 0.077, 0.045, 0.051]),
        ("CRNN (synthetic-trained)",      [0.000, 0.000, 0.010, 0.009]),
        ("EasyOCR",                       [0.160, 0.154, 0.215, 0.211]),
        ("anpr_ocr, shipped upstream",    [0.440, 0.923, 0.810, 0.795]),
        ("anpr_ocr + this repo's decoder",[0.680, 0.923, 0.921, 0.910]),
    ],
}

# README.md:478-481 -- held out by vehicle, 8 splits, before/after re-labelling
# and the decoder work.
README_BEFORE_AFTER = {"before": (0.538, 0.347, 0.811), "after": (0.938, 0.868, 0.971)}

# README.md:498-505 -- the ground-truth repair. The decoder was UNCHANGED across
# these two numbers; only the labels moved.
README_LABEL_FIX = {"before": 0.487, "after": 0.795, "mislabelled": 0.37,
                    "frames": 548, "relabelled": 494, "plates": 84}

# README.md:541-547 -- whole photographs end to end, 8 splits.
README_WHOLE_PHOTO = [
    ("classical",            0.041, 0.000, 0.151),
    ("CRNN",                 0.031, 0.000, 0.053),
    ("EasyOCR",              0.129, 0.028, 0.368),
    ("YOLOv8-OBB + anpr_ocr", 0.915, 0.900, 0.948),
]

# yolo_accuracy.txt
YOLO_VAL = {"val (78 imgs)": (0.999, 0.987, 0.994, 0.875),
            "test (59 imgs)": (0.999, 1.000, 0.995, 0.875)}


def _benchmark() -> dict:
    return json.loads((config.MODEL_DIR / "benchmark.json").read_text())


def _pct(ax, axis="x"):
    fmt = PercentFormatter(xmax=1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(config.ROOT)}")
    return path


# --- 1-4 the reader, from models/benchmark.json --------------------------
def chart_01_burst_vs_single():
    b = _benchmark()
    rows = sorted(b["conditions"], key=lambda c: c["burst_accuracy"])
    names = [c["condition"] for c in rows]
    single = [c["plate_accuracy"] for c in rows]
    burst = [c["burst_accuracy"] for c in rows]
    y = np.arange(len(rows))
    h = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.barh(y + h / 2, burst, h, label=f"burst of {b['burst_frames']}",
            color=theme.CATEGORICAL[0])
    ax.barh(y - h / 2, single, h, label="single frame",
            color=theme.CATEGORICAL[0], alpha=0.38)

    for yi, v in zip(y + h / 2, burst):
        ax.text(v + .012, yi, f"{v:.0%}", va="center", fontsize=8.5,
                color=theme.INK, fontweight="bold")
    for yi, v in zip(y - h / 2, single):
        ax.text(v + .012, yi, f"{v:.0%}", va="center", fontsize=8.5,
                color=theme.MUTED)

    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.08)
    _pct(ax)
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    theme.headline(ax, "Reading a burst, not a photograph",
                   f"Exact whole-registration match, {b['samples_per_condition']} plates "
                       f"per condition. Overall {b['overall']['plate_accuracy']:.1%} on one "
                       f"frame, {b['overall']['burst_accuracy']:.1%} on {b['burst_frames']}.")
    ax.legend(loc="lower right")
    theme.source(fig, "models/benchmark.json  ·  python -m anpr.benchmark --samples 100 --burst 5")
    return _save(fig, "01_burst_vs_single.png")


def chart_02_stored_vs_refused():
    b = _benchmark()
    rows = sorted(b["conditions"], key=lambda c: c["accepted"] * c["accepted_accuracy"])
    names = [c["condition"] for c in rows]
    n = b["samples_per_condition"]
    right = np.array([c["accepted"] * c["accepted_accuracy"] for c in rows])
    wrong = np.array([c["accepted"] * (1 - c["accepted_accuracy"]) for c in rows])
    refused = np.array([n - c["accepted"] for c in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    kw = dict(height=0.62, edgecolor=theme.PAPER, linewidth=1.6)
    ax.barh(y, right, color=theme.GOOD, label="stored, correct", **kw)
    ax.barh(y, wrong, left=right, color=theme.WRONG, label="stored, wrong", **kw)
    ax.barh(y, refused, left=right + wrong, color=theme.NEUTRAL,
            label="refused below the confidence floor", **kw)

    for yi, r, w in zip(y, right, wrong):
        if r > 8:
            ax.text(r / 2, yi, f"{r:.0f}", va="center", ha="center",
                    fontsize=8.5, color="white", fontweight="bold")
        if w >= 3:
            ax.text(r + w / 2, yi, f"{w:.0f}", va="center", ha="center",
                    fontsize=8, color="white")

    ax.set_yticks(y, names)
    ax.set_xlim(0, n)
    ax.set_xlabel(f"captures (of {n} per condition)")
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    o = b["overall"]
    theme.headline(ax, "A refused read is not a wrong read",
                   f"Of {o['samples']:,} captures the engine stores {o['accepted']}, "
                       f"and {o['accepted_accuracy']:.1%} of those are exactly right — "
                       f"the number an operator actually experiences.")
    theme.legend_below(ax, fig, 3)
    theme.source(fig, "models/benchmark.json  ·  confidence floor: config.MIN_PLATE_CONFIDENCE_CRNN", y=-0.085)
    return _save(fig, "02_stored_vs_refused.png")


def chart_03_accuracy_funnel():
    b = _benchmark()
    o = b["overall"]
    total = o["samples"]
    stored = o["accepted"]
    correct = stored * o["accepted_accuracy"]
    stages = [("captured", total, theme.NEUTRAL),
              ("confident enough to store", stored, theme.CATEGORICAL[0]),
              ("exactly right", correct, theme.GOOD)]

    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    y = np.arange(len(stages))[::-1]
    for yi, (label, value, colour) in zip(y, stages):
        ax.barh(yi, value, height=0.55, color=colour)
        ax.text(value + total * .012, yi, f"{value:,.0f}", va="center",
                fontsize=12, fontweight="bold", color=theme.INK)
    ax.text(stored + total * .012, y[1] - 0.34, f"{stored / total:.1%} of captures",
            va="center", fontsize=8.5, color=theme.MUTED)
    ax.text(correct + total * .012, y[2] - 0.34,
            f"{correct / stored:.1%} of what is stored", va="center",
            fontsize=8.5, color=theme.MUTED)

    ax.set_yticks(y, [s[0] for s in stages])
    ax.set_xlim(0, total * 1.14)
    ax.grid(axis="y", visible=False)
    ax.set_xticks([])
    theme.despine(ax, keep=())
    theme.headline(ax, "What reaches the operator",
                   "The platform discards what it cannot read rather than showing a guess.")
    theme.source(fig, "models/benchmark.json  ·  1,000 synthetic captures, 100 per condition")
    return _save(fig, "03_accuracy_funnel.png")


def chart_04_layout_difficulty():
    b = _benchmark()
    items = sorted(b["layouts"].items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.bar(names, vals, width=0.55, color=theme.CATEGORICAL[0])
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + .012, f"{v:.0%}",
                ha="center", fontsize=10, fontweight="bold", color=theme.INK)
    ax.set_ylim(0, max(vals) * 1.28)
    _pct(ax, "y")
    ax.grid(axis="x", visible=False)
    theme.despine(ax, keep=("left",))
    theme.headline(ax, "Where the plate itself is the problem",
                   "Single frame, 100 plates each. Grime and a second row cost more than the weather.")
    theme.source(fig, "models/benchmark.json  ·  python -m anpr.benchmark --layouts 100")
    return _save(fig, "04_layout_difficulty.png")


# --- 5-8 real photographs -----------------------------------------------
def chart_05_backends_real():
    src = README_REAL_BACKENDS["sources"]
    rows = README_REAL_BACKENDS["rows"]
    y = np.arange(len(rows))
    h = 0.19

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for i, (label, n) in enumerate(src):
        vals = [r[1][i] for r in rows]
        off = (i - (len(src) - 1) / 2) * h
        ax.barh(y + off, vals, h * 0.9, label=f"{label}  (n={n})",
                color=theme.CATEGORICAL[i])
        if label == "all":
            for yi, v in zip(y + off, vals):
                ax.text(v + .01, yi, f"{v:.1%}", va="center", fontsize=8.5,
                        fontweight="bold", color=theme.INK)

    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.06)
    _pct(ax)
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    theme.headline(ax, "On real photographs, synthetic training does not transfer",
                   "532 real Indian plate crops. The CRNN trained on our own camera model reads "
                       "0.9%.\nSwapping to a recogniser pretrained on real plates, then fixing its "
                       "decoder, reaches 91.0%.")
    theme.legend_below(ax, fig, 4)
    theme.source(fig, "README.md:462-468  ·  python -m anpr.compare_backends --real-all", y=-0.075)
    return _save(fig, "05_backends_real.png")


def chart_06_seed_stability():
    raw = json.loads((config.MODEL_DIR / "backend_comparison_real.json").read_text())
    seeds = sorted(raw, key=int)
    backends = list(raw[seeds[0]])
    stats = []
    for b in backends:
        vals = np.array([raw[s][b] for s in seeds], float)
        stats.append((b, vals.mean(), vals.min(), vals.max(), vals))
    stats.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(stats))
    for yi, (name, mean, lo, hi, vals) in zip(y, stats):
        colour = theme.CATEGORICAL[0] if name == "anpr_ocr" else theme.NEUTRAL
        ax.hlines(yi, lo, hi, color=colour, linewidth=2.5, alpha=.45)
        ax.plot(vals, [yi] * len(vals), "o", ms=5, color=colour, alpha=.55,
                markeredgecolor=theme.PAPER, markeredgewidth=1.2)
        ax.plot(mean, yi, "D", ms=8, color=colour,
                markeredgecolor=theme.PAPER, markeredgewidth=1.5)
        ax.text(hi + .018, yi, f"mean {mean:.1%}   spread {(hi - lo) * 100:.1f} pts",
                va="center", fontsize=8.5,
                color=theme.INK if name == "anpr_ocr" else theme.MUTED,
                fontweight="bold" if name == "anpr_ocr" else "normal")

    ax.set_yticks(y, [s[0] for s in stats])
    ax.set_xlim(-0.02, 1.32)
    ax.set_xticks(np.arange(0, 1.01, .2))
    _pct(ax)
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    theme.headline(ax, "One seed is not a number",
                   f"Each dot is one of {len(seeds)} vehicle-held-out splits; the diamond is the mean. "
                       "With ~118 vehicles,\nwhich ones land in validation moves the result more than a "
                       "small code change does.")
    theme.source(fig, "models/backend_comparison_real.json  ·  python -m anpr.compare_backends --real --seeds 8")
    return _save(fig, "06_seed_stability.png")


def chart_07_label_fix():
    d = README_LABEL_FIX
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    names = ["cluster-propagated labels", "re-labelled frame by frame"]
    vals = [d["before"], d["after"]]
    colours = [theme.NEUTRAL, theme.CATEGORICAL[0]]
    bars = ax.barh(names[::-1], vals[::-1], height=0.5, color=colours[::-1])
    for rect, v in zip(bars, vals[::-1]):
        ax.text(v + .012, rect.get_y() + rect.get_height() / 2, f"{v:.1%}",
                va="center", fontsize=12, fontweight="bold", color=theme.INK)
    ax.annotate(f"+{(d['after'] - d['before']) * 100:.1f} points, decoder unchanged",
                xy=(d["after"], 1), xytext=(d["before"] + .02, 1.42),
                fontsize=9, color=theme.MUTED)
    ax.set_xlim(0, 1.0)
    _pct(ax)
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    theme.headline(ax, "The ground truth was wrong",
                   f"{d['mislabelled']:.0%} of {d['frames']} dashcam frames carried a plate that is "
                       f"not in the picture — adjacency\nis not vehicle identity. "
                       f"{d['relabelled']} frames re-read by eye, {d['plates']} distinct plates.")
    theme.source(fig, "README.md:498-505, REAL_LABELS.md  ·  real_plate_frame_labels.json")
    return _save(fig, "07_label_fix.png")


def chart_08_whole_photo_e2e():
    rows = sorted(README_WHOLE_PHOTO, key=lambda r: r[1])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for yi, (name, mean, lo, hi) in zip(y, rows):
        best = name.startswith("YOLO")
        colour = theme.CATEGORICAL[0] if best else theme.NEUTRAL
        ax.hlines(yi, lo, hi, color=colour, linewidth=2.5, alpha=.45)
        ax.plot([lo, hi], [yi, yi], "|", ms=10, color=colour)
        ax.plot(mean, yi, "D", ms=8, color=colour,
                markeredgecolor=theme.PAPER, markeredgewidth=1.5)
        ax.text(hi + .018, yi, f"mean {mean:.1%}   spread {(hi - lo) * 100:.1f} pts",
                va="center", fontsize=8.5,
                color=theme.INK if best else theme.MUTED,
                fontweight="bold" if best else "normal")

    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-0.02, 1.32)
    ax.set_xticks(np.arange(0, 1.01, .2))
    _pct(ax)
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    theme.headline(ax, "End to end on whole photographs",
                   "Detector finds the plate, recogniser reads it — the path the platform deploys. "
                       "8 splits.\nEvery seed clears 90%, and the spread is the tightest of any mode.")
    theme.source(fig, "README.md:541-547  ·  python -m anpr.compare_backends --real-frames --seeds 8")
    return _save(fig, "08_whole_photo_e2e.png")


# --- 9-12 the platform, from the seeded database ------------------------
def chart_09_camera_network():
    from core import analytics, network as net_mod

    net = net_mod.network()
    spec = json.loads(config.CAMERA_FILE.read_text())
    density = analytics.camera_density(minutes=360)
    flows = analytics.link_flows(minutes=360, net=net)

    load = {r["camera_id"]: r["count"] for _, r in density.iterrows()} if not density.empty else {}
    level = {}
    if not flows.empty:
        for _, r in flows.iterrows():
            key = frozenset((r["from_camera"], r["to_camera"]))
            level[key] = max(level.get(key, ""), r["level"], key=lambda l: list(theme.LEVEL).index(l) if l in theme.LEVEL else -1)

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    for link in spec["links"]:
        pts = net.edge_geometry(link["a"], link["b"])
        lat = [p[0] for p in pts]
        lon = [p[1] for p in pts]
        lv = level.get(frozenset((link["a"], link["b"])))
        ax.plot(lon, lat, "-", linewidth=2.6 if lv else 1.4,
                color=theme.LEVEL.get(lv, "#c8d2dd"),
                solid_capstyle="round", zorder=1 if not lv else 2)

    cams = spec["cameras"]
    peak = max(load.values()) if load else 1
    for c in cams:
        n = load.get(c["id"], 0)
        size = 22 + 210 * (n / peak)
        gateway = c["type"] == "gateway"
        ax.scatter(c["lon"], c["lat"], s=size, zorder=3,
                   color=theme.CATEGORICAL[3] if gateway else theme.CATEGORICAL[0],
                   edgecolor=theme.PAPER, linewidth=1.4)

    busiest = sorted(cams, key=lambda c: -load.get(c["id"], 0))[:5]
    for c in busiest:
        ax.annotate(f"{c['name']}  {load.get(c['id'], 0):,}",
                    xy=(c["lon"], c["lat"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=theme.INK)

    handles = [plt.Line2D([], [], marker="o", ls="", color=theme.CATEGORICAL[0],
                          ms=7, label="junction camera"),
               plt.Line2D([], [], marker="o", ls="", color=theme.CATEGORICAL[3],
                          ms=7, label="gateway camera")]
    handles += [plt.Line2D([], [], color=theme.LEVEL[l], lw=2.6, label=l)
                for l in ("free", "moderate", "heavy", "severe") if l in level.values()]
    ax.legend(handles=handles, loc="lower left", ncols=2)

    ax.set_aspect(1 / np.cos(np.radians(config.CITY_CENTER[0])))
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    theme.despine(ax, keep=())
    theme.headline(ax, f"{len(cams)} cameras, {len(spec['links'])} road links — {config.CITY_NAME}",
                   "Real junction coordinates; roads drawn from their own shape points, "
                       "so a corridor bends the way it bends.\nCamera size is read volume over "
                       "the seeded 6 hours; link colour is measured congestion.")
    theme.source(fig, "data/cameras.json + data/godseye.db  ·  core.analytics.camera_density / link_flows")
    return _save(fig, "09_camera_network.png")


def chart_10_alerts_by_rule():
    from core import alerts as alerts_mod

    # alerts.summary() anchors its window to time.time(); a seeded database is
    # usually older than that, so anchor to the data's own last alert instead.
    rows = alerts_mod.db.alerts(limit=5000)
    if not rows:
        print("  skipped 10: no alerts (python seed.py)")
        return None
    latest = max(r["ts"] for r in rows)
    hours = 6.0
    rows = [r for r in rows if r["ts"] >= latest - hours * 3600]

    by_kind, sev_of = {}, {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        sev_of.setdefault(r["kind"], r["severity"])
    items = sorted(by_kind.items(), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colours = [theme.SEVERITY.get(sev_of.get(k, "low"), theme.NEUTRAL) for k in names]
    bars = ax.barh(names, vals, height=0.55, color=colours)
    for rect, v in zip(bars, vals):
        ax.text(v + max(vals) * .012, rect.get_y() + rect.get_height() / 2, f"{v}",
                va="center", fontsize=10, fontweight="bold", color=theme.INK)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=8, color=c, label=k)
               for k, c in theme.SEVERITY.items()
               if c in colours]
    ax.legend(handles=handles, loc="lower right", title="severity",
              title_fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.14)
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    theme.headline(ax, f"{len(rows)} alerts raised over {hours:.0f} hours",
                   "Every sighting is checked as it lands. Alerts are de-duplicated per plate "
                       "per rule\nwithin a rolling window, so a listed plate seen all day is not one alert.")
    theme.source(fig, "data/godseye.db  ·  core.alerts.summary(hours=6)")
    return _save(fig, "10_alerts_by_rule.png")


def chart_11_traffic_6h():
    from core import analytics

    flow = analytics.flow_trend(hours=6.0, bucket_min=5.0)
    if flow.empty:
        print("  skipped 11: no sightings")
        return None

    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.plot(flow["bucket"], flow["sightings"], lw=2,
            color=theme.CATEGORICAL[0], label="reads")
    ax.plot(flow["bucket"], flow["unique_plates"], lw=2,
            color=theme.CATEGORICAL[1], label="distinct vehicles")
    ax.fill_between(flow["bucket"], flow["sightings"], alpha=.10,
                    color=theme.CATEGORICAL[0])

    peak = flow.loc[flow["sightings"].idxmax()]
    ax.annotate(f"peak {int(peak['sightings'])} reads",
                xy=(peak["bucket"], peak["sightings"]), xytext=(6, 8),
                textcoords="offset points", fontsize=9,
                fontweight="bold", color=theme.INK)

    ax.set_ylabel("per 5-minute bucket")
    ax.grid(axis="x", visible=False)
    theme.despine(ax, keep=("left", "bottom"))
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    theme.headline(ax, "Six hours of city traffic",
                   "Both series are counts on one axis. The gap between them is how often the "
                       "same vehicle\nis seen twice — the signal every trajectory and O-D figure is built on.")
    theme.source(fig, "data/godseye.db  ·  core.analytics.flow_trend(hours=6, bucket_min=5)")
    return _save(fig, "11_traffic_6h.png")


def chart_12_bottlenecks():
    from core import analytics

    bn = analytics.bottlenecks(minutes=360, top=10)
    if bn.empty:
        print("  skipped 12: no measured links")
        return None
    bn = bn.sort_values("delay_min_total")

    # A corridor appears once per direction; without the bearing the two rows
    # look like a duplicate, and inbound/outbound is exactly the distinction
    # the control room exists to show.
    labels = [f"{r['name']}  {r['direction']}" for _, r in bn.iterrows()]
    vals = bn["delay_min_total"].tolist()
    colours = [theme.LEVEL.get(l, theme.NEUTRAL) for l in bn["level"]]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bars = ax.barh(np.arange(len(bn)), vals, height=0.6, color=colours)
    for i, (rect, (_, r)) in enumerate(zip(bars, bn.iterrows())):
        ax.text(r["delay_min_total"] + max(vals) * .012,
                rect.get_y() + rect.get_height() / 2,
                f"{r['delay_min_total']:,.0f} veh-min   "
                f"({r['vehicles']:.0f} veh @ {r['median_kmph']:.0f} of {r['free_kmph']:.0f} km/h)",
                va="center", fontsize=8, color=theme.MUTED)

    ax.set_yticks(np.arange(len(bn)), labels)
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("vehicle-minutes of delay caused")
    ax.grid(axis="y", visible=False)
    theme.despine(ax, keep=("bottom",))
    levels = set(bn["level"])
    if len(levels) > 1:
        handles = [plt.Line2D([], [], marker="s", ls="", ms=8, color=theme.LEVEL[l], label=l)
                   for l in ("free", "moderate", "heavy", "severe") if l in levels]
        ax.legend(handles=handles, loc="lower right", title="congestion",
                  title_fontsize=8.5)
    else:
        only = next(iter(levels))
        ax.annotate(f"every link in this top 10 is {only}", xy=(1, 0),
                    xytext=(0, -42), xycoords="axes fraction",
                    textcoords="offset points", ha="right", fontsize=8.5,
                    color=theme.LEVEL.get(only, theme.MUTED))
    theme.headline(ax, "Ranked by delay caused, not by slowness",
                   "A slow empty road is not a problem. Volume times delay per vehicle is what "
                       "an operator\nshould be sent to, and it reorders the list against speed alone.")
    theme.source(fig, "data/godseye.db  ·  core.analytics.bottlenecks(top=10)")
    return _save(fig, "12_bottlenecks.png")


# --- 13 the detector -----------------------------------------------------
def chart_13_yolo_training():
    runs = [("run 1", config.ROOT / "runs/obb/runs/yolov8n-obb-plate-model/results.csv"),
            ("run 2", config.ROOT / "runs/obb/runs/yolov8n-obb-plate-model-2/results.csv"),
            ("run 3, shipped", config.ROOT / "runs/obb/runs/yolov8n-obb-plate-model-3/results.csv")]
    runs = [(n, p) for n, p in runs if p.exists()]
    if not runs:
        print("  skipped 13: runs/obb not present")
        return None

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for i, (name, path) in enumerate(runs):
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        ep = [float(r["epoch"]) for r in rows]
        m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        last = i == len(runs) - 1
        ax.plot(ep, m, lw=2.2 if last else 1.6,
                color=theme.CATEGORICAL[i], alpha=1.0 if last else .65,
                label=f"{name}  |  final {m[-1]:.3f}")
        ax.plot(ep[-1], m[-1], "o", ms=6, color=theme.CATEGORICAL[i],
                markeredgecolor=theme.PAPER, markeredgewidth=1.4)

    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP50-95")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    theme.despine(ax, keep=("left", "bottom"))
    ax.legend(loc="lower right")
    theme.headline(ax, "Fine-tuning the plate detector on real photographs",
                   "Each run resumes from the previous checkpoint. Treat the absolute value with "
                       "caution:\nthe split is image-level, so near-identical video frames leak across it.")
    theme.source(fig, "runs/obb/runs/*/results.csv  ·  python train_yolo.py  ·  see CLAUDE.md on split leakage")
    return _save(fig, "13_yolo_training.png")


CHARTS = [
    (1, chart_01_burst_vs_single, False),
    (2, chart_02_stored_vs_refused, False),
    (3, chart_03_accuracy_funnel, False),
    (4, chart_04_layout_difficulty, False),
    (5, chart_05_backends_real, False),
    (6, chart_06_seed_stability, False),
    (7, chart_07_label_fix, False),
    (8, chart_08_whole_photo_e2e, False),
    (9, chart_09_camera_network, True),
    (10, chart_10_alerts_by_rule, True),
    (11, chart_11_traffic_6h, True),
    (12, chart_12_bottlenecks, True),
    (13, chart_13_yolo_training, False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the judge-facing chart set.")
    ap.add_argument("only", nargs="*", type=int,
                    help="chart numbers to render (default: all)")
    ap.add_argument("--no-db", action="store_true",
                    help="skip the charts that need a seeded database")
    args = ap.parse_args()

    theme.apply()
    wanted = set(args.only) if args.only else None
    have_db = config.DB_PATH.exists() and not args.no_db
    if not have_db and not args.no_db:
        print(f"note: {config.DB_PATH.name} not found — run `python seed.py` for charts 9-12")

    made, skipped = 0, 0
    for number, fn, needs_db in CHARTS:
        if wanted and number not in wanted:
            continue
        if needs_db and not have_db:
            print(f"  skipped {number:2d}: needs a seeded database (python seed.py)")
            skipped += 1
            continue
        try:
            if fn() is not None:
                made += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  FAILED {number:2d}: {type(exc).__name__}: {exc}")
            skipped += 1

    print(f"\n{made} chart{'s' if made != 1 else ''} in {OUT.relative_to(config.ROOT)}/"
          + (f", {skipped} skipped" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
