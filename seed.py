"""Seed the GodsEye database with a demo city day.

    python seed.py                 # 6 hours of traffic, watchlist, scripted incidents
    python seed.py --hours 12 --fleet 400
    python seed.py --ocr-fraction 1.0     # decode every capture with the real engine (slow)

Backfilling matters for the demo: the analytics views need hours of movement
before origin-destination flows and bottlenecks mean anything, and nobody wants
to watch six hours of live feed at a judging table.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import config
from core import alerts as alert_rules
from core import db
from core.network import network


WATCH_REASONS = [
    ("stolen vehicle - FIR 214/2026, Bidhannagar PS", "critical"),
    ("suspect vehicle - narcotics watch, Lalbazar STF", "critical"),
    ("outstanding e-challans above Rs 25,000", "medium"),
    ("expired fitness certificate - commercial", "medium"),
    ("BOLO - missing person case 88/2026, Jadavpur PS", "high"),
]


def seed(hours: float = 6.0, fleet: int = config.FLEET_SIZE, ocr_fraction: float = 0.05,
         watchlist_size: int = 5, incidents: bool = True, reset: bool = True,
         seed_value: int = 11, verbose: bool = True) -> dict:
    from sim.city import CitySimulator

    t0 = time.time()
    with open(config.CAMERA_FILE, encoding="utf-8") as fh:
        spec = json.load(fh)
    conn = db.init(cameras=spec["cameras"])
    if reset:
        db.reset()

    net = network()
    sim = CitySimulator(net, seed=seed_value, fleet_size=fleet)
    if verbose:
        print(f"[seed] {len(net)} cameras, {fleet} vehicles, {hours:g}h of traffic "
              f"({ocr_fraction:.0%} of captures decoded by the real OCR engine)")
    stored = sim.history(hours=hours, ocr_fraction=ocr_fraction, conn=conn, verbose=verbose)

    rng = random.Random(seed_value)
    watched = []
    # Watchlist plates are chosen from vehicles that actually moved, so the demo
    # has real trajectories to show rather than empty search results.
    active = db.rows("SELECT plate, COUNT(*) n FROM sightings GROUP BY plate"
                     " HAVING n >= 6 ORDER BY n DESC LIMIT 60", conn=conn)
    for i, row in enumerate(rng.sample(active, min(watchlist_size, len(active)))):
        reason, severity = WATCH_REASONS[i % len(WATCH_REASONS)]
        db.add_watch(row["plate"], reason, severity, conn=conn)
        watched.append(row["plate"])
    if verbose and watched:
        print(f"[seed] watchlist: {', '.join(watched)}")

    if verbose:
        print("[seed] replaying history through the alert rules ...")
    fired = alert_rules.backfill(hours=min(hours, 6.0), conn=conn, net=net)

    injected = {}
    if incidents:
        injected["clone"] = sim.inject_clone(conn=conn)
        injected["loiterer"] = sim.inject_loiterer(conn=conn)
        if verbose:
            print(f"[seed] injected cloned plate {injected['clone']} and "
                  f"loiterer {injected['loiterer']}")

    stats = db.stats(conn=conn)
    if verbose:
        print(f"[seed] done in {time.time()-t0:.1f}s - {stats['sightings']} sightings, "
              f"{stats['unique_plates']} plates, {stats['alerts']} alerts")
    return {"stored": stored, "alerts": fired, "watchlist": watched,
            "injected": injected, "stats": stats}


def main():
    ap = argparse.ArgumentParser(description="Seed the GodsEye demo database")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--fleet", type=int, default=config.FLEET_SIZE)
    ap.add_argument("--ocr-fraction", type=float, default=0.05,
                    help="share of captures decoded by the real engine (rest use the "
                         "calibrated error model)")
    ap.add_argument("--watchlist", type=int, default=5)
    ap.add_argument("--no-incidents", action="store_true")
    ap.add_argument("--keep", action="store_true", help="append instead of wiping")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    seed(hours=args.hours, fleet=args.fleet, ocr_fraction=args.ocr_fraction,
         watchlist_size=args.watchlist, incidents=not args.no_incidents,
         reset=not args.keep, seed_value=args.seed)


if __name__ == "__main__":
    main()
