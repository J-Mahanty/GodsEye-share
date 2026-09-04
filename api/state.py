"""Shared process state for the API: network, OCR engine, simulator, live feed."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

import config
from core import db
from core.network import CameraNetwork, network


class Hub:
    """Fan-out of live sightings and alerts to every connected WebSocket."""

    def __init__(self, maxsize: int = 256):
        self._subs: set[asyncio.Queue] = set()
        self.maxsize = maxsize

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, message: dict) -> None:
        for q in list(self._subs):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(message)

    @property
    def subscribers(self) -> int:
        return len(self._subs)


class AppState:
    def __init__(self):
        self.net: CameraNetwork = network()
        self.hub = Hub()
        self.simulator = None
        self.engine = None
        self.started_at = time.time()
        self.sim_task: asyncio.Task | None = None
        self.live = True
        self.last_tick: float = 0.0
        self.benchmark_cache: dict | None = None

    # --- lazy heavy objects --------------------------------------------
    def get_engine(self):
        if self.engine is None:
            from anpr.ocr import engine
            self.engine = engine()
        return self.engine

    def get_simulator(self):
        if self.simulator is None:
            from sim.city import CitySimulator
            self.simulator = CitySimulator(self.net, engine=self.get_engine())
        return self.simulator

    # --- lifecycle ------------------------------------------------------
    def bootstrap(self) -> None:
        """Create the schema, load cameras, and seed history the first time."""
        with open(config.CAMERA_FILE, encoding="utf-8") as fh:
            spec = json.load(fh)
        db.init(cameras=spec["cameras"])
        stats = db.stats()
        if stats["sightings"] == 0:
            print("[godseye] empty database - seeding 6 hours of city traffic "
                  "(run `python seed.py` yourself for more control)")
            import seed
            seed.seed(hours=6.0, verbose=True)

    async def sim_loop(self) -> None:
        """Advance the simulator in real time and broadcast what it produces."""
        sim = self.get_simulator()
        conn = db.connect()
        while True:
            try:
                if self.live:
                    now = time.time()
                    sightings, fired = await asyncio.to_thread(sim.step, now, None, conn)
                    self.last_tick = now
                    for s in sightings:
                        self.hub.publish({"type": "sighting", "data": s.as_dict()})
                    for a in fired:
                        self.hub.publish({"type": "alert", "data": a})
                await asyncio.sleep(config.SIM_TICK_SECONDS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                     # keep the feed alive
                print(f"[godseye] simulator tick failed: {exc!r}")
                await asyncio.sleep(2.0)

    def status(self) -> dict:
        sim = self.simulator
        return {
            "status": "ok",
            "city": self.net.city,
            "cameras": len(self.net),
            "uptime_s": round(time.time() - self.started_at, 1),
            "live_feed": self.live,
            "last_tick": self.last_tick,
            "subscribers": self.hub.subscribers,
            "ocr_engine_loaded": self.engine is not None,
            "inline_ocr": config.INLINE_OCR,
            "simulator": (sim.stats | {"fleet": len(sim.fleet)}) if sim else None,
            "db": db.stats(),
            "db_path": str(Path(config.DB_PATH)),
        }


state = AppState()
