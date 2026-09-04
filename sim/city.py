"""City traffic simulator — the stand-in for a live ANPR camera network.

A fleet of vehicles is routed across the camera graph on realistic origin-
destination demand, with a time-of-day profile and standing congestion on the
corridors that are actually congested in Kolkata. Every time a vehicle passes
a camera the simulator produces a *plate image*, degrades it for the conditions
at that camera at that hour, and pushes it through the real OCR engine. What
lands in the database is therefore what the engine read - errors included -
which is the point: the trajectory and analytics layers have to cope with
imperfect reads exactly as they would in deployment.

Each camera contributes a *burst* per vehicle, not a single frame, which is what
a loop-triggered ANPR node actually does; the frames are read independently and
voted. That costs roughly `BURST_FRAMES` x 75 ms per vehicle, so live mode caps
how many frames are decoded per tick and falls back to a calibrated error model
for the rest; `ocr_mode` on each sighting records which path produced it.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import config
from anpr import plates
from core import alerts as alert_rules
from core import db
from core.network import CameraNetwork, network

VEHICLE_CLASSES = [
    ("car", 0.52), ("two_wheeler", 0.24), ("auto", 0.09),
    ("taxi", 0.07), ("bus", 0.03), ("truck", 0.05),
]

# Corridors that are genuinely congested in Kolkata; multiplies free-flow time.
STANDING_CONGESTION = {
    ("CAM-07", "CAM-15"): 2.6,   # Ultadanga - Chingrighata, the city's worst
    ("CAM-24", "CAM-25"): 2.5,   # Howrah Bridge
    ("CAM-04", "CAM-07"): 2.2,   # Shyambazar - Kankurgachi
    ("CAM-22", "CAM-27"): 2.2,   # Park Circus - Gariahat
    ("CAM-20", "CAM-24"): 2.0,   # Strand Road into the bridge
    ("CAM-18", "CAM-19"): 2.0,   # Sealdah approach
    ("CAM-27", "CAM-28"): 1.9,   # Rashbehari Avenue
    ("CAM-01", "CAM-04"): 1.8,   # BT Road
    ("CAM-15", "CAM-16"): 1.7,   # EM Bypass mid
    ("CAM-08", "CAM-07"): 1.7,   # VIP Road into Ultadanga
}

# Hour-of-day demand multiplier (index 0 = midnight).
HOUR_PROFILE = [0.18, 0.12, 0.10, 0.11, 0.16, 0.32, 0.58, 0.86, 1.00, 0.94,
                0.78, 0.72, 0.74, 0.76, 0.74, 0.79, 0.90, 1.00, 0.98, 0.84,
                0.62, 0.48, 0.36, 0.25]

# Plate-read accuracy per capture condition, measured by anpr.benchmark on the
# burst path the platform actually runs (config.BURST_FRAMES frames per vehicle,
# fused by CTC score). Used only for captures the live loop cannot afford to
# decode for real.
CONDITION_ACCURACY = {
    "daylight": 1.00,
    "night_ir": 1.00,
    "night_glare": 0.87,
    "monsoon": 0.98,
    "fog": 0.83,
    "high_speed": 0.80,
    "far_lane": 0.67,
    "dusk_highiso": 0.89,
    "cheap_cam": 0.27,
    "storm": 0.00,
}


@dataclass
class Vehicle:
    plate: str
    vehicle_class: str
    commercial: bool
    route: list[str]
    leg: int = 0
    next_ts: float = 0.0
    pace: float = 1.0                    # <1 drives faster than the crowd
    speeder: bool = False                # a small share of drivers really do
    trips: int = 0

    @property
    def camera(self) -> str:
        return self.route[self.leg]

    @property
    def next_camera(self) -> str | None:
        return self.route[self.leg + 1] if self.leg + 1 < len(self.route) else None


@dataclass
class Capture:
    """One camera crossing before OCR."""
    ts: float
    camera_id: str
    true_plate: str
    vehicle_class: str
    commercial: bool
    speed_kmph: float
    lane: int
    heading: float
    direction: str
    next_camera: str | None
    condition: str


class CitySimulator:
    def __init__(self, net: CameraNetwork | None = None, seed: int = 11,
                 fleet_size: int = config.FLEET_SIZE, engine=None, weather: str = "auto"):
        self.net = net or network()
        self.rng = random.Random(seed)
        self.weather = weather
        self.engine = engine
        self.fleet: list[Vehicle] = []
        self.cameras = list(self.net.cameras)
        self.gateways = self.net.gateways or self.cameras
        self.stats = {"captures": 0, "reads_ocr": 0, "reads_modelled": 0,
                      "frames_decoded": 0, "dropped_low_confidence": 0, "misreads": 0,
                      "evidence_kept": 0}
        self._schema_ready = False
        for _ in range(fleet_size):
            self.fleet.append(self._spawn(time.time()))

    # --- fleet ---------------------------------------------------------
    def _spawn(self, now: float, plate: str | None = None) -> Vehicle:
        vclass = self.rng.choices([c for c, _ in VEHICLE_CLASSES],
                                  [w for _, w in VEHICLE_CLASSES])[0]
        commercial = vclass in ("bus", "truck", "taxi", "auto")
        route = self._route(now)
        return Vehicle(plate=plate or plates.random_plate(self.rng), vehicle_class=vclass,
                       commercial=commercial, route=route,
                       next_ts=now + self.rng.uniform(0, 900),
                       pace=self.rng.uniform(0.85, 1.3),
                       speeder=self.rng.random() < 0.05)

    def _route(self, now: float, origin: str | None = None) -> list[str]:
        """Pick a destination and route to it.

        A vehicle's next trip starts where its last one ended - anything else
        teleports the plate across the city and manufactures exactly the
        impossible-speed pattern the alert rules exist to catch. Trips start or
        end at a gateway most of the time: city traffic is largely through
        traffic and commuting from the edges.
        """
        for _ in range(12):
            a = origin or (self.rng.choice(self.gateways) if self.rng.random() < 0.6
                           else self.rng.choice(self.cameras))
            b = (self.rng.choice(self.gateways) if origin and self.rng.random() < 0.45
                 else self.rng.choice(self.cameras))
            if a == b:
                continue
            route = self.net.route(a, b)
            if len(route) >= 3:
                return list(route)
        return list(self.net.route(origin or self.cameras[0], self.cameras[-1]))

    # --- movement ------------------------------------------------------
    def _demand(self, ts: float) -> float:
        hour = time.localtime(ts).tm_hour
        return HOUR_PROFILE[hour]

    def _congestion(self, a: str, b: str, ts: float) -> float:
        """Travel-time multiplier on a link: standing congestion x rush hour."""
        base = STANDING_CONGESTION.get((a, b)) or STANDING_CONGESTION.get((b, a)) or 1.0
        rush = 0.75 + 0.85 * self._demand(ts)
        return base * rush * self.rng.uniform(0.85, 1.25)

    def _condition(self, ts: float) -> str:
        """Which camera scenario this crossing is.

        A site is what it is - the mounting and the lens do not change - but the
        hour and the weather decide which regime it is working in, and roughly a
        fifth of crossings are on the far carriageway or through an older camera.
        """
        hour = time.localtime(ts).tm_hour
        night = hour >= 19 or hour <= 5
        dusk = hour in (18, 19, 6)
        wet = self.weather == "rain" or (self.weather == "auto" and self.rng.random() < 0.12)
        r = self.rng.random()
        if wet:
            return "storm" if r < 0.15 else "monsoon"
        if night:
            return self.rng.choices(["night_ir", "night_glare", "cheap_cam"],
                                    [0.62, 0.28, 0.10])[0]
        if dusk:
            return self.rng.choices(["dusk_highiso", "night_ir", "daylight"],
                                    [0.55, 0.20, 0.25])[0]
        if self.weather == "auto" and r < 0.07:
            return "fog"
        return self.rng.choices(
            ["daylight", "high_speed", "far_lane", "cheap_cam"],
            [0.62, 0.16, 0.14, 0.08])[0]

    def advance(self, until_ts: float, max_captures: int = 5000) -> list[Capture]:
        """Move every vehicle up to `until_ts`, returning the camera crossings."""
        out: list[Capture] = []
        for v in self.fleet:
            while v.next_ts <= until_ts and len(out) < max_captures:
                nxt = v.next_camera
                ts = v.next_ts
                km = self.net.link_km(v.camera, nxt) if nxt else 0.0
                free = self.net.link_free_kmph(v.camera, nxt) if nxt else 40.0
                if nxt:
                    factor = self._congestion(v.camera, nxt, ts) * v.pace
                    # Nobody outruns the road by more than a third, however
                    # empty it is - except the ~5% who genuinely speed, which is
                    # what gives the enforcement rule something real to catch.
                    ceiling = free * (2.1 if v.speeder else 1.35)
                    kmph = min(max(6.0, free / factor), ceiling)
                    if v.speeder:
                        kmph = max(kmph, free * self.rng.uniform(1.0, 1.9))
                    heading = self.net.heading(v.camera, nxt)
                    direction = self.net.direction(v.camera, nxt)
                    travel_s = km / kmph * 3600.0
                else:
                    kmph, heading, direction, travel_s = free * 0.6, 0.0, "-", 0.0

                out.append(Capture(
                    ts=ts, camera_id=v.camera, true_plate=v.plate,
                    vehicle_class=v.vehicle_class, commercial=v.commercial,
                    speed_kmph=round(kmph * self.rng.uniform(0.9, 1.1), 1),
                    lane=self.rng.randint(1, max(1, self.net.camera(v.camera)["lanes"])),
                    heading=heading, direction=direction, next_camera=nxt,
                    condition=self._condition(ts)))

                if nxt is None:                       # trip finished - new journey
                    v.trips += 1
                    if self.rng.random() < 0.78:
                        # drives on from where it stopped after a short halt
                        v.route = self._route(ts, origin=v.camera)
                        idle = self.rng.expovariate(1 / 900.0) / max(self._demand(ts), 0.15)
                        v.next_ts = ts + min(idle, 5400)
                    else:
                        # parked off-network / left the city: reappears elsewhere,
                        # but only after long enough that it is physically possible
                        v.route = self._route(ts)
                        v.next_ts = ts + self.rng.uniform(45 * 60, 180 * 60)
                    v.leg = 0
                else:
                    v.leg += 1
                    v.next_ts = ts + travel_s
        out.sort(key=lambda c: c.ts)
        return out

    # --- OCR -----------------------------------------------------------
    def _engine(self):
        if self.engine is None:
            from anpr.ocr import engine
            self.engine = engine()
        return self.engine

    def read(self, cap: Capture, use_ocr: bool):
        """-> (plate as read, confidence, ocr mode, winning variant, evidence)

        `evidence` is the packed CTC lattices, kept only when the read is about
        to be refused: those are the captures the repair pass can still rescue
        once the neighbouring cameras have reported.
        """
        if use_ocr:
            # A camera sees a vehicle several times as it crosses the trigger
            # zone. Reading only one of those frames throws away every other
            # look at the same plate, and the looks fail independently: over 250
            # vehicles, one frame reads 40.4% and five read 71.2%.
            n = max(1, config.BURST_FRAMES)
            frames = [plates.capture(cap.true_plate, cap.condition, self.rng).image
                      for _ in range(n)]
            eng = self._engine()
            if config.REPAIR_ENABLED:
                read, lattices = eng.read_evidence(frames)
            else:
                read, lattices = (eng.read_burst(frames) if n > 1
                                  else eng.read(frames[0])), []
            self.stats["reads_ocr"] += 1
            self.stats["frames_decoded"] = self.stats.get("frames_decoded", 0) + n
            if read.text and read.text != cap.true_plate:
                self.stats["misreads"] += 1
            # only pay to keep the lattice for captures that are about to be
            # thrown away - a stored read has nothing left to recover
            evidence = None
            if lattices and not read.accepted:
                from core import repair as repair_mod
                evidence = repair_mod.pack_evidence(lattices)
            return read.text, read.confidence, "engine", read.variant, evidence
        # calibrated stand-in: same error *rate* as the engine, cheap
        acc = CONDITION_ACCURACY.get(cap.condition, 0.9)
        self.stats["reads_modelled"] += 1
        # The modelled path has to imitate not just the engine's error *rate* but
        # its willingness to store: measured on the burst path, the engine keeps
        # 71.3% of captures and 96.3% of those are right. Solving for the two
        # store rates that reproduce both numbers at 73.0% accuracy gives 0.94
        # if the read is correct and 0.10 if it is not. Drawing confidence from a
        # wide band instead let far too many misreads through, and every stored
        # misread invents a vehicle that never existed - 260 cars became 932
        # plates, and trajectory tracking fell apart.
        floor = config.MIN_PLATE_CONFIDENCE
        if self.rng.random() < acc:
            store = self.rng.random() < 0.94
            conf = (self.rng.uniform(floor, 1.0) if store
                    else self.rng.uniform(0.05, floor * 0.95))
            return cap.true_plate, conf, "modelled", "", None
        store = self.rng.random() < 0.10          # a misread rarely looks certain
        conf = (self.rng.uniform(floor, 0.85) if store
                else self.rng.uniform(0.01, floor * 0.9))
        return _corrupt(cap.true_plate, self.rng), conf, "modelled", "", None

    def to_sighting(self, cap: Capture, use_ocr: bool, conn=None) -> db.Sighting | None:
        plate, conf, mode, variant, evidence = self.read(cap, use_ocr)
        self.stats["captures"] += 1
        # engine reads are judged by the backend that produced them; modelled
        # ones by the classical floor they were calibrated against
        eng = self.engine
        floor = (config.MIN_PLATE_CONFIDENCE_CRNN
                 if (mode == "engine" and eng is not None and getattr(eng, "crnn", None))
                 else config.MIN_PLATE_CONFIDENCE)
        if not plate or conf < floor:
            # A real ANPR node discards reads it cannot stand behind. So do we —
            # which is why a trajectory can have gaps the operator must bridge.
            # The read is refused, but the evidence is parked: core.repair can
            # often name the vehicle once its neighbours have reported.
            self.stats["dropped_low_confidence"] += 1
            if evidence:
                if not self._schema_ready:
                    db.ensure_schema(conn)      # a database seeded before the
                    self._schema_ready = True   # repair tables existed
                db.add_unresolved(cap.ts, cap.camera_id, plate, conf, evidence,
                                  frames=config.BURST_FRAMES, condition=cap.condition,
                                  true_plate=cap.true_plate, conn=conn)
                self.stats["evidence_kept"] = self.stats.get("evidence_kept", 0) + 1
            return None
        return db.Sighting(
            ts=cap.ts, camera_id=cap.camera_id, plate=plate, confidence=round(conf, 4),
            speed_kmph=cap.speed_kmph, lane=cap.lane, heading=round(cap.heading, 1),
            direction=cap.direction, next_camera=cap.next_camera,
            vehicle_class=cap.vehicle_class, condition=cap.condition,
            ocr_variant=f"{mode}:{variant}" if variant else mode,
            true_plate=cap.true_plate)

    # --- driving the store ---------------------------------------------
    def step(self, until_ts: float | None = None, ocr_budget: int | None = None,
             conn=None, run_alerts: bool = True) -> tuple[list[db.Sighting], list[dict]]:
        """Advance to `until_ts`, store the sightings, return them with any alerts."""
        conn = conn or db.connect()
        until_ts = until_ts if until_ts is not None else time.time()
        # INLINE_OCR_MAX_PER_TICK is a budget of decoded *frames*, so raising
        # BURST_FRAMES buys accuracy per vehicle rather than more wall clock.
        frames = config.INLINE_OCR_MAX_PER_TICK if ocr_budget is None else ocr_budget
        budget = 0 if not config.INLINE_OCR else max(1, frames // max(1, config.BURST_FRAMES))
        caps = self.advance(until_ts)
        sightings, fired = [], []
        for i, cap in enumerate(caps):
            s = self.to_sighting(cap, use_ocr=i < budget, conn=conn)
            if s is None:
                continue
            s.id = db.add_sighting(s, conn=conn)
            sightings.append(s)
            if run_alerts:
                fired.extend(alert_rules.evaluate(s.as_dict(), conn=conn, net=self.net))
        return sightings, fired

    def history(self, hours: float = 6.0, end_ts: float | None = None,
                ocr_fraction: float = 0.05, conn=None, verbose: bool = True) -> int:
        """Backfill `hours` of traffic ending now, in one batch write."""
        conn = conn or db.connect()
        end_ts = end_ts or time.time()
        start = end_ts - hours * 3600.0
        for v in self.fleet:                     # rewind the fleet to the window start
            v.next_ts = start + self.rng.uniform(0, 600)
        caps = self.advance(end_ts, max_captures=400_000)
        rows = []
        for cap in caps:
            use = self.rng.random() < ocr_fraction
            s = self.to_sighting(cap, use_ocr=use, conn=conn)
            if s is not None:
                rows.append(s)
        db.add_sightings(rows, conn=conn)
        if verbose:
            print(f"  {len(caps)} camera crossings -> {len(rows)} stored sightings "
                  f"({self.stats['reads_ocr']} decoded by the engine, "
                  f"{self.stats['dropped_low_confidence']} dropped below confidence floor)")
        return len(rows)

    # --- scripted incidents --------------------------------------------
    def inject_clone(self, conn=None, gap_minutes: float = 3.0) -> str:
        """Same plate at opposite ends of the city minutes apart.

        The vehicle has to be one the network has *not* just seen: the clone rule
        compares a sighting against the previous sighting of the same plate, so a
        genuine read landing between the two scripted ones would be the one it
        measures against, and the implied speed would be ordinary. That was rare
        when the platform stored a fifth of its captures and is common now it
        stores four fifths, so the injector picks a quiet plate rather than
        assuming one.
        """
        conn = conn or db.connect()
        now = time.time()
        fleet = list(self.fleet)
        self.rng.shuffle(fleet)
        v = next((c for c in fleet
                  if not db.sightings_for_plate(c.plate, since=now - gap_minutes * 60.0,
                                                conn=conn)),
                 fleet[0])
        a, b = "CAM-35", "CAM-12"                     # Kona toll and New Town, ~22 km apart
        for cam, ts in ((a, now - gap_minutes * 60), (b, now)):
            s = db.Sighting(ts=ts, camera_id=cam, plate=v.plate, confidence=0.93,
                            speed_kmph=48.0, lane=1, heading=90.0, direction="E",
                            next_camera=None, vehicle_class=v.vehicle_class,
                            condition="clean", ocr_variant="scripted", true_plate=v.plate)
            s.id = db.add_sighting(s, conn=conn)
            alert_rules.evaluate(s.as_dict(), conn=conn, net=self.net)
        return v.plate

    def inject_loiterer(self, camera: str = "CAM-20", passes: int = 5, conn=None) -> str:
        """A vehicle circling one junction — the classic reconnaissance pattern."""
        conn = conn or db.connect()
        v = self.rng.choice(self.fleet)
        now = time.time()
        for i in range(passes):
            ts = now - (passes - i) * 7 * 60
            s = db.Sighting(ts=ts, camera_id=camera, plate=v.plate, confidence=0.91,
                            speed_kmph=22.0, lane=2, heading=180.0, direction="S",
                            next_camera=None, vehicle_class=v.vehicle_class,
                            condition="clean", ocr_variant="scripted", true_plate=v.plate)
            s.id = db.add_sighting(s, conn=conn)
            alert_rules.evaluate(s.as_dict(), conn=conn, net=self.net)
        return v.plate

    def sample_plates(self, n: int = 6) -> list[str]:
        return [v.plate for v in self.rng.sample(self.fleet, min(n, len(self.fleet)))]


def _corrupt(plate: str, rng: random.Random) -> str:
    """Apply one plausible OCR confusion, mirroring the engine's real failures."""
    from anpr.ocr import _CONF

    idx = [i for i, c in enumerate(plate) if c in _CONF]
    if not idx:
        return plate
    i = rng.choice(idx)
    return plate[:i] + rng.choice(sorted(_CONF[plate[i]])) + plate[i + 1:]
