"""The city camera network as a routable graph.

Cameras are nodes with real coordinates; links are the road segments between
adjacent cameras with a length and a free-flow speed. Everything spatial in the
platform — travel times, implied speeds, congestion ratios, map polylines,
plausibility checks on a trajectory — comes off this graph.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import networkx as nx

import config

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compass(bearing: float) -> str:
    return COMPASS[int((bearing % 360) / 22.5 + 0.5) % 16]


class CameraNetwork:
    def __init__(self, spec: dict):
        self.city = spec.get("city", config.CITY_NAME)
        self.cameras: dict[str, dict] = {c["id"]: c for c in spec["cameras"]}
        self.graph = nx.Graph()
        for cam in spec["cameras"]:
            self.graph.add_node(cam["id"], **cam)
        for link in spec["links"]:
            a, b = link["a"], link["b"]
            if self.graph.has_edge(a, b):
                continue
            km = float(link["km"])
            free = float(link["free_kmph"])
            self.graph.add_edge(a, b, km=km, free_kmph=free, name=link.get("name", ""),
                                free_minutes=km / free * 60.0,
                                # intermediate points that bend the road between
                                # the two cameras, stored a->b
                                shape=[tuple(pt) for pt in link.get("shape", [])],
                                shape_from=a)

    # --- basics --------------------------------------------------------
    @classmethod
    def load(cls, path=None) -> "CameraNetwork":
        with open(path or config.CAMERA_FILE, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def __len__(self) -> int:
        return len(self.cameras)

    def camera(self, cam_id: str) -> dict:
        return self.cameras[cam_id]

    def name(self, cam_id: str) -> str:
        cam = self.cameras.get(cam_id)
        return cam["name"] if cam else cam_id

    def coords(self, cam_id: str) -> tuple[float, float]:
        cam = self.cameras[cam_id]
        return cam["lat"], cam["lon"]

    @property
    def gateways(self) -> list[str]:
        return [c["id"] for c in self.cameras.values() if c.get("type") == "gateway"]

    def neighbours(self, cam_id: str) -> list[str]:
        return list(self.graph.neighbors(cam_id))

    def links(self) -> list[dict]:
        out = []
        for a, b, d in self.graph.edges(data=True):
            la, lo = self.coords(a)
            lb, lob = self.coords(b)
            out.append({"a": a, "b": b, "name": d["name"], "km": d["km"],
                        "free_kmph": d["free_kmph"], "free_minutes": d["free_minutes"],
                        "a_lat": la, "a_lon": lo, "b_lat": lb, "b_lon": lob,
                        "path": [[lon, lat] for lat, lon in self.edge_geometry(a, b)]})
        return out

    # --- routing -------------------------------------------------------
    @lru_cache(maxsize=4096)
    def route(self, a: str, b: str) -> list[str]:
        """Fastest camera-to-camera path (free-flow), [] if disconnected."""
        if a == b:
            return [a]
        try:
            return nx.shortest_path(self.graph, a, b, weight="free_minutes")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def route_km(self, a: str, b: str) -> float:
        path = self.route(a, b)
        if len(path) < 2:
            return 0.0
        return sum(self.graph[u][v]["km"] for u, v in zip(path, path[1:]))

    def free_flow_minutes(self, a: str, b: str) -> float:
        path = self.route(a, b)
        if len(path) < 2:
            return 0.0
        return sum(self.graph[u][v]["free_minutes"] for u, v in zip(path, path[1:]))

    def edge_geometry(self, u: str, v: str) -> list[tuple[float, float]]:
        """[(lat, lon), ...] from u to v along one link, including the shape
        points that bend it. Stored geometry is direction-agnostic, so it is
        reversed when the vehicle travelled the link the other way."""
        if not self.graph.has_edge(u, v):
            return [self.coords(u), self.coords(v)]
        d = self.graph[u][v]
        shape = list(d.get("shape") or [])
        if shape and d.get("shape_from") != u:
            shape.reverse()
        return [self.coords(u), *shape, self.coords(v)]

    def polyline(self, a: str, b: str) -> list[tuple[float, float]]:
        """[(lat, lon), ...] along the whole routed path — what the map draws."""
        route = self.route(a, b)
        if len(route) < 2:
            return [self.coords(c) for c in route]
        pts: list[tuple[float, float]] = []
        for u, v in zip(route, route[1:]):
            seg = self.edge_geometry(u, v)
            pts.extend(seg if not pts else seg[1:])
        return pts

    def heading(self, a: str, b: str) -> float:
        (la, lo), (lb, lob) = self.coords(a), self.coords(b)
        return bearing_deg(la, lo, lb, lob)

    def direction(self, a: str, b: str) -> str:
        return compass(self.heading(a, b))

    def link_km(self, a: str, b: str) -> float:
        if self.graph.has_edge(a, b):
            return float(self.graph[a][b]["km"])
        return self.route_km(a, b)

    def link_free_kmph(self, a: str, b: str) -> float:
        if self.graph.has_edge(a, b):
            return float(self.graph[a][b]["free_kmph"])
        mins = self.free_flow_minutes(a, b)
        return (self.route_km(a, b) / mins * 60.0) if mins else 0.0

    def as_dict(self) -> dict:
        return {"city": self.city,
                "cameras": list(self.cameras.values()),
                "links": self.links()}


_network: CameraNetwork | None = None


def network() -> CameraNetwork:
    global _network
    if _network is None:
        _network = CameraNetwork.load()
    return _network
