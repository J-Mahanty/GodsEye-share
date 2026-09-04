"""Render every camera scenario side by side and look at it.

    python -m anpr.camera_sheet [out.png]

The calibration errors in this model were all found by eye, not by reasoning
about the code. If the daylight tile is not comfortably readable, the camera is
wrong, not the reader.
"""
from __future__ import annotations

import random
import sys

import cv2
import numpy as np

from anpr import camera, plates


def sheet(path: str = "camera_scenarios.png", text: str = "WB24AB1234",
          seed: int = 7, width: int = 620) -> str:
    rng = random.Random(seed)
    rows = []
    for name in camera.SCENARIOS:
        cap = camera.scenario(name, rng)
        plate = plates.render_plate(text, rng=rng, width=760,
                                    two_row=(name == "far_lane"), ind_strip=True)
        shot = camera.shoot(plate, cap, rng)
        scaled = cv2.resize(shot, (width, max(28, int(shot.shape[0] * width / shot.shape[1]))))
        bar = np.full((26, width, 3), 245, np.uint8)
        cv2.putText(bar, f"{name}  |  {cap.describe()}"[:96], (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
        rows.append(np.vstack([bar, scaled]))
        print(f"  {name:14} {shot.shape[1]:4}x{shot.shape[0]:3}px  {cap.describe()}")
    h = max(r.shape[0] for r in rows)
    rows = [np.vstack([r, np.full((h - r.shape[0], width, 3), 245, np.uint8)]) for r in rows]
    grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
    cv2.imwrite(path, grid)
    return path


if __name__ == "__main__":
    out = sheet(sys.argv[1] if len(sys.argv) > 1 else "camera_scenarios.png")
    print(f"wrote {out}")
