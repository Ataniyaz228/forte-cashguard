"""Laptop screen state from pixels.

Measured on the training rooms: an open work application (light spreadsheet) gives many
light-blue pixels inside the laptop box (~0.2-0.35), a locked screen with a dark wallpaper
gives ~0.06, and a laptop seen from behind gives ~0. Thresholds live in rules.py."""

import numpy as np

from .timeline import iou


def screen_fraction(frame: np.ndarray, box) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                      int(min(w, box[2])), int(min(h, box[3])))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    crop = frame[y1:y2, x1:x2].astype(np.int16)
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    lit = (b > r + 25) & (b > g + 5)
    return float(lit.mean())


def best_laptop(dets: list, prev=None):
    laps = [(d[1], d[2]) for d in dets if d[0] in ("laptop", "tvmonitor") and d[1] >= 0.4]
    if not laps:
        return None
    if prev is not None:
        laps.sort(key=lambda sb: -iou(sb[1], prev))
        if iou(laps[0][1], prev) > 0.3:
            return laps[0][1]
    return max(laps)[1]


def screen_series(frames: list[np.ndarray], dets: list[list]) -> list[float | None]:
    out: list[float | None] = []
    prev = None
    for f, d in zip(frames, dets):
        b = best_laptop(d, prev)
        if b is None:
            out.append(None)
            continue
        prev = b
        out.append(round(screen_fraction(f, b), 4))
    return out
