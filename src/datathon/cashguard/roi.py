"""Automatic zoom: the region where the action happens, found per video.

The union (robust percentiles) of in-room people and the laptop over the whole video,
padded, is cropped and upscaled to the VLM pixel budget. A camera that is farther away
or tilted then gives the VLM roughly the same object scale as the training cameras.
"""

import math

import numpy as np
from PIL import Image

from .timeline import people_stats, room_people

MAX_PIXELS = 464 * 848
MAX_UPSCALE = 2.5


MIN_W, MIN_H = 0.6, 0.4  # keep room context: never zoom into less than this


def action_roi(dets: list, height: int, width: int, extra: list | None = None) -> tuple | None:
    """`extra`: boxes that must stay visible (e.g. the safe found by VLM grounding)."""
    stats = people_stats(dets)
    boxes = [list(b) for b in (extra or []) if b]
    for fd in dets:
        boxes += [b for _, b in room_people(fd, height, width, stats)]
        boxes += [d[2] for d in fd if d[0] == "laptop" and d[1] >= 0.5]
    if len(boxes) < 3:
        return None
    a = np.asarray(boxes, dtype=float)
    x1, y1 = np.percentile(a[:, 0], 2), np.percentile(a[:, 1], 2)
    x2, y2 = np.percentile(a[:, 2], 98), np.percentile(a[:, 3], 98)
    for b in extra or []:  # extra boxes are never trimmed by the percentiles
        if b:
            x1, y1, x2, y2 = min(x1, b[0]), min(y1, b[1]), max(x2, b[2]), max(y2, b[3])
    # room context around the action (door, desk edges) + headroom
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - 0.2 * w, x2 + 0.2 * w
    y1, y2 = y1 - 0.15 * h, y2 + 0.15 * h
    x1, x2 = _grow(x1, x2, MIN_W * width, width)
    y1, y2 = _grow(y1, y2, MIN_H * height, height)
    if (x2 - x1) * (y2 - y1) >= 0.8 * width * height:
        return None  # the action already fills the frame
    return (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))


def _grow(a: float, b: float, min_len: float, limit: float) -> tuple[float, float]:
    """Widen [a, b] to at least min_len around its centre, then clamp to [0, limit]."""
    if b - a < min_len:
        c = (a + b) / 2
        a, b = c - min_len / 2, c + min_len / 2
    if a < 0:
        a, b = 0.0, min(limit, b - a)
    if b > limit:
        a, b = max(0.0, a - (b - limit)), limit
    return a, b


def apply_roi(img: Image.Image, roi) -> Image.Image:
    if roi is None:
        return img
    c = img.crop(tuple(round(v) for v in roi))
    k = min(MAX_UPSCALE, math.sqrt(MAX_PIXELS / max(1, c.width * c.height)))
    if k > 1.05:
        c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
    return c
