"""Per-video signals consumed by the rules (same structure for live runs and dev caches)."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .timeline import Timeline, build_timeline


@dataclass
class Features:
    video: str
    duration: float
    width: int
    height: int
    det_fps: float
    det_t: np.ndarray
    dets: list
    timeline: Timeline
    screen: np.ndarray  # laptop screen "lit" fraction per det sample, nan = laptop not seen
    vlm_t: np.ndarray
    vlm: dict[str, np.ndarray]
    # second-stage targeted check: probe(name, times) -> max P(yes); None = unavailable
    probe: Callable[[str, list[float]], float] | None = None

    # ---- helpers used by rules ----
    def v(self, key: str) -> np.ndarray:
        return self.vlm.get(key, np.zeros_like(self.vlm_t))

    def vwin(self, key: str, t0: float, t1: float, agg=np.max) -> float:
        m = (self.vlm_t >= t0 - 1e-6) & (self.vlm_t <= t1 + 1e-6)
        if not m.any():  # nearest sample
            i = int(np.argmin(np.abs(self.vlm_t - (t0 + t1) / 2)))
            return float(self.v(key)[i])
        return float(agg(self.v(key)[m]))

    def screen_win(self, t0: float, t1: float) -> float | None:
        m = (self.det_t >= t0) & (self.det_t <= t1) & ~np.isnan(self.screen)
        return float(np.median(self.screen[m])) if m.any() else None

    def person_heights(self) -> np.ndarray:
        """Box height of the single visible person per det sample (nan otherwise)."""
        out = np.full(len(self.det_t), np.nan)
        for i, bx in enumerate(self.timeline.boxes):
            if len(bx) == 1:
                out[i] = bx[0][3] - bx[0][1]
        return out


def from_cache(video: str, root: Path = Path(".build/cache")) -> Features:
    d = json.loads((root / "det" / f"{video}.json").read_text())
    s = json.loads((root / "screen" / f"{video}.json").read_text())
    v = json.loads((root / "vlm" / f"{video}.json").read_text())
    tl = build_timeline(d["t"], d["dets"], d["height"], d["width"], d["fps"])
    n = len(d["t"])
    scr = np.array([np.nan if x is None else x for x in s["screen"]][:n] +
                   [np.nan] * max(0, n - len(s["screen"])), dtype=float)
    P = np.asarray(v["p"], dtype=float)
    from .vlm import PERIODIC_KEYS  # same questions as a live run

    vlm = {k: P[:, j] for j, k in enumerate(v["keys"]) if k in PERIODIC_KEYS}
    return Features(video, d["duration"], d["width"], d["height"], d["fps"],
                    np.asarray(d["t"], float), d["dets"], tl, scr,
                    np.asarray(v["t"], float), vlm)
