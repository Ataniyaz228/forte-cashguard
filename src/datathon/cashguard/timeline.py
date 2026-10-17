"""People-in-room timeline from frame detections: counts, entries and exits."""

from dataclasses import dataclass, field

import numpy as np

MIN_PERSON_SCORE = 0.6
MIN_PERSON_HEIGHT = 0.12  # fraction of frame height
MIN_PERSON_WIDTH = 0.13  # fraction of frame width; people behind the glass wall look narrow


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / min(area(a), area(b))


@dataclass
class PeopleStats:
    """Typical look of clearly visible people in this video (camera- and scene-relative)."""
    w: float
    h: float
    sat: float | None
    sharp: float | None


def people_stats(all_dets: list) -> PeopleStats | None:
    ws, hs, sats, sharps = [], [], [], []
    for fd in all_dets:
        for d in fd:
            if d[0] == "person" and d[1] >= 0.85:
                b = d[2]
                ws.append(b[2] - b[0])
                hs.append(b[3] - b[1])
                if len(d) > 3 and d[3]:
                    sats.append(d[3]["sat"])
                    sharps.append(d[3]["sharp"])
    if len(ws) < 5:
        return None
    med = lambda x: float(np.median(x)) if len(x) >= 5 else None
    return PeopleStats(float(np.median(ws)), float(np.median(hs)), med(sats), med(sharps))


def ghost_votes(d, stats: PeopleStats | None, height: int, width: int) -> int:
    """Evidence that a person box is someone seen through the glass wall, not in the room.

    Each cue alone is ambiguous (a far person is small, dark clothes are unsaturated), so a
    box is dropped only when at least two independent cues agree. All limits are relative
    to this video's clearly visible people, so a new camera angle or zoom rescales them.
    """
    b, score = d[2], d[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    min_w, min_h = MIN_PERSON_WIDTH * width, MIN_PERSON_HEIGHT * height
    if stats is not None:
        min_w = min(min_w, 0.72 * stats.w)
        min_h = min(min_h, 0.5 * stats.h)
    votes = int(bw < min_w or bh < min_h) + int(score < 0.8)
    look = d[3] if len(d) > 3 else None
    if look and stats is not None:
        if stats.sat:
            votes += int(look["sat"] < 0.5 * stats.sat)
        if stats.sharp:
            votes += int(look["sharp"] < 0.75 * stats.sharp)
    elif look is None:
        votes += int(bw < min_w)  # no appearance data: fall back to the size rule
    return votes


def room_people(dets: list, height: int, width: int,
                stats: PeopleStats | None = None) -> list[tuple[float, tuple]]:
    """Persons inside the room: confident, not behind the glass, de-duplicated."""
    cand = []
    for d in dets:
        if d[0] != "person" or d[1] < MIN_PERSON_SCORE:
            continue
        b = d[2]
        if (b[3] - b[1]) < 0.5 * MIN_PERSON_HEIGHT * height:  # tiny = background
            continue
        if ghost_votes(d, stats, height, width) >= 2:
            continue
        cand.append((d[1], tuple(b)))
    kept: list[tuple[float, tuple]] = []
    for s, b in sorted(cand, reverse=True):
        if all(iou(b, k) < 0.6 for _, k in kept):
            kept.append((s, b))
    return kept


@dataclass
class Event:
    kind: str  # "exit_all" | "enter_first" | "count_up" | "count_down"
    t: float
    before: int
    after: int


@dataclass
class Timeline:
    t: np.ndarray
    count: np.ndarray  # smoothed people count per sample
    boxes: list[list[tuple]]
    events: list[Event] = field(default_factory=list)

    def count_at(self, t: float) -> int:
        i = int(np.argmin(np.abs(self.t - t)))
        return int(self.count[i])

    def present_ratio(self, t0: float, t1: float) -> float:
        m = (self.t >= t0) & (self.t <= t1)
        return float((self.count[m] > 0).mean()) if m.any() else 0.0


def _mode_filter(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        w = xp[i:i + k]
        vals, cnt = np.unique(w, return_counts=True)
        best = vals[cnt == cnt.max()]
        out[i] = x[i] if x[i] in best else best.max()
    return out


def build_timeline(ts: list[float], dets: list[list], height: int, width: int,
                   fps: float) -> Timeline:
    stats = people_stats(dets)
    boxes = [[b for _, b in room_people(fd, height, width, stats)] for fd in dets]
    raw = np.array([len(b) for b in boxes], dtype=int)
    k = max(3, round(fps * 1.5) | 1)  # ~1.5 s odd window
    cnt = _mode_filter(raw, k)
    t = np.asarray(ts, dtype=float)
    tl = Timeline(t, cnt, boxes)
    min_hold = max(2, round(fps))
    for i in range(1, len(cnt)):
        # a change counts only if it persists for min_hold samples
        persists = np.all(cnt[i:i + min_hold] == cnt[i]) or i + min_hold > len(cnt)
        if cnt[i] == cnt[i - 1] or not persists:
            continue
        before, after = int(cnt[i - 1]), int(cnt[i])
        kind = "count_up" if after > before else "count_down"
        if after == 0:
            kind = "exit_all"
        elif before == 0:
            kind = "enter_first"
        # exit time: last moment someone was seen; entry: first seen
        te = float(t[i - 1]) if kind in ("exit_all", "count_down") else float(t[i])
        tl.events.append(Event(kind, te, before, after))
    return tl
