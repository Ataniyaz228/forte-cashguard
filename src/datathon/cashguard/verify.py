"""Second-stage targeted checks at specific moments (zoomed crops, stricter questions).

Rules ask `probe(name, times)` and get max P(yes) over the given times. The safe is
located once per video by Qwen3-VL grounding; the lock region is a fixed part of it.
"""

import json
import re
from pathlib import Path

from PIL import Image

from .roi import apply_roi
from .video import frames_at
from .vlm import SceneVLM

PROBES = {
    "desk_items": "Is anything lying on the desk besides the laptop, the desk phone and the "
                  "money counting machine (for example papers, a tablet, money, a bag or a "
                  "drink)?",
    "cash": "Are banknotes or stacks of cash lying on the desk?",
    "safe_open": "Is the door of this metal safe open?",
    "key_in_lock": "Is a key inserted in this lock?",
    "door_open": "Is the door of this room standing open?",
    # same questions asked on a zoomed crop of the desk (robust to a farther camera)
    "desk_left": "Is anything lying on the desk besides the laptop, the desk phone and the "
                 "money counting machine (for example papers, a tablet, money, a bag or a "
                 "drink)?",
    "desk_cash": "Are banknotes or stacks of cash lying on the desk?",
    "id_card": "Is someone holding or showing an ID card or an identity document?",
    "cash_in_hands": "Is someone holding banknotes in their hands or passing money across "
                     "the desk?",
}
CROP = {"safe_open": "safe", "key_in_lock": "lock", "desk_left": "desk", "desk_cash": "desk"}
GROUND_Q = ('Locate the grey metal safe (cabinet) in the image and output its bbox '
            'coordinates in JSON format as [{"bbox_2d": [x1, y1, x2, y2], "label": "safe"}].')
DESK_Q = ('Locate the wooden desk (work table) in the image and output its bbox coordinates '
          'in JSON format as [{"bbox_2d": [x1, y1, x2, y2], "label": "desk"}].')
LOCK_Q = ('Locate the lock (keyhole) on the door of this safe and output its bbox coordinates '
          'in JSON format as [{"bbox_2d": [x1, y1, x2, y2], "label": "lock"}].')
BOX_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def _rel_box(txt: str, w: float, h: float):
    m = BOX_RE.search(txt)
    if not m:
        return None
    x1, y1, x2, y2 = (int(v) for v in m.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000]


def _crop(img: Image.Image, b, pad=0.25, size=448) -> Image.Image:
    w, h = b[2] - b[0], b[3] - b[1]
    c = img.crop((max(0, b[0] - pad * w), max(0, b[1] - pad * h),
                  min(img.width, b[2] + pad * w), min(img.height, b[3] + pad * h)))
    s = size / max(c.size)
    return c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.LANCZOS)


def _lock(img: Image.Image, b, lock=None) -> Image.Image:
    """Zoomed lock region. `lock` = lock box found by grounding (image coordinates);
    fallback: the typical lock position on the door edge (left middle of the safe)."""
    w, h = b[2] - b[0], b[3] - b[1]
    if lock is not None:
        lw, lh = lock[2] - lock[0], lock[3] - lock[1]
        side = max(lw, lh, 0.25 * w) * 1.6
        cx, cy = (lock[0] + lock[2]) / 2, (lock[1] + lock[3]) / 2
        c = img.crop((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    else:
        c = img.crop((b[0] - 0.1 * w, b[1] + 0.25 * h, b[0] + 0.45 * w, b[1] + 0.7 * h))
    s = max(1, round(160 / max(1, max(c.size))))
    return c.resize((max(1, c.width * s), max(1, c.height * s)), Image.LANCZOS)


class Verifier:
    def __init__(self, vlm: SceneVLM, path: Path, cache_file: Path | None = None,
                 roi=None):
        self.vlm = vlm
        self.path = path
        self.roi = roi  # full-frame questions see the auto-zoomed action region
        self.cache_file = cache_file
        self.cache: dict = {}
        if cache_file and cache_file.exists():
            self.cache = json.loads(cache_file.read_text())
        self._safe_box = self.cache.get("_safe_box", "unset")
        self._lock_box = self.cache.get("_lock_box", "unset")
        self._desk_box = self.cache.get("_desk_box", "unset")

    def _save(self):
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache))

    def safe_box(self, t_hint: float):
        if self._safe_box != "unset":
            return self._safe_box
        img = frames_at(self.path, [t_hint])[t_hint]
        box = _rel_box(self.vlm.describe(img, GROUND_Q, 80), *img.size)
        if box is not None and (box[2] - box[0] < 8 or box[3] - box[1] < 8):
            box = None
        self._safe_box = box
        self.cache["_safe_box"] = box
        self._save()
        return box

    def desk_box(self, t_hint: float):
        """Desk found by grounding once per video; None if the 'desk' is most of the frame
        (then the zoom would add nothing and the full-frame question is used)."""
        if self._desk_box != "unset":
            return self._desk_box
        img = frames_at(self.path, [t_hint])[t_hint]
        box = _rel_box(self.vlm.describe(img, DESK_Q, 80), *img.size)
        if box is not None:
            area = (box[2] - box[0]) * (box[3] - box[1]) / (img.width * img.height)
            if area < 0.01 or area > 0.6:
                box = None
        self._desk_box = box
        self.cache["_desk_box"] = box
        self._save()
        return box

    def lock_box(self, img: Image.Image, safe):
        """Lock position found inside the zoomed safe crop (adapts to any camera angle)."""
        if self._lock_box != "unset":
            return self._lock_box
        pad = 0.1
        w, h = safe[2] - safe[0], safe[3] - safe[1]
        x0, y0 = max(0, safe[0] - pad * w), max(0, safe[1] - pad * h)
        crop = img.crop((x0, y0, min(img.width, safe[2] + pad * w),
                         min(img.height, safe[3] + pad * h)))
        k = 448 / max(crop.size)
        big = crop.resize((int(crop.width * k), int(crop.height * k)), Image.LANCZOS)
        rb = _rel_box(self.vlm.describe(big, LOCK_Q, 80), *big.size)
        lock = None
        if rb is not None:
            lock = [x0 + rb[0] / k, y0 + rb[1] / k, x0 + rb[2] / k, y0 + rb[3] / k]
            lw, lh = lock[2] - lock[0], lock[3] - lock[1]
            if lw > 0.6 * w or lh > 0.6 * h:  # the whole door, not the lock
                lock = None
        self._lock_box = lock
        self.cache["_lock_box"] = lock
        self._save()
        return lock

    def series(self, names: list[str], times: list[float]) -> dict[str, list[float]]:
        """Several full-frame questions on each time (image encoded once per frame)."""
        times = [round(t, 2) for t in times]
        if self.vlm is None:
            return {n: [self.cache.get(f"{n}@{t}", self._nearest_cached(n, t))
                        for t in times] for n in names}
        todo = [t for t in times if any(f"{n}@{t}" not in self.cache for n in names)]
        if todo:
            frames = frames_at(self.path, todo)
            P = self.vlm.yes_prob_shared([apply_roi(frames[t], self.roi) for t in todo],
                                         [PROBES[n] for n in names])
            for i, t in enumerate(todo):
                for j, n in enumerate(names):
                    self.cache[f"{n}@{t}"] = round(float(P[i, j]), 4)
            self._save()
        return {n: [self.cache[f"{n}@{t}"] for t in times] for n in names}

    def _nearest_cached(self, name: str, t: float, tol: float = 0.8) -> float:
        best, val = tol, 0.0
        for k, v in self.cache.items():
            if k.startswith(name + "@") and abs(float(k.split("@")[1]) - t) <= best:
                best, val = abs(float(k.split("@")[1]) - t), v
        return val

    def __call__(self, name: str, times: list[float]) -> float:
        times = [round(t, 2) for t in times]
        if self.vlm is None:  # cache-only mode (development without a free GPU)
            vals = [self.cache.get(f"{name}@{t}", self._nearest_cached(name, t))
                    for t in times]
            return max(vals) if vals else 0.0
        todo = [t for t in times if f"{name}@{t}" not in self.cache]
        if todo:
            frames = frames_at(self.path, todo)
            imgs = []
            for t in todo:
                img = frames[t]
                kind = CROP.get(name)
                if not kind:
                    img = apply_roi(img, self.roi)
                elif kind == "desk":
                    box = self.desk_box(t)
                    img = _crop(img, box, pad=0.12, size=640) if box else apply_roi(img, self.roi)
                else:
                    box = self.safe_box(t)
                    if box is None:
                        self.cache[f"{name}@{t}"] = 0.0
                        continue
                    if kind == "safe":
                        img = _crop(img, box)
                    else:
                        img = _lock(img, box, self.lock_box(img, box))
                imgs.append((t, img))
            if imgs:
                p = self.vlm.yes_prob([i for _, i in imgs], [PROBES[name]])[:, 0]
                for (t, _), v in zip(imgs, p):
                    self.cache[f"{name}@{t}"] = round(float(v), 4)
            self._save()
        return max(self.cache[f"{name}@{t}"] for t in times) if times else 0.0
