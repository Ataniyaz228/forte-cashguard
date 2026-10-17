"""Appearance statistics of person boxes: are people behind the glass separable?

Groups: 'narrow' = boxes rejected by the old width rule (mostly people behind the glass),
'wide' = accepted ones. Prints contrast/saturation/sharpness distributions per group.
"""

import json
from pathlib import Path

import numpy as np

from datathon.cashguard.video import sample_frames

DET = Path(".build/cache/det")


def box_stats(fr, b):
    x1, y1, x2, y2 = (int(max(0, v)) for v in b)
    c = fr[y1:y2, x1:x2].astype(np.float32)
    if c.size == 0:
        return None
    g = c.mean(axis=2)
    contrast = g.std()
    sat = (c.max(axis=2) - c.min(axis=2)).mean()
    lap = np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()
    dark = (g < 60).mean()  # dark clothes/hair are rarely dark behind glass
    return contrast, sat, lap, dark


def main():
    rows = []
    for p in sorted(DET.glob("*.json")):
        d = json.loads(p.read_text())
        W, H = d["width"], d["height"]
        frames = dict(sample_frames(Path("data/raw/open/videos") / f"{d['video']}.mov", 1.0))
        for t, fr in frames.items():
            i = int(np.argmin(np.abs(np.asarray(d["t"]) - t)))
            for lab, s, b in d["dets"][i]:
                if lab != "person" or s < 0.6 or (b[3] - b[1]) < 0.12 * H:
                    continue
                st = box_stats(fr, b)
                if st:
                    rows.append((d["video"], t, (b[2] - b[0]) / W, b[3] / H, s, *st))
    a = np.array([r[2:] for r in rows], dtype=float)
    narrow = a[:, 0] < 0.13
    names = ["width", "bottom", "score", "contrast", "saturation", "sharpness", "dark"]
    for j, n in enumerate(names):
        qn = np.percentile(a[narrow, j], [10, 50, 90]).round(3)
        qw = np.percentile(a[~narrow, j], [10, 50, 90]).round(3)
        print(f"{n:10s} narrow(n={narrow.sum()}) {qn}   wide(n={(~narrow).sum()}) {qw}")


if __name__ == "__main__":
    main()
