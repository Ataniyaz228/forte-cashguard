"""Compare VLM variants on labelled targeted checks from the training videos (local only).

For each check prints P(yes) on positives and negatives and the margin
min(pos) - max(neg) (>0 = perfectly separable on these cases), plus seconds per query.
"""

import json
import sys
import time
from pathlib import Path

from datathon.cashguard.verify import PROBES, _crop, _lock
from datathon.cashguard.video import frames_at
from datathon.cashguard.vlm import SceneVLM, VLMConfig

V = Path("data/raw/open/videos")
SAFE_BOX = {}  # reuse boxes found by the 2B grounding pass
for p in Path(".build/cache/probe").glob("*.json"):
    b = json.loads(p.read_text()).get("_safe_box")
    if b:
        SAFE_BOX[p.stem] = b

CASES = {
    "key_in_lock": ([("Х-22", 17.0), ("Х-22", 19.0), ("Х-22", 20.5)],
                    [("Х-23", 16.0), ("Х-23", 18.0), ("Х-25", 30.0), ("Х-26", 18.0),
                     ("Х-27", 25.5), ("Х-28", 15.8), ("Х-29", 14.5), ("Х-29", 16.5),
                     ("Х-24", 14.0), ("Х-12", 41.5)]),
    "safe_open": ([("Х-20", 8.0), ("Х-20", 10.0), ("Х-5", 50.0), ("Х-21", 4.0)],
                  [("Х-21", 12.0), ("Х-23", 17.0), ("Х-25", 30.0), ("Х-22", 20.0),
                   ("Х-8", 30.0), ("Х-27", 26.0)]),
    "desk_items": ([("Х-24", 13.5), ("Х-26", 17.0), ("Х-15", 19.5)],
                   [("Х-23", 17.0), ("Х-25", 30.5), ("Х-27", 25.5), ("Х-28", 15.8),
                    ("Х-29", 16.5), ("Х-11", 24.0), ("Х-12", 41.5)]),
    "door_open": ([("Х-32", 12.0), ("Х-32", 18.0), ("Х-33", 9.0), ("Х-33", 12.0)],
                  [("Х-31", 12.0), ("Х-31", 20.0), ("Х-43", 10.0), ("Х-45", 20.0)]),
    "cash": ([("Х-38", 30.7), ("Х-38", 32.2), ("Х-4", 56.7), ("Х-4", 58.2)],
             [("Х-39", 44.7), ("Х-39", 72.0), ("Х-5", 16.3), ("Х-5", 62.7), ("Х-3", 7.0),
              ("X-1", 2.0)]),
}


def image_for(name, video, t, frames, vlm=None):
    img = frames[(video, t)]
    if name in ("safe_open", "key_in_lock") and video not in SAFE_BOX:
        from datathon.cashguard.verify import Verifier

        SAFE_BOX[video] = Verifier(vlm, V / f"{video}.mov").safe_box(t)
    if name == "safe_open":
        return _crop(img, SAFE_BOX[video])
    if name == "key_in_lock":
        return _lock(img, SAFE_BOX[video])
    return img


def main():
    model = sys.argv[1]
    four = "--4bit" in sys.argv
    vlm = SceneVLM(cfg=VLMConfig(model=model, max_pixels=464 * 848, batch=4, load_4bit=four))
    frames = {}
    for name, (pos, neg) in CASES.items():
        for v, t in pos + neg:
            frames[(v, t)] = frames_at(V / f"{v}.mov", [t])[t]
    total_q, total_s = 0, 0.0
    for name, (pos, neg) in CASES.items():
        imgs = [image_for(name, v, t, frames, vlm) for v, t in pos + neg]
        t0 = time.time()
        p = vlm.yes_prob(imgs, [PROBES[name]])[:, 0]
        total_s += time.time() - t0
        total_q += len(imgs)
        pp, pn = p[:len(pos)], p[len(pos):]
        print(f"{name:12s} margin={pp.min() - pn.max():+.2f} pos={pp.round(2).tolist()} "
              f"neg_max={pn.max():.2f} neg={pn.round(2).tolist()}")
    print(f"{model} 4bit={four}: {total_s / total_q:.2f}s/query")


if __name__ == "__main__":
    main()
