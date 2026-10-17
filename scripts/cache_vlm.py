"""Cache periodic VLM yes-probabilities for every training video (local only).

  python scripts/cache_vlm.py [fps] [--roi]
--roi: same automatic zoom as the live service (people + laptop + grounded safe),
       written to .build/cache/vlm_roi; otherwise full frames to .build/cache/vlm.
"""

import json
import sys
import time
from pathlib import Path

from PIL import Image

from datathon.cashguard.roi import action_roi, apply_roi
from datathon.cashguard.verify import Verifier
from datathon.cashguard.video import sample_frames
from datathon.cashguard.vlm import PERIODIC_KEYS, SceneVLM, VLMConfig, model_tag

VIDEOS = Path("data/raw/open/videos")
ROI = "--roi" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FPS = float(ARGS[0]) if ARGS else 0.5
SUFFIX = "" if model_tag() == "2b" else "_" + model_tag()
OUT = Path((".build/cache/vlm_roi" if ROI else ".build/cache/vlm") + SUFFIX)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vlm = SceneVLM(cfg=VLMConfig(max_pixels=464 * 848, batch=7))
    for v in sorted(VIDEOS.glob("*.mov")):
        dst = OUT / f"{v.stem}.json"
        if dst.exists():
            continue
        t0 = time.time()
        ts, imgs = [], []
        for t, f in sample_frames(v, FPS):
            ts.append(round(t, 3))
            imgs.append(Image.fromarray(f))
        roi = None
        if ROI:
            d = json.loads((Path(".build/cache/det") / f"{v.stem}.json").read_text())
            ver = Verifier(vlm, v, Path(f".build/cache/probe_roi{SUFFIX}/{v.stem}.json"))
            safe = ver.safe_box(ts[len(ts) // 2])
            roi = action_roi(d["dets"], d["height"], d["width"], [safe] if safe else None)
            imgs = [apply_roi(im, roi) for im in imgs]
        P = vlm.yes_prob_shared(imgs, PERIODIC_KEYS)
        dst.write_text(json.dumps({"video": v.stem, "fps": FPS, "t": ts, "keys": PERIODIC_KEYS,
                                   "p": P.round(4).tolist(), "roi": roi}))
        print(f"{v.stem}: {len(ts)} frames in {time.time() - t0:.1f}s roi={roi}", flush=True)


if __name__ == "__main__":
    main()
