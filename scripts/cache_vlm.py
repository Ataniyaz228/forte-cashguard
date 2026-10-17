"""Cache periodic VLM yes-probabilities for every training video (local only)."""

import json
import sys
import time
from pathlib import Path

from PIL import Image

from datathon.cashguard.video import sample_frames
from datathon.cashguard.vlm import SceneVLM, VLMConfig

VIDEOS = Path("data/raw/open/videos")
OUT = Path(".build/cache/vlm")
FPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
KEYS = ["safe_open", "cash_on_desk", "bag_on_desk", "docs_on_desk", "food",
        "personal_items", "phone", "counting", "id_shown"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vlm = SceneVLM(cfg=VLMConfig(max_pixels=464 * 848, batch=9))
    for v in sorted(VIDEOS.glob("*.mov")):
        dst = OUT / f"{v.stem}.json"
        if dst.exists():
            continue
        t0 = time.time()
        ts, imgs = [], []
        for t, f in sample_frames(v, FPS):
            ts.append(round(t, 3))
            imgs.append(Image.fromarray(f))
        P = vlm.yes_prob(imgs, KEYS)
        dst.write_text(json.dumps({"video": v.stem, "fps": FPS, "t": ts, "keys": KEYS,
                                   "p": P.round(4).tolist()}))
        print(f"{v.stem}: {len(ts)} frames in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
