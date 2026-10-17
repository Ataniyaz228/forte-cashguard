"""Run the detector over every training video and cache results (local only)."""

import json
import os
import sys
import time
from pathlib import Path

from datathon.cashguard.detect import Detector
from datathon.cashguard.video import probe, sample_frames

VIDEOS = Path(os.environ.get("CASHGUARD_DEV_VIDEOS", "data/raw/open/videos"))
OUT = Path(".build/cache/det")
FPS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    det = Detector()
    for v in sorted(VIDEOS.glob("*.mov")):
        dst = OUT / f"{v.stem}.json"
        if dst.exists():
            continue
        t0 = time.time()
        info = probe(v)
        ts, frames = [], []
        for t, f in sample_frames(v, FPS):
            ts.append(round(t, 3))
            frames.append(f)
        dets = det(frames)
        h, w = frames[0].shape[:2]
        dst.write_text(json.dumps({"video": v.stem, "fps": FPS, "width": w, "height": h,
                                   "duration": info.duration_sec, "t": ts,
                                   "dets": [[d.to_list() for d in fd] for fd in dets]}))
        print(f"{v.stem}: {len(ts)} frames in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
