"""Cache laptop-screen white fraction per detection sample (local only)."""

import json
import os
from pathlib import Path

from datathon.cashguard.screen import screen_series
from datathon.cashguard.video import sample_frames

DET = Path(".build/cache/det")
VIDEOS = Path(os.environ.get("CASHGUARD_DEV_VIDEOS", "data/raw/open/videos"))
OUT = Path(".build/cache/screen")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in sorted(DET.glob("*.json")):
        if (OUT / p.name).exists() or not (VIDEOS / f"{p.stem}.mov").exists():
            continue
        d = json.loads(p.read_text())
        frames = [f for _, f in sample_frames(VIDEOS / f"{d['video']}.mov", d["fps"])]
        n = min(len(frames), len(d["dets"]))
        s = screen_series(frames[:n], d["dets"][:n])
        (OUT / p.name).write_text(json.dumps({"video": d["video"], "t": d["t"][:n], "screen": s}))


if __name__ == "__main__":
    main()
