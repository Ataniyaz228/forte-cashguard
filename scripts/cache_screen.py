"""Cache laptop-screen white fraction per detection sample (local only)."""

import json
from pathlib import Path

from datathon.cashguard.screen import screen_series
from datathon.cashguard.video import sample_frames

DET = Path(".build/cache/det")
OUT = Path(".build/cache/screen")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in sorted(DET.glob("*.json")):
        d = json.loads(p.read_text())
        frames = [f for _, f in sample_frames(Path("data/raw/open/videos") / f"{d['video']}.mov",
                                              d["fps"])]
        n = min(len(frames), len(d["dets"]))
        s = screen_series(frames[:n], d["dets"][:n])
        (OUT / p.name).write_text(json.dumps({"video": d["video"], "t": d["t"][:n], "screen": s}))


if __name__ == "__main__":
    main()
