"""Stress test for a different camera: run the live pipeline on geometrically altered copies
of training videos and compare verdict/type/timecode with the original labels.

Transforms (ffmpeg, local only, outputs in .build/robust/):
  mirror   - horizontal flip (door/safe on the other side)
  persp    - perspective tilt (camera moved sideways)
  zoomin   - 80% centre crop upscaled (camera closer)
  zoomout  - shrunk to 70% and padded (camera farther / wider lens)
Usage: python scripts/robustness.py [video ...]   (default: a balanced subset)
Uses the running local service (http://127.0.0.1:8090, key from .env) so no second model
copy is loaded on the GPU.
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from datathon.cashguard.config import settings

BASE = os.environ.get("CASHGUARD_BASE", "http://127.0.0.1:8090")

V = Path("data/raw/open/videos")
OUT = Path(".build/robust")
FILTERS = {
    "mirror": "hflip",
    "persp": "perspective=x0=40:y0=0:x1=W:y1=40:x2=0:y2=H:x3=W-40:y3=H-20:sense=destination",
    "zoomin": "crop=iw*0.8:ih*0.8,scale=464:848",
    "zoomout": "scale=iw*0.7:ih*0.7,pad=464:848:(ow-iw)/2:(oh-ih)/2:color=gray",
}
DEFAULT = ["Х-20", "Х-21", "Х-26", "Х-27", "Х-28", "Х-29", "Х-17", "Х-16", "Х-38", "Х-39",
           "Х-46", "Х-43", "Х-15", "Х-30", "Х-31"]


def transformed(video: str, name: str) -> Path:
    dst = OUT / f"{video}__{name}.mov"
    if not dst.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i",
                        str(V / f"{video}.mov"), "-vf", FILTERS[name], "-c:v", "libx264",
                        "-crf", "20", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def analyze_http(path: Path, key: str) -> dict:
    with httpx.Client(timeout=120) as c, path.open("rb") as fh:
        job = c.post(f"{BASE}/api/v1/analyze", headers={"X-API-Key": key},
                     files={"file": (path.name, fh, "video/quicktime")}).json()["job_id"]
        while True:
            time.sleep(1.5)
            # the UI route returns the extended result (type_code etc.)
            body = c.get(f"{BASE}/ui/jobs/{job}").json()
            if body["status"] != "processing":
                return body


def main():
    with open("data/annotations/dev_types.csv") as fh:
        ann = {r["video"]: r for r in csv.DictReader(fh)}
    videos = sys.argv[1:] or DEFAULT
    key = settings.api_keys[0]
    rows = []
    for v in videos:
        a = ann[v]
        for name in ["orig", *FILTERS]:
            path = V / f"{v}.mov" if name == "orig" else transformed(v, name)
            t0 = time.time()
            res = analyze_http(path, key)
            codes = [(x.get("type_code", x["type"]), x["timecode_sec"])
                     for x in res.get("violations", [])]
            gt = a["label"] == "violation"
            ok_verdict = (res["verdict"] == "violation") == gt
            ok_type = not gt or a["type"] in {c for c, _ in codes}
            rows.append({"video": v, "variant": name, "gt": a["type"] or "ok",
                         "pred": codes, "verdict_ok": ok_verdict, "type_ok": ok_type,
                         "sec": round(time.time() - t0, 1)})
            print(f"{v:5s} {name:8s} gt={a['type'] or 'ok':26s} "
                  f"{'OK ' if ok_verdict else 'ERR'} {'type-ok' if ok_type else 'type-x'} "
                  f"{codes} {rows[-1]['sec']}s", flush=True)
    Path(".build/robustness.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for name in ["orig", *FILTERS]:
        r = [x for x in rows if x["variant"] == name]
        print(f"{name:8s} verdict {sum(x['verdict_ok'] for x in r)}/{len(r)}  "
              f"type {sum(x['type_ok'] for x in r)}/{len(r)}")


if __name__ == "__main__":
    main()
