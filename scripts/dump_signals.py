"""Print compact per-video signal strips (one digit = P*9 per VLM sample)."""

import csv
import json
import sys
from pathlib import Path

with open("data/annotations/dev_types.csv") as fh:
    ann = {r["video"]: r for r in csv.DictReader(fh)}
keys = sys.argv[1].split(",") if len(sys.argv) > 1 else None
for p in sorted(Path(".build/cache/vlm").glob("*.json")):
    d = json.loads(p.read_text())
    a = ann.get(d["video"], {})
    print(f"{d['video']:5s} {a.get('type') or 'ok':24s} @{a.get('timecodes', '')}")
    for j, k in enumerate(d["keys"]):
        if keys and k not in keys:
            continue
        strip = "".join(str(min(9, int(row[j] * 10))) for row in d["p"])
        print(f"   {k:15s}{strip}")
