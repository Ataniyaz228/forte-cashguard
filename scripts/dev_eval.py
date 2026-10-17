"""Evaluate the rules on cached features of the training videos.

Metrics are ours (no official formula is published): video-level verdict, type match of
the best-matching event, |timecode error|. Duplicates Х-47/Х-48 are skipped.
"""

import argparse
import csv
import json
from pathlib import Path

from datathon.cashguard.features import from_cache
from datathon.cashguard.rules import run_rules

ANN = Path("data/annotations/dev_types.csv")
SKIP = {"Х-47", "Х-48"}


def load_ann():
    rows = {}
    with ANN.open() as f:
        for r in csv.DictReader(f):
            r["tc"] = [float(x) for x in r["timecodes"].split(";") if x]
            rows[r["video"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--probe", action="store_true", help="run/caches 2nd-stage VLM checks")
    ap.add_argument("--probe-cache-only", action="store_true",
                    help="use cached 2nd-stage checks only (no GPU)")
    args = ap.parse_args()
    ann = load_ann()
    vlm = None
    if args.probe:
        from datathon.cashguard.vlm import SceneVLM, VLMConfig

        vlm = SceneVLM(cfg=VLMConfig(max_pixels=464 * 848, batch=6))
    res = []
    for v, a in ann.items():
        if v in SKIP or not Path(f".build/cache/vlm/{v}.json").exists():
            continue
        f = from_cache(v)
        if vlm is not None or args.probe_cache_only:
            from datathon.cashguard.verify import Verifier

            f.probe = Verifier(vlm, Path(f"data/raw/open/videos/{v}.mov"),
                               Path(f".build/cache/probe/{v}.json"))
        cands = run_rules(f)
        pred = bool(cands)
        gt = a["label"] == "violation"
        best = None
        if gt and cands:
            same = [c for c in cands if c.code == a["type"]]
            pool = same or cands
            best = min(pool, key=lambda c: min(abs(c.t - t) for t in a["tc"]))
        terr = min(abs(best.t - t) for t in a["tc"]) if best else None
        res.append({"video": v, "angle": a["angle"], "gt": gt, "pred": pred, "type": a["type"],
                    "pred_types": [f"{c.code}@{c.t:.1f}({c.score:.2f})" for c in cands],
                    "type_ok": bool(best and best.code == a["type"]), "terr": terr})
    tp = sum(r["gt"] and r["pred"] for r in res)
    fn = sum(r["gt"] and not r["pred"] for r in res)
    fp = sum(not r["gt"] and r["pred"] for r in res)
    tn = sum(not r["gt"] and not r["pred"] for r in res)
    typed = [r for r in res if r["gt"] and r["type"] != "unknown"]
    type_ok = sum(r["type_ok"] for r in typed)
    terrs = [r["terr"] for r in res if r["terr"] is not None and r["type_ok"]]
    for r in res:
        mark = "OK " if r["gt"] == r["pred"] else ("FN " if r["gt"] else "FP ")
        if args.verbose or mark != "OK " or (r["gt"] and not r["type_ok"]):
            print(f"{mark}{r['video']:5s} [{r['angle']}] gt={r['type'] or '-':26s} "
                  f"pred={r['pred_types']}")
    n = len(res)
    print(f"\nvideos={n}  TP={tp} FN={fn} FP={fp} TN={tn}  acc={(tp + tn) / max(n, 1):.3f}")
    print(f"type correct {type_ok}/{len(typed)} (known types)")
    if terrs:
        terrs.sort()
        print(f"timecode |err| for correctly typed: median={terrs[len(terrs) // 2]:.1f}s "
              f"max={terrs[-1]:.1f}s  <=2s: {sum(e <= 2 for e in terrs)}/{len(terrs)}")
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
