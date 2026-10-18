"""Evaluate the rules on cached features of the training videos.

Metrics are ours (no official formula is published): video-level verdict, type match of
the best-matching event, |timecode error|. Duplicates Х-47/Х-48 are skipped.
"""

import argparse
import csv
import json
import os
from pathlib import Path

from datathon.cashguard.features import from_cache
from datathon.cashguard.rules import run_rules

ANN = Path("data/annotations/dev_types.csv")
# transformed copies (scripts/robustness.py) are named "<video>__<variant>.mov"
VIDEOS = Path(os.environ.get("CASHGUARD_DEV_VIDEOS", "data/raw/open/videos"))
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
    ap.add_argument("--roi", action="store_true", help="auto-zoom caches (vlm_roi/probe_roi)")
    ap.add_argument("--probe-cache-only", action="store_true",
                    help="use cached 2nd-stage checks only (no GPU)")
    args = ap.parse_args()
    ann = load_ann()
    vlm = None
    if args.probe:
        from datathon.cashguard.vlm import SceneVLM, VLMConfig

        vlm = SceneVLM(cfg=VLMConfig(max_pixels=464 * 848, batch=6))
    res = []
    items = list(ann.items())
    if VIDEOS != Path("data/raw/open/videos"):
        items = [(p.stem, ann[p.stem.split("__")[0]]) for p in sorted(VIDEOS.glob("*.mov"))
                 if p.stem.split("__")[0] in ann]
    for v, a in items:
        from datathon.cashguard.vlm import model_tag

        sfx = "" if model_tag() == "2b" else "_" + model_tag()
        vdir = ("vlm_roi" if args.roi else "vlm") + sfx
        if v in SKIP or not Path(f".build/cache/{vdir}/{v}.json").exists():
            continue
        f = from_cache(v, vlm_dir=vdir)
        if vlm is not None or args.probe_cache_only:
            from datathon.cashguard.verify import Verifier

            pdir = ("probe_roi" if args.roi else "probe") + sfx
            ver = Verifier(vlm, VIDEOS / f"{v}.mov",
                           Path(f".build/cache/{pdir}/{v}.json"), roi=f.roi)
            f.probe, f.probe_series = ver, ver.series
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
                    "pred_strict": any(c.level == "confirmed" for c in cands),
                    "pred_types": [f"{c.code}@{c.t:.1f}({c.score:.2f})"
                                   + ("?" if c.level == "review" else "") for c in cands],
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
    print(f"\nvideos={n}  TP={tp} FN={fn} FP={fp} TN={tn}  acc={(tp + tn) / max(n, 1):.3f}"
          f"  precision={tp / max(tp + fp, 1):.3f} recall={tp / max(tp + fn, 1):.3f}")
    # confirmed-only operating point ("?" = review-level events, recall-first mode)
    s_tp = sum(r["gt"] and r["pred_strict"] for r in res)
    s_fp = sum(not r["gt"] and r["pred_strict"] for r in res)
    s_fn = sum(r["gt"] for r in res) - s_tp
    s_tn = sum(not r["gt"] for r in res) - s_fp
    print(f"confirmed only: TP={s_tp} FN={s_fn} FP={s_fp} TN={s_tn}  "
          f"acc={(s_tp + s_tn) / max(n, 1):.3f}  precision={s_tp / max(s_tp + s_fp, 1):.3f} "
          f"recall={s_tp / max(s_tp + s_fn, 1):.3f}")
    print(f"type correct {type_ok}/{len(typed)} (known types)")
    if terrs:
        terrs.sort()
        print(f"timecode |err| for correctly typed: median={terrs[len(terrs) // 2]:.1f}s "
              f"max={terrs[-1]:.1f}s  <=2s: {sum(e <= 2 for e in terrs)}/{len(terrs)}")
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
