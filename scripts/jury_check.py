"""Simulate the jury: POST videos to a deployed service, poll, validate schema and timing.

Usage: python scripts/jury_check.py https://host [video ...] --key KEY [--parallel N]
The key can also come from CASHGUARD_API_KEY. Prints one line per video and a summary.
"""

import argparse
import base64
import concurrent.futures as cf
import io
import os
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

LIMIT_SEC = 300


def check_result(body: dict) -> list[str]:
    errs = []
    for k in ("job_id", "status", "verdict", "violations"):
        if k not in body:
            errs.append(f"нет поля {k}")
    if body.get("verdict") not in ("violation", "no_violation"):
        errs.append(f"verdict={body.get('verdict')!r}")
    v = body.get("violations", [])
    if not isinstance(v, list):
        return errs + ["violations не список"]
    if (body.get("verdict") == "violation") != bool(v):
        errs.append("verdict не согласован со списком нарушений")
    for i, x in enumerate(v):
        for k in ("type", "timecode_sec", "description", "evidence_frames"):
            if k not in x:
                errs.append(f"[{i}] нет {k}")
        if not isinstance(x.get("timecode_sec"), (int, float)):
            errs.append(f"[{i}] timecode_sec не число")
        fr = x.get("evidence_frames", [])
        if not 1 <= len(fr) <= 3:
            errs.append(f"[{i}] кадров {len(fr)}")
        for e in fr:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(e["image_base64"])))
                if img.format != "JPEG":
                    errs.append(f"[{i}] формат {img.format}")
            except Exception as ex:  # noqa: BLE001 - report any decode problem
                errs.append(f"[{i}] кадр не декодируется: {ex}")
    return errs


def run_one(base: str, key: str, path: Path) -> dict:
    t0 = time.time()
    with httpx.Client(timeout=120) as c:
        with path.open("rb") as fh:
            r = c.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key},
                       files={"file": (path.name, fh, "video/quicktime")})
        if r.status_code != 200:
            return {"video": path.name, "ok": False, "err": f"POST {r.status_code} {r.text[:200]}"}
        job = r.json()["job_id"]
        while True:
            time.sleep(2)
            g = c.get(f"{base}/api/v1/jobs/{job}", headers={"X-API-Key": key})
            body = g.json()
            el = time.time() - t0
            if body.get("status") != "processing" or el > LIMIT_SEC + 30:
                break
    status_msg = f"status={body.get('status')} {body.get('error', '')}"
    errs = [] if body.get("status") == "done" else [status_msg]
    if body.get("status") == "done":
        errs += check_result(body)
    if el > LIMIT_SEC:
        errs.append(f"дольше {LIMIT_SEC} с")
    return {"video": path.name, "ok": not errs, "err": "; ".join(errs), "sec": round(el, 1),
            "verdict": body.get("verdict"),
            "events": [(x.get("type"), x.get("timecode_sec")) for x in body.get("violations", [])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--key", default=os.environ.get("CASHGUARD_API_KEY", ""))
    ap.add_argument("--parallel", type=int, default=1)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    with httpx.Client(timeout=30) as c:
        bad = c.post(f"{base}/api/v1/analyze", files={"file": ("x.mov", b"0")}).status_code
        wrong = c.get(f"{base}/api/v1/jobs/x", headers={"X-API-Key": "wrong"}).status_code
    print(f"без ключа: {bad}, неверный ключ: {wrong} (ожидается 401)")
    if not a.videos:
        return
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(lambda p: run_one(base, a.key, p), a.videos))
    for r in res:
        print(("OK  " if r["ok"] else "ERR ") + f"{r['video']:12s} {r.get('sec', '-')}s "
              f"{r.get('verdict')} {r.get('events')} {r['err']}")
    sys.exit(0 if all(r["ok"] for r in res) else 1)


if __name__ == "__main__":
    main()
