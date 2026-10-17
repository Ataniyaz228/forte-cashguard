"""python -m datathon.cashguard serve | analyze VIDEO [--out result.json]"""

import argparse
import json
import logging
import os
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="cashguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    a = sub.add_parser("analyze")
    a.add_argument("video", type=Path)
    a.add_argument("--out", type=Path)
    a.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # weights are downloaded once; analysis must not need the internet (§7 of the brief)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.cmd == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
        return
    from .config import settings
    from .pipeline import build_analyzer

    analyzer = build_analyzer(settings)
    t0 = time.time()
    res = analyzer(args.video)
    res["processing_time_sec"] = round(time.time() - t0, 2)
    if args.no_images:
        for v in res["violations"]:
            for e in v["evidence_frames"]:
                e["image_base64"] = f"<{len(e['image_base64'])} chars>"
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
