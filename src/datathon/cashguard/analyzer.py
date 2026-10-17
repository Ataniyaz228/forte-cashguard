"""Full local pipeline: video -> signals -> rules -> jury JSON with evidence frames."""

import logging
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .detect import Detector, pick_device
from .features import Features
from .report import render
from .roi import action_roi, apply_roi
from .rules import run_rules
from .screen import screen_series
from .timeline import build_timeline
from .verify import Verifier
from .video import probe, sample_frames
from .vlm import PERIODIC_KEYS, SceneVLM, VLMConfig

log = logging.getLogger("cashguard.analyzer")
DET_FPS = 3.0
VLM_FPS = 0.5
VLM_FPS_LONG = 0.4  # videos longer than 90 s: keeps the 5-minute budget with a queue

class Analyzer:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.lock = threading.Lock()  # one GPU job at a time
        t0 = time.time()
        self.det = Detector(self.device)
        self.vlm = SceneVLM(self.device, VLMConfig(max_pixels=464 * 848, batch=7))
        log.info("models loaded on %s in %.1fs", self.device, time.time() - t0)

    # ---------------- signals ----------------
    def scan(self, path: Path) -> Features:
        info = probe(path)
        det_t, dets, scr = [], [], []
        vlm_t, vlm_imgs = [], []
        buf_t, buf_f = [], []
        next_vlm = 0.0
        vlm_step = 1.0 / (VLM_FPS_LONG if info.duration_sec > 90 else VLM_FPS)

        def flush():
            d = self.det(buf_f)
            s = screen_series(buf_f, [[x.to_list() for x in fd] for fd in d])
            det_t.extend(buf_t)
            dets.extend([[x.to_list() for x in fd] for fd in d])
            scr.extend(s)
            buf_t.clear()
            buf_f.clear()

        h = w = 0
        for t, fr in sample_frames(path, DET_FPS):
            h, w = fr.shape[:2]
            buf_t.append(round(t, 3))
            buf_f.append(fr)
            if t + 1e-6 >= next_vlm:
                vlm_t.append(round(t, 3))
                vlm_imgs.append(Image.fromarray(fr))
                next_vlm += vlm_step
            if len(buf_f) >= 24:
                flush()
        if buf_f:
            flush()
        if not det_t:
            raise ValueError("не удалось декодировать кадры видео")
        # automatic zoom: people + laptop over the video + the safe found by grounding
        ver = Verifier(self.vlm, path)
        mid = vlm_t[len(vlm_t) // 2]
        safe = ver.safe_box(mid)
        roi = action_roi(dets, h, w, [safe] if safe else None)
        P = self.vlm.yes_prob_shared([apply_roi(im, roi) for im in vlm_imgs], PERIODIC_KEYS)
        tl = build_timeline(det_t, dets, h, w, DET_FPS)
        f = Features(path.stem, info.duration_sec or det_t[-1], w, h, DET_FPS,
                     np.asarray(det_t, float), dets, tl,
                     np.array([np.nan if x is None else x for x in scr], dtype=float),
                     np.asarray(vlm_t, float),
                     {k: P[:, j] for j, k in enumerate(PERIODIC_KEYS)})
        f.roi = roi
        ver.roi = roi
        f.probe, f.probe_series = ver, ver.series
        return f

    def __call__(self, path: Path) -> dict:
        with self.lock:
            t0 = time.time()
            f = self.scan(path)
            t1 = time.time()
            cands = run_rules(f)
            out = render(path, f, cands)
            log.info("%s: scan %.1fs, total %.1fs, %d events", path.name, t1 - t0,
                     time.time() - t0, len(cands))
            return out
