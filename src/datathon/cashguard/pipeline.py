"""Analysis entry point: video -> verdict + violations in the jury JSON schema."""

import queue
from pathlib import Path

from .config import Settings
from .video import probe

PIPELINE_VERSION = "0.1-stub"


def stub_analyze(path: Path) -> dict:
    info = probe(path)
    return {
        "verdict": "no_violation",
        "violations": [],
        "summary": "Заглушка: модели не запускались.",
        "video": {"duration_sec": round(info.duration_sec, 2), "fps": round(info.fps, 2),
                  "width": info.width, "height": info.height},
        "pipeline_version": PIPELINE_VERSION,
    }


class AnalyzerPool:
    """N independent model copies; each job takes a free one (parallel videos on a big GPU)."""

    def __init__(self, analyzers: list):
        self.free: queue.Queue = queue.Queue()
        for a in analyzers:
            self.free.put(a)

    def __call__(self, path: Path, progress=None, waited: float = 0.0) -> dict:
        a = self.free.get()
        try:
            return a(path, progress=progress, waited=waited)
        finally:
            self.free.put(a)


def build_analyzer(cfg: Settings):
    if cfg.pipeline == "stub":
        return stub_analyze
    from .analyzer import Analyzer

    if cfg.workers <= 1:
        return Analyzer(cfg)
    return AnalyzerPool([Analyzer(cfg) for _ in range(cfg.workers)])
