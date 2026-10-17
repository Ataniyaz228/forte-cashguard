"""Analysis entry point: video -> verdict + violations in the jury JSON schema."""

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


def build_analyzer(cfg: Settings):
    if cfg.pipeline == "stub":
        return stub_analyze
    from .analyzer import Analyzer

    return Analyzer(cfg)
