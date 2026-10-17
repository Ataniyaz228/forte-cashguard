import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _keys() -> tuple[str, ...]:
    raw = os.environ.get("CASHGUARD_API_KEYS", "")
    return tuple(k.strip() for k in raw.split(",") if k.strip())


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = field(default_factory=_keys)
    work_dir: Path = Path(os.environ.get("CASHGUARD_WORK_DIR", ".build/cashguard"))
    max_upload_mb: int = int(os.environ.get("CASHGUARD_MAX_UPLOAD_MB", "300"))
    max_duration_sec: float = float(os.environ.get("CASHGUARD_MAX_DURATION_SEC", "900"))
    workers: int = int(os.environ.get("CASHGUARD_WORKERS", "1"))
    result_ttl_hours: float = float(os.environ.get("CASHGUARD_RESULT_TTL_HOURS", "72"))
    # "full" runs every model; "stub" only decodes video (tests, smoke checks)
    pipeline: str = os.environ.get("CASHGUARD_PIPELINE", "full")
    device: str = os.environ.get("CASHGUARD_DEVICE", "auto")


settings = Settings()
