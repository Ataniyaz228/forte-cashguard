"""In-process job queue. Uploaded videos are deleted right after analysis."""

import json
import logging
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("cashguard.jobs")

Analyzer = Callable[[Path], dict]


class JobStore:
    def __init__(self, work_dir: Path, analyzer: Analyzer, workers: int = 1,
                 ttl_hours: float = 72):
        self.dir = work_dir
        (self.dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.dir / "results").mkdir(parents=True, exist_ok=True)
        self.analyzer = analyzer
        self.ttl = ttl_hours * 3600
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analyze")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self._load_existing()

    def _result_path(self, job_id: str) -> Path:
        return self.dir / "results" / f"{job_id}.json"

    def _load_existing(self) -> None:
        now = time.time()
        for p in (self.dir / "results").glob("*.json"):
            try:
                job = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if now - job.get("_created", now) > self.ttl:
                p.unlink(missing_ok=True)
                continue
            if job.get("status") == "processing":  # interrupted by restart
                job.update(status="failed", error="обработка прервана перезапуском сервиса")
            self.jobs[job["job_id"]] = job
        for p in (self.dir / "uploads").glob("*"):
            p.unlink(missing_ok=True)

    def _save(self, job: dict) -> None:
        tmp = self._result_path(job["job_id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False))
        tmp.replace(self._result_path(job["job_id"]))

    def new_upload_path(self, suffix: str) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        return job_id, self.dir / "uploads" / f"{job_id}{suffix}"

    def submit(self, job_id: str, video: Path, filename: str, source: str) -> dict:
        job = {"job_id": job_id, "status": "processing", "_created": time.time(),
               "_filename": filename, "_source": source}
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
        self.pool.submit(self._run, job_id, video)
        return job

    def _run(self, job_id: str, video: Path) -> None:
        t0 = time.time()
        try:
            result = self.analyzer(video)
            update = {"status": "done", **result}
        except Exception as e:  # noqa: BLE001 - any analyzer failure -> job "failed"
            log.error("job %s failed: %s\n%s", job_id, e, traceback.format_exc())
            update = {"status": "failed", "error": str(e) or e.__class__.__name__}
        finally:
            video.unlink(missing_ok=True)
        update["processing_time_sec"] = round(time.time() - t0, 2)
        with self.lock:
            job = self.jobs[job_id]
            job.update(update)
            self._save(job)
        log.info("job %s %s in %.1fs", job_id, job["status"], job["processing_time_sec"])

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def queue_position(self, job_id: str) -> int:
        with self.lock:
            pending = sorted((j["_created"], j["job_id"]) for j in self.jobs.values()
                             if j["status"] == "processing")
        ids = [j for _, j in pending]
        return ids.index(job_id) if job_id in ids else -1
