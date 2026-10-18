"""HTTP API (jury contract) and the web UI."""

import logging
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, settings
from .jobs import JobStore
from .video import VideoError, probe

log = logging.getLogger("cashguard")
STATIC = Path(__file__).parent / "static"
KNOWN_SUFFIXES = {".mov", ".mp4", ".m4v", ".qt", ".avi", ".mkv", ".webm", ".mts", ".m2ts"}
PUBLIC_FIELDS = ("job_id", "status", "verdict", "violations")


def public_view(job: dict, extended: bool = False) -> dict:
    out = {k: job[k] for k in PUBLIC_FIELDS if k in job}
    if job["status"] == "failed":
        out["error"] = job.get("error", "")
    if extended:
        for k in ("summary", "processing_time_sec", "video", "pipeline_version", "signals",
                  "stage", "stage_progress"):
            if k in job:
                out[k] = job[k]
        out["filename"] = job.get("_filename")
    return out


def create_app(cfg: Settings = settings, analyzer=None) -> FastAPI:
    if analyzer is None:
        from .pipeline import build_analyzer

        analyzer = build_analyzer(cfg)
    store = JobStore(cfg.work_dir, analyzer, cfg.workers, cfg.result_ttl_hours)
    app = FastAPI(title="CashGuard", version="1.0",
                  description="Подсказки для проверки кассовой дисциплины по видео")
    app.state.store = store

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not cfg.api_keys:
            raise HTTPException(503, "API-ключ не настроен на сервере")
        if not x_api_key or not any(secrets.compare_digest(x_api_key, k)
                                    for k in cfg.api_keys):
            raise HTTPException(401, "неверный или отсутствующий X-API-Key")

    async def accept_upload(file: UploadFile, source: str) -> dict:
        name = file.filename or "video.mov"
        # any name/extension is accepted: the decoder decides whether it is a video
        suffix = Path(name).suffix.lower()
        if suffix not in KNOWN_SUFFIXES:
            suffix = ".mov"
        job_id, path = store.new_upload_path(suffix)
        limit = cfg.max_upload_mb * 1024 * 1024
        size = 0
        try:
            with path.open("wb") as f:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"файл больше {cfg.max_upload_mb} МБ")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "пустой файл")
            info = probe(path)
            limit_sec = cfg.max_duration_sec if source == "api" else cfg.ui_max_duration_sec
            if info.duration_sec > limit_sec:
                raise HTTPException(400, f"видео длиннее {limit_sec:.0f} с")
        except VideoError as e:
            path.unlink(missing_ok=True)
            raise HTTPException(400, str(e)) from e
        except HTTPException:
            path.unlink(missing_ok=True)
            raise
        job = store.submit(job_id, path, name, source)
        log.info("job %s accepted from %s: %s, %.1f MB", job_id, source, name, size / 2**20)
        return {"job_id": job["job_id"], "status": job["status"]}

    @app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
    async def analyze(file: UploadFile = File(...)):  # noqa: B008 - FastAPI idiom
        return JSONResponse(await accept_upload(file, "api"), status_code=200)

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def get_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "задача не найдена")
        return public_view(job)

    # UI routes: the customer's reviewer needs no key (FAQ 27), jury API stays keyed.
    @app.post("/ui/analyze")
    async def ui_analyze(file: UploadFile = File(...)):  # noqa: B008 - FastAPI idiom
        # the public page shares the GPU with the jury API: no UI uploads behind a long queue
        if store.processing() >= cfg.workers + 1:
            raise HTTPException(429, "Сервер сейчас занят проверкой записей, попробуйте "
                                     "через минуту")
        return JSONResponse(await accept_upload(file, "ui"), status_code=200)

    @app.get("/ui/jobs/{job_id}")
    def ui_job(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "задача не найдена")
        out = public_view(job, extended=True)
        out["queue_position"] = store.queue_position(job_id)
        return out

    @app.get("/ui/types")
    def ui_types():
        from .norms import TYPES
        from .rules import REPORTED_TYPES

        return [{"code": c, "name": TYPES[c].name, "why": TYPES[c].why,
                 "rules": [{"title": r.title, "clause": r.clause, "url": r.url}
                           for r in TYPES[c].rules]} for c in REPORTED_TYPES]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
