import subprocess
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from datathon.cashguard.app import create_app
from datathon.cashguard.config import Settings

KEY = "test-key-123"


@pytest.fixture(scope="module")
def sample_mov(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "sample.mov"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=232x424:rate=30",
         "-t", "3", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)], check=True)
    return path


def fake_analyzer(path):
    return {
        "verdict": "violation",
        "violations": [{"type": "Тест", "timecode_sec": 1.5, "description": "d",
                        "evidence_frames": [{"timestamp_sec": 1.5, "image_base64": "AAAA"}]}],
        "summary": "s",
    }


@pytest.fixture()
def client(tmp_path):
    cfg = replace(Settings(), api_keys=(KEY,), work_dir=tmp_path, pipeline="stub")
    with TestClient(create_app(cfg, analyzer=fake_analyzer)) as c:
        yield c


def wait_done(client, job_id, headers, url="/api/v1/jobs/"):
    for _ in range(100):
        r = client.get(url + job_id, headers=headers)
        if r.json()["status"] != "processing":
            return r
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_missing_and_wrong_key_rejected(client, sample_mov):
    files = {"file": ("v.mov", sample_mov.read_bytes(), "video/quicktime")}
    assert client.post("/api/v1/analyze", files=files).status_code == 401
    r = client.post("/api/v1/analyze", files=files, headers={"X-API-Key": "nope"})
    assert r.status_code == 401
    assert client.get("/api/v1/jobs/abc").status_code == 401


def test_contract_roundtrip(client, sample_mov):
    h = {"X-API-Key": KEY}
    r = client.post("/api/v1/analyze", headers=h,
                    files={"file": ("video_017.mov", sample_mov.read_bytes(), "video/quicktime")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    body = wait_done(client, job_id, h).json()
    assert set(body) == {"job_id", "status", "verdict", "violations"}
    assert body["status"] == "done" and body["verdict"] == "violation"
    v = body["violations"][0]
    assert {"type", "timecode_sec", "description", "evidence_frames"} <= set(v)
    assert {"timestamp_sec", "image_base64"} <= set(v["evidence_frames"][0])
    # upload is deleted after processing
    assert not list((client.app.state.store.dir / "uploads").iterdir())


def test_bad_files(client):
    h = {"X-API-Key": KEY}
    r = client.post("/api/v1/analyze", headers=h, files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/v1/analyze", headers=h,
                    files={"file": ("x.mov", b"not a video at all", "video/quicktime")})
    assert r.status_code == 400


def test_unknown_job(client):
    assert client.get("/api/v1/jobs/doesnotexist", headers={"X-API-Key": KEY}).status_code == 404


def test_failed_job_reported(tmp_path, sample_mov):
    def boom(path):
        raise RuntimeError("model crashed")

    cfg = replace(Settings(), api_keys=(KEY,), work_dir=tmp_path)
    with TestClient(create_app(cfg, analyzer=boom)) as c:
        h = {"X-API-Key": KEY}
        job_id = c.post("/api/v1/analyze", headers=h,
                        files={"file": ("v.mov", sample_mov.read_bytes())}).json()["job_id"]
        body = wait_done(c, job_id, h).json()
        assert body["status"] == "failed" and "model crashed" in body["error"]


def test_ui_routes_without_key(client, sample_mov):
    r = client.post("/ui/analyze", files={"file": ("v.mov", sample_mov.read_bytes())})
    assert r.status_code == 200
    body = wait_done(client, r.json()["job_id"], {}, url="/ui/jobs/").json()
    assert body["status"] == "done" and body["summary"] == "s"
    assert client.get("/").status_code == 200


def test_any_extension_accepted(client, sample_mov):
    h = {"X-API-Key": KEY}
    for name in ("video", "clip.MP4", "cam.bin"):
        r = client.post("/api/v1/analyze", headers=h, files={"file": (name, sample_mov.read_bytes())})
        assert r.status_code == 200, name
