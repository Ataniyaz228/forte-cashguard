import base64
import subprocess

import numpy as np
import pytest

from datathon.cashguard.features import Features
from datathon.cashguard.norms import TYPES
from datathon.cashguard.report import render
from datathon.cashguard.rules import Candidate
from datathon.cashguard.timeline import build_timeline


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "v.mov"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=232x424:rate=30", "-t", "6", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", str(p)], check=True)
    return p


def feats(duration=6.0):
    t = np.arange(0, duration, 1 / 3)
    tl = build_timeline(list(t), [[] for _ in t], 424, 232, 3.0)
    return Features("v", duration, 232, 424, 3.0, t, [[] for _ in t], tl,
                    np.full(len(t), np.nan), np.arange(0, duration, 2.0), {})


def test_every_type_has_rule_and_text():
    for vt in TYPES.values():
        assert vt.rules and vt.name and vt.why
        for r in vt.rules:
            assert r.title and r.clause and r.text and r.url.startswith("https://")


def test_render_schema(video):
    cands = [Candidate("valuables_unattended", 2.0, 0.9, [2.0, 5.9],
                       {"items": ["cash_on_desk"], "t_exit": 3.0}),
             Candidate("food", 1.0, 0.8, [1.0], {"span": [1.0, 4.0]})]
    out = render(video, feats(), cands)
    assert out["verdict"] == "violation" and len(out["violations"]) == 2
    for v in out["violations"]:
        assert isinstance(v["timecode_sec"], float)
        assert 1 <= len(v["evidence_frames"]) <= 3
        jpg = base64.b64decode(v["evidence_frames"][0]["image_base64"])
        assert jpg[:2] == b"\xff\xd8"
        assert "Основание" in v["description"] and v["rule"]["clause"]


def test_render_no_events(video):
    out = render(video, feats(), [])
    assert out["verdict"] == "no_violation" and out["violations"] == []
