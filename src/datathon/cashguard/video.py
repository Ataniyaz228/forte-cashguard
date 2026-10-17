"""Video probing, timestamp-accurate sampling and evidence-frame encoding."""

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str


def probe(path: Path) -> VideoInfo:
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise VideoError("в файле нет видеопотока")
            s = c.streams.video[0]
            if s.duration is not None and s.time_base is not None:
                duration = float(s.duration * s.time_base)
            elif c.duration is not None:
                duration = c.duration / av.time_base
            else:
                duration = 0.0
            fps = float(s.average_rate) if s.average_rate else 0.0
            w, h = s.codec_context.width, s.codec_context.height
            codec = s.codec_context.name
    except VideoError:
        raise
    except Exception as e:  # av raises many container-specific errors
        raise VideoError(f"не удалось прочитать видео: {e}") from e
    if w <= 0 or h <= 0:
        raise VideoError("некорректный размер кадра")
    return VideoInfo(duration, fps, w, h, codec)


def _oriented(frame: av.VideoFrame) -> Image.Image:
    img = frame.to_image()
    rot = getattr(frame, "rotation", 0) or 0
    if rot:
        img = img.rotate(rot, expand=True)
    return img


def sample_frames(path: Path, fps: float, max_side: int | None = None):
    """Yield (timestamp_sec, RGB uint8 array) at roughly `fps`, using real frame PTS."""
    step = 1.0 / fps
    next_t = 0.0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        start = float(s.start_time * s.time_base) if s.start_time is not None else 0.0
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            t = float(frame.pts * s.time_base) - start
            if t + 1e-6 < next_t:
                continue
            img = _oriented(frame)
            if max_side and max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            yield t, np.asarray(img)
            next_t = (int(t / step) + 1) * step


def frames_at(path: Path, times: list[float]) -> dict[float, Image.Image]:
    """Full-resolution frames closest to requested times (single decode pass)."""
    want = sorted(set(times))
    out: dict[float, Image.Image] = {}
    if not want:
        return out
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        start = float(s.start_time * s.time_base) if s.start_time is not None else 0.0
        i = 0
        prev = None
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            t = float(frame.pts * s.time_base) - start
            while i < len(want) and t >= want[i]:
                pick = frame
                if prev is not None and abs(prev[0] - want[i]) < abs(t - want[i]):
                    pick = prev[1]
                out[want[i]] = _oriented(pick)
                i += 1
            if i >= len(want):
                break
            prev = (t, frame)
        if i < len(want) and prev is not None:
            for tt in want[i:]:
                out[tt] = _oriented(prev[1])
    return out


def to_jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
