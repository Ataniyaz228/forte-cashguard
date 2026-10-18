"""Jury JSON: types, timecodes, factual descriptions, normative references, evidence frames."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .features import Features
from .norms import TYPES
from .rules import Candidate
from .video import frames_at, to_jpeg_b64

PIPELINE_VERSION = "1.0"
ITEM_RU = {"cash_on_desk": "наличные", "bag_on_desk": "инкассаторская сумка",
           "docs_on_desk": "документы/бумаги"}

# region compared "before / after" per event type (box found by grounding in verify.py)
REGION = {"safe_open": "_safe_box", "key_in_safe": "_safe_box", "door_open": "_door_box",
          "valuables_unattended": "_desk_box", "cash_exposed_next_client": "_desk_box",
          "screen_unlocked": "_laptop_box", "food": "_desk_box", "personal_items": "_desk_box"}


def frame_marks(f: Features, t: float, code: str) -> list[dict]:
    """Boxes drawn on an evidence frame: people seen by the detector at that moment and,
    for safe-related events, the safe found by grounding. Coordinates are 0..1."""
    marks = []
    if len(f.det_t):
        i = int(abs(f.det_t - t).argmin())
        if abs(f.det_t[i] - t) <= 0.6:
            for b in f.timeline.boxes[i]:
                marks.append({"label": "человек", "kind": "person",
                              "box": [round(b[0] / f.width, 4), round(b[1] / f.height, 4),
                                      round(b[2] / f.width, 4), round(b[3] / f.height, 4)]})
    obj = {"safe_open": ("_safe_box", "сейф"), "key_in_safe": ("_safe_box", "сейф"),
           "door_open": ("_door_box", "дверь")}.get(code)
    box = getattr(f.probe, obj[0], None) if obj else None
    if isinstance(box, list):
        marks.append({"label": obj[1], "kind": "object",
                      "box": [round(box[0] / f.width, 4), round(box[1] / f.height, 4),
                              round(box[2] / f.width, 4), round(box[3] / f.height, 4)]})
    return marks


def people_per_second(f: Features) -> dict:
    """People in the room per second (max over the second) for the UI timeline."""
    n = int(f.duration) + 1
    people = [0] * n
    for t, c in zip(f.timeline.t, f.timeline.count):
        k = min(n - 1, int(t))
        people[k] = max(people[k], int(c))
    return {"t": list(range(n)), "people": people}


def exposure(f: Features, c: Candidate) -> dict | None:
    """Vulnerability window: how long the unsafe state lasted and who was in the room."""
    w = c.facts.get("window")
    if not w:
        return None
    a, b = float(w[0]), float(min(w[1], f.duration))
    m = (f.timeline.t >= a) & (f.timeline.t < b)
    people = int(f.timeline.count[m].max()) if m.any() else 0
    return {"from_sec": round(a, 1), "to_sec": round(b, 1), "seconds": round(max(0.0, b - a), 1),
            "until_end": b >= f.duration - 0.5, "max_people": people}


def before_after(path: Path, f: Features, c: Candidate, frames: dict) -> dict | None:
    """The same region at the start of the video (reference) and at the event, with the
    changed pixels (people masked out) painted red: the video is its own reference."""
    box = getattr(f.probe, REGION.get(c.code, ""), None)
    if not isinstance(box, list):
        return None
    t0 = 0.3
    t1 = round(min(max(c.evidence_t[min(1, len(c.evidence_t) - 1)], 0.0), f.duration - 0.05), 2)
    if t1 - t0 < 2.0:
        return None
    fr = frames_at(path, [t0])
    a, b = fr.get(t0), frames.get(t1)
    if a is None or b is None:
        return None
    w, h = box[2] - box[0], box[3] - box[1]
    r = (max(0, box[0] - 0.15 * w), max(0, box[1] - 0.15 * h),
         min(f.width, box[2] + 0.15 * w), min(f.height, box[3] + 0.15 * h))
    ca, cb = a.crop(r), b.crop(r)
    k = 320 / max(ca.size)
    size = (max(1, int(ca.width * k)), max(1, int(ca.height * k)))
    ca, cb = ca.resize(size, Image.LANCZOS), cb.resize(size, Image.LANCZOS)

    def gray(im):
        return np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(1.5)), float)

    diff = np.abs(gray(ca) - gray(cb)) > 25
    for t in (t0, t1):  # people moving through the region are not a change of state
        i = int(np.argmin(np.abs(f.det_t - t)))
        for d in f.dets[i]:
            if d[0] == "person" and d[1] >= 0.3:
                x0, y0, x1, y1 = d[2]
                diff[max(0, int((y0 - r[1]) * k)):max(0, int((y1 - r[1]) * k)),
                     max(0, int((x0 - r[0]) * k)):max(0, int((x1 - r[0]) * k))] = False
    if diff.mean() < 0.01:  # nothing visible changed (e.g. the reference was occupied)
        return None
    out = np.asarray(cb.convert("RGB"), float).copy()
    out[diff] = 0.45 * out[diff] + 0.55 * np.array([230, 0, 40])
    gap = 8
    pair = Image.new("RGB", (size[0] * 2 + gap, size[1]), (255, 255, 255))
    pair.paste(ca.convert("RGB"), (0, 0))
    pair.paste(Image.fromarray(out.astype(np.uint8)), (size[0] + gap, 0))
    return {"t_before": t0, "t_after": t1, "changed_share": round(float(diff.mean()), 3),
            "image_base64": to_jpeg_b64(pair)}


def fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


def describe(c: Candidate) -> str:
    t, fx = c.t, c.facts
    if c.code == "valuables_unattended":
        items = ", ".join(ITEM_RU.get(k, k) for k in fx.get("items", []))
        left = (f"на столе остаются: {items}" if items else
                "на столе остаются предметы, не убранные в хранилище (см. кадр после ухода)")
        obs = (f"В {fmt_t(t)} кассир покидает рабочее место и в {fmt_t(fx['t_exit'])} "
               f"выходит из помещения; {left}.")
    elif c.code == "safe_open" and fx.get("while") == "left":
        obs = (f"В {fmt_t(t)} кассир покидает рабочее место и выходит из помещения, "
               f"дверца сейфа остаётся открытой.")
    elif c.code == "safe_open":
        obs = f"С {fmt_t(t)} дверца сейфа открыта, при этом кассир сидит за столом."
    elif c.code == "no_client_id_check":
        after = (f" Документ появляется в кадре только в {fmt_t(fx['t_id'])}."
                 if fx.get("t_id") is not None else " Документ в кадре не появляется.")
        obs = (f"Клиент входит в {fmt_t(fx['t_enter'])}; в {fmt_t(t)} наличные уже в руках "
               f"кассира, удостоверение личности до этого не показано.{after}")
    elif c.code == "key_in_safe":
        obs = (f"В {fmt_t(t)} кассир покидает рабочее место и выходит из помещения; "
               f"в замке сейфа остаётся ключ.")
    elif c.code == "screen_unlocked":
        obs = (f"В {fmt_t(t)} кассир покидает рабочее место; после ухода экран ноутбука "
               f"остаётся включённым с открытым рабочим окном (не заблокирован).")
    elif c.code == "cash_exposed_next_client":
        prev = (f"Предыдущий посетитель выходит в {fmt_t(fx['t_prev_exit'])}; "
                if fx.get("t_prev_exit") is not None else "")
        obs = (f"{prev}в {fmt_t(t)} в кассу входит посетитель, а на столе открыто лежат "
               f"наличные.")
    elif c.code == "door_open":
        obs = (f"С {fmt_t(fx.get('t_open', t))} вид двери кассового помещения отличается от "
               f"начала записи (дверь была закрыта) и не возвращается к нему до конца записи: "
               f"дверь не закрыта и в {fmt_t(t)}.")
    elif c.code == "recount_without_client":
        obs = (f"В {fmt_t(t)} клиент выходит из кассы, а работа с наличными "
               f"(пересчёт) продолжается после его ухода.")
    elif c.code in ("personal_items", "food", "phone"):
        what = {"personal_items": "личные вещи (сумка/верхняя одежда) на рабочем столе",
                "food": "еда или напиток на рабочем месте",
                "phone": "мобильный телефон в руках у человека в кассе"}[c.code]
        a, b = fx.get("span", [t, t])
        obs = f"С {fmt_t(a)} по {fmt_t(b)} в кадре видны {what}."
    else:
        obs = f"В {fmt_t(t)} зафиксировано событие типа «{TYPES[c.code].name}»."
    vt = TYPES[c.code]
    refs = "; ".join(f"{r.title}, {r.clause}" for r in vt.rules)
    if c.level == "review":
        obs = f"Слабый сигнал, требует проверки. {obs}"
    return (f"{obs} Почему это может быть нарушением: {vt.why}. Основание: {refs}. "
            f"Требуется проверка человеком.")

def render(path: Path, f: Features, cands: list[Candidate]) -> dict:
    times = sorted({round(min(max(t, 0.0), f.duration - 0.05), 2)
                    for c in cands for t in c.evidence_t[:3]})
    frames = frames_at(path, times) if times else {}
    violations = []
    for c in cands:
        vt = TYPES[c.code]
        ev = []
        for t in c.evidence_t[:3]:
            tt = round(min(max(t, 0.0), f.duration - 0.05), 2)
            img = frames.get(tt)
            if img is not None:
                ev.append({"timestamp_sec": round(tt, 1), "image_base64": to_jpeg_b64(img),
                           "marks": frame_marks(f, tt, c.code)})
        r0 = vt.rules[0]
        violations.append({
            "type": vt.name,
            "timecode_sec": round(float(c.t), 1),
            "description": describe(c),
            "evidence_frames": ev,
            # extra fields (ignored by strict parsers, used by the UI)
            "type_code": c.code,
            "confidence": round(float(c.score), 2),
            "exposure": exposure(f, c),
            "before_after": before_after(path, f, c, frames),
            # "confirmed" or "review": a weaker signal reported so that nothing is missed
            "level": c.level,
            "rule": {"title": r0.title, "clause": r0.clause, "text": r0.text,
                     "url": r0.url},
            "rules": [{"title": r.title, "clause": r.clause, "url": r.url}
                      for r in vt.rules],
        })
    n = len(violations)
    n_rev = sum(c.level == "review" for c in cands)
    rev = f" (из них слабых сигналов «требует проверки»: {n_rev})" if n_rev else ""
    summary = ("Признаков нарушений по поддерживаемым типам не обнаружено. "
               "Отсутствие срабатывания не гарантирует отсутствие нарушений."
               if not n else
               f"Найдено событий: {n}{rev}. Каждое требует проверки человеком по кадрам "
               f"и видео.")
    return {"verdict": "violation" if n else "no_violation", "violations": violations,
            "summary": summary,
            "video": {"duration_sec": round(f.duration, 2), "width": f.width,
                      "height": f.height},
            "signals": people_per_second(f),
            "pipeline_version": PIPELINE_VERSION}

