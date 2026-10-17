"""Jury JSON: types, timecodes, factual descriptions, normative references, evidence frames."""

from pathlib import Path

from .features import Features
from .norms import TYPES
from .rules import Candidate
from .video import frames_at, to_jpeg_b64

PIPELINE_VERSION = "1.0"
ITEM_RU = {"cash_on_desk": "наличные", "bag_on_desk": "инкассаторская сумка",
           "docs_on_desk": "документы/бумаги"}


SAFE_TYPES = {"safe_open", "key_in_safe"}


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
    safe = getattr(f.probe, "_safe_box", None)
    if code in SAFE_TYPES and isinstance(safe, list):
        marks.append({"label": "сейф", "kind": "object",
                      "box": [round(safe[0] / f.width, 4), round(safe[1] / f.height, 4),
                              round(safe[2] / f.width, 4), round(safe[3] / f.height, 4)]})
    return marks


def people_per_second(f: Features) -> dict:
    """People in the room per second (max over the second) for the UI timeline."""
    n = int(f.duration) + 1
    people = [0] * n
    for t, c in zip(f.timeline.t, f.timeline.count):
        k = min(n - 1, int(t))
        people[k] = max(people[k], int(c))
    return {"t": list(range(n)), "people": people}


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
            "rule": {"title": r0.title, "clause": r0.clause, "text": r0.text,
                     "url": r0.url},
            "rules": [{"title": r.title, "clause": r.clause, "url": r.url}
                      for r in vt.rules],
        })
    n = len(violations)
    summary = ("Признаков нарушений по поддерживаемым типам не обнаружено. "
               "Отсутствие срабатывания не гарантирует отсутствие нарушений."
               if not n else
               f"Найдено событий: {n}. Каждое требует проверки человеком по кадрам и видео.")
    return {"verdict": "violation" if n else "no_violation", "violations": violations,
            "summary": summary,
            "video": {"duration_sec": round(f.duration, 2), "width": f.width,
                      "height": f.height},
            "signals": people_per_second(f),
            "pipeline_version": PIPELINE_VERSION}

