"""Temporal rules: signals -> candidate violation events.

Every threshold below was chosen on the training videos (see docs/EXPERIMENTS.csv) and is
kept in one place (TH) so it can be re-tuned without touching the logic.
"""

import os
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from .features import Features

TH = {
    "stand_ratio": 1.3,       # standing box height / seated height
    "absence_min": 1.5,       # seconds without people to accept an exit
    "absence_sure": 3.0,      # ...or this long without people even with no posture cue
    "absence_end": 2.0,       # ...or until the video ends (short tail)
    "post_exit": 4.5,         # seconds after exit inspected for left-behind objects
    "valuables": 0.6,         # P(cash/bag/docs on desk) in the empty room
    "safe_open": 0.5,         # P(safe door open)
    "screen_lit": 0.12,       # fallback pixel heuristic (only if the laptop is not found)
    "screen_app": 0.7,        # VLM on the laptop crop: violations 0.97-1.0, normals <=0.22       # lit fraction of the laptop box = active screen
    "periodic": 0.6,          # personal items / food / phone
    "periodic_min_hits": 2,
    "cash_next": 0.6,
    "entry_min_t": 5.0,
    "client_window": 18.0,     # seconds after a client enters that are checked for ID/cash
    "client_cash": 0.6,
    "client_id": 0.5,         # cash on desk when the next visitor enters
    "count_after_client": 0.5,
    "id_shown": 0.3,
    "counting": 0.5,
    "bag": 0.8,
    "desk_items": 0.3,
    "desk_crop": 0.8,          # zoomed desk: violations >=0.95 on all frames, normals <=0.62         # targeted: anything left on the desk after the exit
    "key_in_lock": 0.3,        # max over 4 post-exit lock crops (E005/E006: weak signal)
    "safe_seated_samples": 3,
    "safe_open_long": 15.0,    # or stays open until the video ends
    "det_support_min": 3,
    "stranger_sec": 5.0,       # newcomer keeps standing this long (or until the video ends)      # detector frames with cup/bottle or handbag/backpack  # open safe while seated for >= 3 VLM samples (~4+ s)
    "recount_min_sec": 10.0,  # opened bag must be recounted at least this long
}


# types the active rules can report (UI catalogue, docs); the rest are documented limits
REPORTED_TYPES = ("safe_open", "key_in_safe", "valuables_unattended", "screen_unlocked",
                  "cash_exposed_next_client", "no_courier_check", "no_recount", "stranger",
                  "food", "personal_items")


@dataclass
class Candidate:
    code: str
    t: float
    score: float
    evidence_t: list[float]
    facts: dict = field(default_factory=dict)


@dataclass
class Departure:
    t_exit: float      # last moment a person is seen
    t_depart: float    # moment the cashier leaves the workplace (event timecode)
    t_back: float      # when someone re-appears (or video end)


def _seated_height(f: Features) -> float:
    h = f.person_heights()
    h = h[~np.isnan(h)]
    return float(np.percentile(h, 10)) if len(h) else 0.0


def departures(f: Features) -> list[Departure]:
    tl = f.timeline
    h = f.person_heights()
    seated = _seated_height(f)
    out = []
    for e in tl.events:
        if e.kind != "exit_all":
            continue
        later = [x.t for x in tl.events if x.t > e.t and x.kind == "enter_first"]
        back = later[0] if later else f.duration
        if back - e.t < TH["absence_min"] and back < f.duration - 0.5:
            continue
        # evidence of really leaving: standing up right before disappearing (camera-dependent
        # box height ratio) OR staying out of the room for long (camera-independent)
        m = (f.det_t >= e.t - 2.0) & (f.det_t <= e.t)
        hh = h[m]
        hh = hh[~np.isnan(hh)]
        stood = not seated or (len(hh) > 0 and hh.max() >= TH["stand_ratio"] * seated)
        long_gone = (back - e.t) >= TH["absence_sure"] or (
            back >= f.duration - 0.5 and back - e.t >= TH["absence_end"])
        if not (stood or long_gone):
            continue
        # someone must have been in the room for a while (not a person just walking in)
        first_seen = f.det_t[np.argmax(tl.count > 0)] if (tl.count > 0).any() else e.t
        if e.t - first_seen < 4.0:
            continue
        # stand-up moment: walk back from the exit while the person is standing
        idx = np.where(f.det_t <= e.t)[0]
        stand_t = e.t
        for i in idx[::-1]:
            if np.isnan(h[i]) or h[i] >= TH["stand_ratio"] * seated:
                stand_t = f.det_t[i]
            else:
                break
        # labels mark the moment the cashier leaves the workplace: right after standing up
        # if she walks straight out, otherwise ~3 s before she disappears through the door
        if not stood:  # no posture cue: the cashier leaves the workplace ~2 s before the exit
            depart = e.t - 2.0
        elif e.t - stand_t <= 8.0:
            depart = min(stand_t + 1.5, e.t)
        else:
            depart = e.t - 3.0
        out.append(Departure(e.t, round(float(depart), 2), back))
    return out



def _close_gaps(mask: np.ndarray, gap: int = 1) -> np.ndarray:
    """Fill runs of up to `gap` False samples that sit between True samples."""
    m = mask.copy()
    idx = np.where(mask)[0]
    for a, b in pairwise(idx):
        if 1 < b - a <= gap + 1:
            m[a:b] = True
    return m


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def rule_departures(f: Features) -> list[Candidate]:
    out = []
    for d in departures(f):
        t0, t1 = d.t_exit + 0.3, min(d.t_back, d.t_exit + TH["post_exit"])
        t1 = max(t1, t0)
        last = round(min(t1, f.duration - 0.1), 2)
        probe_t = sorted({round(min(d.t_exit + 1.0, last), 2),
                          round(min(d.t_exit + 2.0, last), 2), last})
        periodic = {k: f.vwin(k, t0, t1) for k in ("cash_on_desk", "bag_on_desk",
                                                   "docs_on_desk", "safe_open")}
        ev = [d.t_depart, last]
        items = [k for k in ("cash_on_desk", "bag_on_desk", "docs_on_desk")
                 if periodic[k] >= TH["valuables"]]
        if f.probe is not None:
            # scene view OR a zoomed desk crop (robust to a farther camera); the crop is more
            # sensitive, so it has its own threshold and needs two frames (E021)
            p_scene = f.probe("desk_items", probe_t)
            crop_vals = sorted((f.probe("desk_left", [t]) for t in probe_t), reverse=True)
            p_crop = crop_vals[1] if len(crop_vals) > 1 else 0.0
            p_cash = max(f.probe("cash", probe_t), f.probe("desk_cash", probe_t))
            if p_cash >= 0.5 and "cash_on_desk" not in items:
                items.insert(0, "cash_on_desk")
            left = p_scene >= TH["desk_items"] or p_crop >= TH["desk_crop"]
            score = max(p_scene, p_crop)
        else:
            left = bool(items)
            score = max([periodic[k] for k in items] or [0.0])
        if left:
            out.append(Candidate("valuables_unattended", d.t_depart, score, ev,
                                 {"items": items, "t_exit": d.t_exit}))
        safe_open = periodic["safe_open"] >= TH["safe_open"]
        if safe_open and f.probe is not None:
            safe_open = f.probe("safe_open", probe_t) >= TH["safe_open"]
        if safe_open:
            out.append(Candidate("safe_open", d.t_depart, periodic["safe_open"], ev,
                                 {"t_exit": d.t_exit, "while": "left"}))
        elif f.probe is not None:
            key_t = sorted({round(min(d.t_exit + k * 0.7, last), 2) for k in (1, 2, 3, 4)})
            # a key must be seen on at least two frames (single dark-frame spikes, Х-12)
            vals = sorted((f.probe("key_in_lock", [t]) for t in key_t), reverse=True)
            p_key = vals[1] if len(vals) > 1 else 0.0
            if p_key >= TH["key_in_lock"]:
                out.append(Candidate("key_in_safe", d.t_depart, p_key, ev,
                                     {"t_exit": d.t_exit}))
        # screen left unlocked: the VLM reads the zoomed laptop screen (open program vs lock
        # screen / wallpaper / off), two frames needed (E022). The pixel colour heuristic is
        # only a fallback when the laptop cannot be located at all.
        scr = f.screen_win(d.t_exit, min(d.t_back, d.t_exit + 3.0))
        app_vals = (sorted((f.probe("screen_app", [t]) for t in probe_t), reverse=True)
                    if f.probe is not None else [])
        p_app = app_vals[1] if len(app_vals) > 1 else 0.0
        laptop_found = getattr(f.probe, "_laptop_box", None) not in (None, "unset")
        unlocked = p_app >= TH["screen_app"] or (
            not laptop_found and scr is not None and scr >= TH["screen_lit"])
        if unlocked:
            out.append(Candidate("screen_unlocked", d.t_depart,
                                 max(p_app, min(1.0, 0.5 + (scr or 0.0))), ev,
                                 {"t_exit": d.t_exit, "screen": round(scr or 0.0, 3),
                                  "screen_vlm": round(p_app, 3)}))
    return out


def rule_safe_open_seated(f: Features) -> list[Candidate]:
    """Safe door open while the cashier is seated at the desk (not working at the safe)."""
    h = f.person_heights()
    seated = _seated_height(f)
    if not seated:
        return []
    # one missed sample inside a run does not break it (VLM answers flicker on single frames)
    is_open = _close_gaps(f.v("safe_open") >= TH["safe_open"])
    out = []
    for a, b in _runs(is_open):
        times = f.vlm_t[a:b]
        seated_times = []
        for t in times:
            m = np.abs(f.det_t - t) <= 0.5
            hh = h[m]
            hh = hh[~np.isnan(hh)]
            if len(hh) and np.median(hh) < 1.15 * seated and f.timeline.count_at(t) >= 1:
                seated_times.append(float(t))
        reaches_end = b >= len(is_open) - 1
        long_open = (f.vlm_t[b - 1] - f.vlm_t[a]) >= TH["safe_open_long"]
        if len(seated_times) >= TH["safe_seated_samples"] and (reaches_end or long_open):
            t = seated_times[0]
            if f.probe is not None and f.probe("safe_open", seated_times[:2]) < TH["safe_open"]:
                continue
            out.append(Candidate("safe_open", t, float(f.v("safe_open")[a:b].max()),
                                 [t, float(times[-1])], {"while": "seated"}))
    return out


def rule_periodic(f: Features) -> list[Candidate]:
    out = []
    # "phone" is not reported: clients routinely hold phones (Х-5) and the VLM confuses
    # cards/banknotes with phones; detector support is required for the other two.
    support = _det_support(f)
    for code, key in (("personal_items", "personal_items"), ("food", "food")):
        if support[code] < TH["det_support_min"]:
            continue
        p = f.v(key).copy()
        if code == "personal_items":  # the green cash bag is not a personal item
            p[f.v("bag_on_desk") >= 0.5] = 0.0
        hits = np.where(p >= TH["periodic"])[0]
        if len(hits) >= TH["periodic_min_hits"]:
            # the longest uninterrupted run = the object is really there, not passing by
            a, b = max(_runs(p >= TH["periodic"]), key=lambda r: r[1] - r[0])
            t = float(f.vlm_t[a])
            ev = [t] + ([float(f.vlm_t[(a + b) // 2])] if b - a > 2 else [])
            out.append(Candidate(code, t, float(np.sort(p[hits])[-2]), ev,
                                 {"hits": len(hits), "span": [t, float(f.vlm_t[b - 1])]}))
    return out


def _det_support(f: Features) -> dict[str, int]:
    food = {"cup", "bottle", "wine glass", "bowl", "sandwich", "donut", "cake", "banana",
            "apple"}
    personal = {"handbag", "backpack"}
    n = {"food": 0, "personal_items": 0}
    for fd in f.dets:
        labs = {d[0] for d in fd if d[1] >= 0.5}
        n["food"] += bool(labs & food)
        n["personal_items"] += bool(labs & personal)
    return n


def client_exits(f: Features, hold: float = 2.5) -> list[float]:
    """Someone leaves while the cashier stays; the lower count must hold for `hold` s."""
    out = []
    tl = f.timeline
    for e in tl.events:
        if e.kind != "count_down" or e.after < 1:
            continue
        m = (tl.t > e.t) & (tl.t <= e.t + hold)
        if m.any() and np.all(tl.count[m] <= e.after):
            out.append(e.t)
    return out


def visitor_entries(f: Features) -> list[float]:
    return [e.t for e in f.timeline.events if e.kind == "count_up" and e.before >= 1]


def rule_client_flow(f: Features) -> list[Candidate]:
    # "recount without the client" is not emitted: in our data the VLM cannot tell an
    # ongoing recount from an idle counting machine (camera C), see docs/EXPERIMENTS.csv.
    out = []
    exits = client_exits(f)
    alone_entries = [e.t for e in f.timeline.events
                     if e.kind == "count_up" and e.before == 1]
    for ti in alone_entries:
        # the cashier is alone (previous visitor gone) and the next one walks in while cash
        # still lies on the desk; 2->3 jumps are flicker, the opening scene is skipped
        if ti < TH["entry_min_t"]:
            continue
        prev = [t for t in exits if t < ti]
        if f.probe is not None:
            pc = f.probe("cash", [round(max(0.0, ti - 1.0), 2), ti, round(ti + 1.5, 2),
                                  round(min(f.duration - 0.1, ti + 3.0), 2)])
        else:
            pc = f.vwin("cash_on_desk", ti - 1.0, ti + 3.0)
        if pc >= TH["cash_next"]:
            out.append(Candidate("cash_exposed_next_client", ti, pc, [ti, ti + 1.5],
                                 {"t_prev_exit": prev[-1] if prev else None}))
    return out


def rule_cash_bag(f: Features) -> list[Candidate]:
    """Cash-in-transit bag: courier check on hand-over, recount after opening."""
    out = []
    bag = f.v("bag_on_desk") >= TH["bag"]
    if not bag.any():
        return out
    i0 = int(np.argmax(bag))
    tb = float(f.vlm_t[i0])
    courier = f.timeline.count_at(tb) >= 2 or f.timeline.count_at(tb + 2) >= 2
    if courier:
        leaves = [e.t for e in f.timeline.events
                  if e.kind == "count_down" and tb < e.t <= tb + 20]
        t_end = leaves[0] if leaves else min(f.duration, tb + 15)
        pid = f.vwin("id_shown", tb - 2, t_end)
        if f.probe_series is not None:  # denser targeted check of the whole visit
            ts = [round(t, 2) for t in np.arange(max(0.0, tb - 4), t_end, 1.0)]
            if ts:
                # an ID must be seen on at least two frames (single-frame confusions with
                # the bag or a card in hand, Х-17)
                vals = sorted(f.probe_series(["id_card"], ts)["id_card"], reverse=True)
                pid = vals[1] if len(vals) > 1 else 0.0
        if pid < TH["id_shown"]:
            t = max(tb, t_end - 2.0) if leaves else tb + 3.0
            out.append(Candidate("no_courier_check", round(t, 2), 1.0 - pid, [tb, t],
                                 {"t_bag": tb, "t_courier_exit": leaves[0] if leaves else None}))
        return out
    # cashier alone with an opened bag: cash taken out, then put away
    cash = f.v("cash_on_desk") >= 0.5
    after = np.where(cash & (f.vlm_t > tb))[0]
    if not len(after):
        return out
    last_cash = after[-1]
    safe_after = np.where((f.v("safe_open") >= TH["safe_open"]) &
                          (f.vlm_t > f.vlm_t[after[0]]))[0]
    # evidence of a recount: time spent counting OR handling the cash on the desk (a VLM
    # often misses "counting", but a real recount keeps the cash on the desk for long:
    # Х-14 ~50 s vs Х-13 ~4 s before the safe)
    period = 1.0 / max(_vlm_fps(f), 1e-6)
    counting = (f.v("counting") >= TH["counting"]) & (f.vlm_t >= tb)
    handled = cash & (f.vlm_t >= tb)
    work_sec = max(int(counting.sum()), int(handled.sum())) * period
    if work_sec < TH["recount_min_sec"]:
        t = float(f.vlm_t[safe_after[0]]) if len(safe_after) else float(f.vlm_t[last_cash])
        out.append(Candidate("no_recount", t, 0.7, [float(f.vlm_t[after[0]]), t],
                             {"t_bag": tb, "count_sec": round(work_sec, 1)}))
    return out


def _vlm_fps(f: Features) -> float:
    return 1.0 / float(np.median(np.diff(f.vlm_t))) if len(f.vlm_t) > 1 else 0.5


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _min_dist(b, others) -> float:
    cx, cy = _center(b)
    return min(np.hypot(cx - _center(o)[0], cy - _center(o)[1]) for o in others)


def rule_stranger(f: Features) -> list[Candidate]:
    """A newcomer who keeps standing next to the cashier (clients sit down within a few
    seconds, couriers come with a cash bag)."""
    out = []
    seated = _seated_height(f)
    if not seated or (f.v("bag_on_desk") >= TH["bag"]).any():
        return out
    tl = f.timeline
    for e in tl.events:
        if e.kind != "count_up" or e.before < 1:
            continue
        i = int(np.argmin(np.abs(f.det_t - e.t)))
        prev = [b for k in range(max(0, i - 3), i) for b in tl.boxes[k]]
        cur = tl.boxes[i]
        if not prev or len(cur) < 2:
            continue
        # newcomer = current box farthest from every box seen just before
        nb = max(cur, key=lambda b, prev=prev: _min_dist(b, prev))
        t_end = min(f.duration, e.t + TH["stranger_sec"])
        if t_end - e.t < 2.5:
            continue
        standing, total = 0, 0
        for k in range(i, len(f.det_t)):
            if f.det_t[k] > t_end:
                break
            if not tl.boxes[k]:
                continue
            b = min(tl.boxes[k], key=lambda x: np.hypot(_center(x)[0] - _center(nb)[0],
                                                        _center(x)[1] - _center(nb)[1]))
            nb = b
            total += 1
            standing += (b[3] - b[1]) >= TH["stand_ratio"] * seated
        if total and standing / total >= 0.8 and tl.count_at(t_end - 0.3) >= 2:
            t = round(e.t + 1.0, 2)
            out.append(Candidate("stranger", t, standing / total, [t, round(t_end - 0.3, 2)],
                                 {"t_enter": e.t}))
    return out


def rule_client_id(f: Features) -> list[Candidate]:
    """Client visit: money reaches the cashier before any ID document is seen (п. 19)."""
    if f.probe_series is None:
        return []
    out = []
    for e in f.timeline.events:
        if e.kind != "count_up" or e.before != 1:
            continue
        # a courier with a cash bag is a different procedure (rule_cash_bag)
        if f.vwin("bag_on_desk", e.t, e.t + 10) >= TH["bag"]:
            continue
        t1 = min(f.duration - 0.1, e.t + TH["client_window"])
        ts = [round(e.t + k, 2) for k in range(int(t1 - e.t) + 1)]
        if len(ts) < 6:
            continue
        s = f.probe_series(["id_card", "cash_in_hands"], ts)
        idp, cash = np.asarray(s["id_card"]), np.asarray(s["cash_in_hands"])
        hit_cash = np.where(cash >= TH["client_cash"])[0]
        if not len(hit_cash):
            continue
        k = int(hit_cash[0])
        if idp[:k + 1].max(initial=0.0) < TH["client_id"]:
            t = ts[k]
            later = np.where(idp >= TH["client_id"])[0]
            out.append(Candidate("no_client_id_check", t, float(cash[k]),
                                 [t, ts[min(k + 2, len(ts) - 1)]],
                                 {"t_enter": e.t,
                                  "t_id": ts[int(later[0])] if len(later) else None}))
    return out


# rule_client_id is kept but not active: the 2B VLM answers "ID shown" for any object in
# the cashier's hands (cash, phone), so the ID-before-cash order is not observable (E011).
ALL_RULES = ([rule_client_id] if os.environ.get("CASHGUARD_CLIENT_ID_RULE") == "1" else []) + [
    rule_stranger, rule_departures, rule_safe_open_seated, rule_periodic, rule_client_flow,
    rule_cash_bag]


def run_rules(f: Features) -> list[Candidate]:
    cands = []
    for r in ALL_RULES:
        cands.extend(r(f))
    return merge(cands)


def merge(cands: list[Candidate], gap: float = 4.0) -> list[Candidate]:
    """One event per type within `gap` seconds: earliest time, highest confidence."""
    cands = sorted(cands, key=lambda c: (c.code, c.t))
    out: list[Candidate] = []
    for c in cands:
        if out and out[-1].code == c.code and c.t - out[-1].t <= gap:
            # same event seen again: keep the earliest moment (when it becomes visible)
            out[-1].score = max(out[-1].score, c.score)
            continue
        out.append(c)
    return sorted(out, key=lambda c: c.t)
