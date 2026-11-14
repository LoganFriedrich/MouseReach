"""How long a person spends settling each pellet in the review tools.

WHY THIS EXISTS
---------------
"How long does it take to settle one pellet?" used to be answered by
estimating from the gaps between save times. A save covers several pellets,
a reviewer pauses, walks away, opens another window -- so the gaps measure
the session, not the pellet. The review window now writes, on each pellet's
own record, when the pellet was put on screen, when its answer was taken, and
how much of that time the person was actually at the window.

WHAT IS WRITTEN (on a segment's record in ``{stem}_causal_review.json``)
------------------------------------------------------------------------
::

    timing: {
        shown_at        first time the pellet was put on screen (ISO, local
                        time with its UTC offset)
        answered_at     when the latest answer was taken, or null if nobody
                        has answered it yet (a visit that left it open)
        active_seconds  total active time over every visit (see below)
        tool            the tool of the latest visit: "triage_review",
                        "resegmentation_review" or "causal_review"
        visits: [ {shown_at, answered_at, active_seconds, tool}, ... ]
    }

One visit = the pellet is put on screen, and later its record is stored
(Next, Prev, Save Segment + Next, or a whole-file save). Coming back to a
pellet adds a visit; it never overwrites one, so the full history of looking
at a pellet survives (the summary fields are recomputed from ``visits``).

ACTIVE TIME
-----------
A second counts as active only when the review window is the active window
AND the person moved the mouse, clicked, scrolled or pressed a key within the
last ``IDLE_LIMIT_SECONDS``. The window samples this about once a second. So a
reviewer who walks away, or switches to email, stops the clock after at most
a minute; watching a pellet play with hands still for longer than a minute
also stops it (the limit is the simple rule chosen, not a measurement of
attention).

No names are recorded here: who answered a segment is the record's own
``reviewer`` field, exactly as before.

Old records have no ``timing`` key. Nothing requires one; every reader of
review records reads keys with ``.get`` and ignores keys it does not know.

This module has no Qt in it, so it can be tested without a display.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# A gap in input longer than this stops the clock. One minute: long enough to
# watch a pellet play through and think, short enough that a reviewer who
# walks away costs at most a minute of over-count.
IDLE_LIMIT_SECONDS = 60.0

# How often the window samples "active and recently used". One second is
# fine-grained enough for a quantity measured in tens of seconds per pellet,
# and cheap.
SAMPLE_INTERVAL_MS = 1000

TIMING_KEY = "timing"
VISITS_KEY = "visits"


def now_iso() -> str:
    """Local time with its UTC offset, e.g. 2026-11-14T10:03:12-06:00.

    WHY the offset: the review PCs follow daylight saving, and a bare local
    time cannot be ordered across the change or compared between machines."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ActiveTimeAccumulator:
    """Counts the seconds a person was actively at the window.

    The window calls ``note_input`` on every mouse or key event and ``tick``
    once per sample with whether its window is the active one. A tick adds
    the time since the previous tick when the window is active and the last
    input is no older than ``idle_limit``. The added time is capped at
    ``idle_limit``, so a stalled timer (the computer slept, a long load)
    never adds more than that in one go.

    ``clock`` is a monotonic seconds function, injectable for tests; wall
    clocks jump (time sync, daylight saving) and would corrupt a duration.
    """

    def __init__(self, idle_limit: float = IDLE_LIMIT_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self.idle_limit = float(idle_limit)
        self._clock = clock
        self.seconds = 0.0
        self._last_tick: Optional[float] = None
        self._last_input: Optional[float] = None

    def reset(self) -> None:
        """Start counting from zero. Putting a pellet on screen follows a
        click, so it counts as input -- the clock runs from the first tick."""
        now = self._clock()
        self.seconds = 0.0
        self._last_tick = now
        self._last_input = now

    def note_input(self) -> None:
        self._last_input = self._clock()

    def tick(self, window_active: bool) -> None:
        now = self._clock()
        last, self._last_tick = self._last_tick, now
        if last is None or not window_active or self._last_input is None:
            return
        if now - self._last_input > self.idle_limit:
            return
        self.seconds += max(0.0, min(now - last, self.idle_limit))


def make_visit(shown_at: str, answered_at: Optional[str],
               active_seconds: float, tool: str) -> Dict[str, Any]:
    return {
        "shown_at": shown_at,
        "answered_at": answered_at,
        "active_seconds": round(float(active_seconds), 1),
        "tool": tool,
    }


def _visits_of(timing: Any) -> List[Dict[str, Any]]:
    """The visit list of an existing ``timing`` value; empty for anything
    that is not one (absent on old records, or hand-edited)."""
    if not isinstance(timing, dict):
        return []
    visits = timing.get(VISITS_KEY)
    if isinstance(visits, list):
        return [v for v in visits if isinstance(v, dict)]
    # A timing block without a visit list counts as one visit.
    if timing.get("shown_at"):
        return [make_visit(timing.get("shown_at"), timing.get("answered_at"),
                           timing.get("active_seconds") or 0.0,
                           timing.get("tool") or "")]
    return []


def summarize_visits(visits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``timing`` block for a list of visits: first shown, latest
    answer, total active time, latest tool, and the visits themselves."""
    answered = [v.get("answered_at") for v in visits if v.get("answered_at")]
    total = 0.0
    for v in visits:
        try:
            total += float(v.get("active_seconds") or 0.0)
        except (TypeError, ValueError):
            pass
    return {
        "shown_at": visits[0].get("shown_at") if visits else None,
        "answered_at": answered[-1] if answered else None,
        "active_seconds": round(total, 1),
        "tool": visits[-1].get("tool") if visits else None,
        VISITS_KEY: list(visits),
    }


def add_visit(record: Dict[str, Any], previous_timing: Any,
              visit: Dict[str, Any]) -> Dict[str, Any]:
    """``record`` with ``visit`` appended to the visits it had before.

    ``previous_timing`` is the ``timing`` of the record this one replaces
    (None for a pellet never timed). Returns a new dict; the caller's record
    is not changed."""
    out = dict(record)
    out[TIMING_KEY] = summarize_visits(_visits_of(previous_timing) + [dict(visit)])
    return out


def carry_timing(record: Dict[str, Any], previous_timing: Any) -> Dict[str, Any]:
    """``record`` keeping the timing it had, when it is stored again without
    a new visit (for example a save right after Next already stored it)."""
    visits = _visits_of(previous_timing)
    if not visits:
        return record
    out = dict(record)
    out[TIMING_KEY] = summarize_visits(visits)
    return out
