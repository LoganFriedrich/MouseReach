"""Where a watcher cycle's time goes, written to the log once per cycle.

WHY THIS EXISTS
---------------
The watcher's loop is: scan for work, pick one item, do it, repeat. Each scan
walks folders on a shared drive, and none of them recorded how long it took. On
one processing node the log showed a video's work taking about 20 seconds and
the next item starting 8 to 10 minutes later, for days -- most of the node's
time went to scanning, and nothing in the log said which scan. A slow step that
says nothing cannot be found or fixed.

So every step of a cycle is timed, and when the cycle as a whole is slow enough
to matter, one line names each step and its time:

    Cycle 512.4 s: scan 491.0 s [stale claims 0.1, re-pose adoption 2.0,
    misfiled singles 1.2, posed discovery 9.8, review return 477.6], pick 0.4 s,
    work 21.0 s

A single step that is slow on its own is also logged as it finishes, so a step
that never finishes (a hang) is visible from the last step that did.

Nothing here changes what the watcher does; it only measures it.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A cycle (or a single step) shorter than this is not worth a log line: the
# point is to explain slow cycles, not to add a line to every fast one.
SLOW_SECONDS = 5.0


class CycleTimer:
    """Times the named steps of ONE cycle. Make a new one for each cycle.

    A step whose name contains '/' is a part of the step named before the
    slash (``"scan/review return"`` is part of ``"scan"``); parts are shown
    inside their step's brackets and are not added to the cycle total twice.
    """

    def __init__(self, slow_seconds: float = SLOW_SECONDS,
                 log: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.slow_seconds = slow_seconds
        self._log = log or logger.info
        self._clock = clock
        self.steps: List[Tuple[str, float]] = []

    @contextmanager
    def step(self, name: str):
        started = self._clock()
        try:
            yield
        finally:
            took = self._clock() - started
            self.steps.append((name, took))
            if took >= self.slow_seconds:
                self._log("Cycle step %s took %.1f s" % (name, took))

    def total(self) -> float:
        return sum(took for name, took in self.steps if "/" not in name)

    def summary(self) -> str:
        parts = {}
        for name, took in self.steps:
            if "/" in name:
                parent, child = name.split("/", 1)
                parts.setdefault(parent, []).append("%s %.1f" % (child, took))
        pieces = []
        for name, took in self.steps:
            if "/" in name:
                continue
            text = "%s %.1f s" % (name, took)
            if parts.get(name):
                text += " [%s]" % ", ".join(parts[name])
            pieces.append(text)
        return "Cycle %.1f s: %s" % (self.total(), ", ".join(pieces))

    def log_if_slow(self) -> Optional[str]:
        """Write the summary line when the cycle took long enough to matter."""
        if self.total() < self.slow_seconds:
            return None
        line = self.summary()
        self._log(line)
        return line
