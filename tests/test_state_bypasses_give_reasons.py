"""Every bypass of the video state rules says why.

WHY: force_state is the escape hatch from the state machine, and it logs each
use at WARNING with the reason it is given -- so the log is the only record of
why a video skipped its normal transitions. Seven call sites passed none, and
the server's log carried "reason: not given" 150 times on 2026-10-22 alone,
which tells a person reading it nothing. This test reads the source: a new
call without a reason fails here, not in a log nobody can interpret.
"""
import re
from pathlib import Path

import mousereach

SRC = Path(mousereach.__file__).parent


def _calls(text):
    for m in re.finditer(r"\.force_state\(", text):
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        yield text.count("\n", 0, m.start()) + 1, text[m.start():i]


def test_every_force_state_call_passes_a_reason():
    missing = []
    for p in sorted(SRC.rglob("*.py")):
        for line, call in _calls(p.read_text(encoding="utf-8", errors="replace")):
            if "reason" not in call:
                missing.append(f"{p.relative_to(SRC)}:{line}")
    assert not missing, "force_state without a reason: " + ", ".join(missing)
