"""Filesystem hardening shared by the pipeline stages and the watcher.

WHY THIS EXISTS
---------------
A reviewer GUI (or an indexer/backup pass) briefly holding a pipeline file
surfaces as PermissionError winerror 5/32 (errno 13) and clears within
seconds. Three work items died on exactly this in three days (2026-09-02..05):
a copy's destination, a copy's cleanup unlink, and finally a stage's own
OUTPUT WRITE (reach save_results, with the file open in a review tool). The
copy paths were hardened in watcher.transfer; this module is the shared
primitive so the STAGE WRITERS get the same absorption -- and so the
"is this lock transient?" rule lives in exactly one place.

Semantics: bounded retries, then the LAST error is re-RAISED. A write that
still fails after retries must fail loudly so the orchestrator marks the
video failed -- swallowing it would strand a stage with no output and no
alarm.
"""
from __future__ import annotations

import builtins
import json
import time
from pathlib import Path

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0

# Test seams: retries are exercised without real files or real sleeping.
_open = builtins.open


def sha256_file(path) -> str:
    """sha256 hex digest of a file's content, chunked."""
    import hashlib
    h = hashlib.sha256()
    with _open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_transient_lock(e: OSError) -> bool:
    """A lock that plausibly clears in seconds, worth retrying.

    Windows sharing violations arrive as winerror 5 (access denied) or 32
    (sharing violation); through Python both usually carry errno 13."""
    return getattr(e, "winerror", None) in (5, 32) or e.errno == 13


def retry_transient(fn, attempts: int = RETRY_ATTEMPTS,
                    base_delay: float = RETRY_BASE_DELAY, what: str = ""):
    """Run ``fn()``; retry on transient locks; re-raise the last error.

    Non-transient errors raise immediately -- retrying "disk full" or
    "no such directory" only delays the real answer.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except OSError as e:
            if attempt < attempts and is_transient_lock(e):
                time.sleep(base_delay * attempt)
                continue
            raise


def dump_json_with_retry(path, data, attempts: int = RETRY_ATTEMPTS,
                         base_delay: float = RETRY_BASE_DELAY,
                         **json_kwargs) -> None:
    """``json.dump`` to ``path``, absorbing transient holds on the file.

    The whole open+dump is one attempt (a partial write from a failed dump
    must not survive as the "output"); after the retries the last error
    raises, so a genuinely stuck file still fails the stage loudly.
    """
    path = Path(path)

    def _attempt():
        with _open(path, "w") as f:
            json.dump(data, f, **json_kwargs)

    retry_transient(_attempt, attempts=attempts, base_delay=base_delay,
                    what=str(path))
