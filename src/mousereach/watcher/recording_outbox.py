"""Send recordings a person has named from a recording PC to the shared intake.

WHY THIS EXISTS
---------------
A recording is renamed by a person on the recording PC (date, the animals in
each position, the tray). That rename is the one check where a human connects
a video to what is in it, so nothing automated may give a recording its name,
and nothing may take an UNNAMED recording off its PC. Once a person has named
it, though, the file still had to be carried to the shared drive by hand, and a
named recording that nobody carried down sat on the recording PC, unprocessed,
until somebody noticed it was missing. This module carries it down: the watcher
on a recording PC sends every file in its recording folder whose name passes
the collage name check, and only those.

WHAT MOVES
----------
Only a file at the TOP of ``watcher.recording_outbox`` (subfolders are never
looked into), with a video extension, whose name passes
``validator.validate_collage_filename`` -- the same check the intake applies.
A file with the recording program's own name ("2024-01-02 09-30-00.mkv") is
never touched and is reported as waiting for a name. A file with any other name
that fails the check stays where it is and is reported, with the reason, in
red: a person has to correct the name.

WHAT NEVER MOVES
----------------
* A file that may still be written: its size and modified time must have been
  unchanged for ``recording_outbox_stable_minutes`` (default 5) as seen by this
  watcher, no other program may hold it open (on Windows an exclusive-open
  test), and while a program from ``watcher.pause_while_running`` is running,
  the newest video in the folder is taken to be its current file and left
  alone.
* A file the recording check cannot be run on (no ffprobe or ffmpeg on this
  PC): held and reported in red until the tool is installed.
* Anything at all during recording time (``watcher.recording_days`` and
  ``watcher.recording_hours``). With neither set, recording time cannot be told
  apart from any other time, so NOTHING is ever sent and the report says why.
  A copy running when recording time begins is stopped between two blocks and
  its unfinished copy removed.

A FAILED RECORDING IS SENT, AND LABELED AT THE INTAKE
-----------------------------------------------------
A finished, named file that fails the recording check (watcher/recording_issue.py:
empty, a fraction of a second, unreadable, implausibly small, or its start or
end will not decode -- the SAME check, with the SAME settings
``recording_min_minutes``, ``recording_min_frames`` and
``recording_min_kb_per_second``, that the intake applies) is still sent. WHY:
a recording that only documents a mistake is still a record of what happened
that day; left on the recording PC it would be red here for ever and nobody
downstream would know the session was recorded and failed. On the shared drive
the intake sets it aside as a recording issue (Processing/Review/_Problematic,
with a reason file), never cuts it, and the database learns of it. The report
lists it under ``recording_issues`` (not red: nothing is wrong with the
sending) and the send log records the reason.

HOW A FILE IS SENT (the order is what makes the delete safe)
-----------------------------------------------------------
1. Look for the name on the shared drive first, WITHOUT walking it: the
   intake and the recording-issue folder by name, and every path the
   processing server's collage index lists for the name
   (``Processing/.collage_index.json``, watcher/collage_index.py: the
   intake, the unsupported-tray, quarantine and recording-issue folders and
   every ``Multi-Animal`` folder under Analyzed and Archive, rebuilt every
   10 minutes). WHY everywhere: a collage leaves the intake once its videos
   are filed, so "not in the intake" does not mean "never sent"; sending it
   again would cut it again. WHY an index and not a walk: a PC must never
   list folder trees on the share (see the section "FINDING A NAME" below).
   Found with the same content (size and SHA-256, both files read in full,
   the copy on the share opened by its name): it was sent before, so only
   the local copy is deleted. Found with different content: a CONFLICT --
   held for a person, never overwritten, never deleted. The index is only a
   pointer: nothing is deleted on its word alone. Two cases mean "sent
   before, moved since" and the file then WAITS, neither sent nor deleted:
   the index lists a path that is no longer there, or the shared claim table
   (one row, asked only when nothing was found) already has the name. It is
   held in red once an index built after it was first noticed still does
   not show it. With the index missing or unreadable, a name found nowhere
   and unknown to the claim table is sent as usual -- copied and verified --
   and the report says the index is missing.
2. Copy to ``<intake>/<name>.partial`` in blocks, hashing the source as it is
   read, at low disk priority. Every picker ignores ``.partial`` files.
3. Check the source did not change during the copy, then read the
   ``.partial`` back FROM THE SHARED DRIVE and hash it. Size and hash must
   match the source.
4. Rename the ``.partial`` to the final name. The rename never replaces a
   file: if the name appeared meanwhile, step 1's rules decide.
5. Check the final name is there with the right size, check once more that
   the source is unchanged, and only then delete the local original.

If verification fails, the source is kept, the bad ``.partial`` is removed and
the file is tried again on the next pass; after
``recording_outbox_warn_after_failures`` (default 3) failures in a row it is
reported in red as well.

CRASH SAFETY
------------
The source is never deleted before step 5. A crash during the copy leaves a
``.partial``; the next pass finds it, and once its size has not changed
between two passes (nothing is still writing it) removes it and copies again.
A crash after the rename but before the delete leaves an identical file on the
shared drive; the next pass finds it in step 1 and deletes the local copy then.

REPORTING
---------
Every pass returns a report that the watcher puts in its status file
(``Processing/.node_status/<host>.json``, key ``recording_outbox``):
what was sent, what is held and why (red in ``mousereach-watch-status`` and in
the Pipeline Dashboard's attention panel), what is waiting, and how many named
recordings are still on this PC. Every send, skip and delete is also appended
to ``recording_sends.jsonl`` in the watcher's log folder.

All text written to a terminal or a log is ASCII.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from mousereach.watcher.recording_guard import (
    _ascii, format_days, format_hours, parse_days, parse_hours, within_hours)
from mousereach.watcher.validator import VIDEO_EXTENSIONS, validate_collage_filename
from mousereach.watcher import collage_index, recording_issue

logger = logging.getLogger(__name__)

PARTIAL_SUFFIX = ".partial"
STATUS_KEY = "recording_outbox"
TRANSFER_LOG_NAME = "recording_sends.jsonl"
STATE_FILE_NAME = "recording_outbox_state.json"

# The recording program's own name for a file: "YYYY-MM-DD hh-mm-ss". Such a
# file has not been named by a person and is never touched.
RAW_RECORDER_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}[ _]\d{2}-\d{2}-\d{2}")

_CHUNK = 8 * 1024 * 1024
_SENT_KEPT = 20
_LISTED = 20          # names listed per kind in the status report (counts stay exact)
_LISTED_HELD = 100    # held files are what a person must act on: list more of them


# =============================================================================
# SETTINGS
# =============================================================================

@dataclass
class OutboxSettings:
    """What the sender needs from the watcher section of the config file."""
    outbox: Optional[Path] = None
    stable_minutes: float = 5.0
    # What counts as a real recording: the intake's own check settings, so
    # the two never disagree (watcher/recording_issue.py).
    check: recording_issue.CheckSettings = field(default_factory=recording_issue.CheckSettings)
    warn_after_failures: int = 3
    pass_minutes: float = 30.0
    recording_hours: Optional[str] = None
    recording_days: Optional[str] = None
    recorder_names: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg) -> "OutboxSettings":
        """From a WatcherConfig (or anything with the same attributes)."""
        raw = getattr(cfg, "recording_outbox", None)
        return cls(
            outbox=Path(raw) if raw else None,
            stable_minutes=float(getattr(cfg, "recording_outbox_stable_minutes", 5)),
            check=recording_issue.CheckSettings.from_config(cfg),
            warn_after_failures=int(getattr(cfg, "recording_outbox_warn_after_failures", 3)),
            pass_minutes=float(getattr(cfg, "recording_outbox_pass_minutes", 30)),
            recording_hours=getattr(cfg, "recording_hours", None),
            recording_days=getattr(cfg, "recording_days", None),
            recorder_names=list(getattr(cfg, "pause_while_running", None) or []),
        )


def send_window_reason(settings: OutboxSettings, now: datetime) -> Optional[str]:
    """None when sending is allowed now; otherwise why not (ASCII).

    Sending is allowed only when recording time is configured AND now is
    outside it. WHY unset means never: with no hours and no days the watcher
    cannot tell a recording day from any other, and a copy competing with a
    recording for the disk is exactly what must never happen. An unreadable
    value counts as unset, for the same reason.
    """
    hours = parse_hours(settings.recording_hours)
    days = parse_days(settings.recording_days)
    if hours is None and days is None:
        return ("not sending: watcher.recording_hours and watcher.recording_days are "
                "not set (or cannot be read), so recording time cannot be told apart")
    if days is not None and now.weekday() not in days:
        return None
    if hours is not None and not within_hours(hours, now):
        return None
    return _ascii("not sending during recording time (%s, %s)" % (
        format_days(days) if days is not None else "every day",
        format_hours(hours) if hours is not None else "all day"))


# =============================================================================
# FILE CHECKS
# =============================================================================

def is_open_for_writing(path: Path) -> bool:
    """True when another program holds ``path`` open (so it may be writing it).

    On Windows the file is opened with NO sharing allowed; that open fails with
    a sharing violation while any other handle is open, which is how a file the
    recording program is still writing is recognized. (Python's own open()
    shares, so it cannot tell.) Any other failure also counts as "open": a file
    this watcher cannot open exclusively is not one it may move. Elsewhere this
    returns False and the size-unchanged rule alone protects the file.
    """
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    generic_read, open_existing, attr_normal = 0x80000000, 3, 0x80
    handle = create(str(path), generic_read, 0, None, open_existing, attr_normal, None)
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid or handle is None:
        return True
    kernel32.CloseHandle(handle)
    return False


_THREAD_MODE_BACKGROUND_BEGIN = 0x00010000
_THREAD_MODE_BACKGROUND_END = 0x00020000


def _set_thread_mode(mode: int) -> bool:
    """SetThreadPriority on the calling thread; True when Windows accepted it."""
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentThread.restype = wintypes.HANDLE
    k.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
    k.SetThreadPriority.restype = wintypes.BOOL
    return bool(k.SetThreadPriority(k.GetCurrentThread(), mode))


@contextlib.contextmanager
def low_io_priority():
    """Run the body with this thread in background (low disk and CPU) mode.

    Yields True when the mode was taken. WHY: the copy must never compete with
    anything else the PC is doing. On Windows this is
    THREAD_MODE_BACKGROUND_BEGIN; elsewhere nothing changes (yields False).
    """
    began = False
    if os.name == "nt":
        try:
            began = _set_thread_mode(_THREAD_MODE_BACKGROUND_BEGIN)
        except Exception:
            began = False
        if not began:
            logger.debug("could not lower this thread's disk priority; copying at normal priority")
    try:
        yield began
    finally:
        if began:
            try:
                _set_thread_mode(_THREAD_MODE_BACKGROUND_END)
            except Exception:
                pass


def sha256_file(path: Path, abort: Optional[Callable[[], Optional[str]]] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if abort is not None:
                why = abort()
                if why:
                    raise _Aborted(why)
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class _Aborted(Exception):
    """A copy or hash stopped because sending is no longer allowed."""


# =============================================================================
# FINDING A NAME ON THE SHARED DRIVE (without walking it)
# =============================================================================
# A recording PC never lists folder trees on the shared drive. It reads ONE
# file the processing server writes (watcher/collage_index.py: where every
# collage rests, rebuilt every 10 minutes) and opens single named files:
#
#   * <intake>/<name>                 -- where this PC puts it; covers everything
#                                        sent since the index was built
#   * <recording-issue folder>/<name> -- a failed recording is set aside there,
#                                        flat, by name (recording_issue.set_aside)
#   * every path the index lists for the name
#
# WHY: one recursive walk of the share from a PC, repeated every pass, turns a
# degraded network connection into hours of crawling, and the shared
# coordination database on the same share waits behind it for every machine.
# A listed path is only a pointer: nothing is deleted until that one file has
# been opened and its size and SHA-256 match (_match_existing).


def _named_file_exists(p: Path) -> bool:
    """True when ``p`` is a file. One stat of one named path; never a listing."""
    try:
        return os.stat(str(p)).st_mode & 0o170000 == 0o100000
    except (OSError, ValueError):
        return False


# =============================================================================
# STATE BETWEEN PASSES (local file)
# =============================================================================

def _load_json(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_json(path: Optional[Path], data: dict) -> None:
    if path is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".outbox.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="ascii", errors="replace") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, str(path))
    except Exception as e:
        logger.debug("could not save the recording-send state: %s", e)


def _rename_no_overwrite(src: Path, dst: Path) -> None:
    """Rename ``src`` to ``dst``; raise FileExistsError rather than replace a file.

    Windows' rename already refuses an existing name. Elsewhere rename would
    replace it, so a hard link (which also refuses) is made and the old name
    removed.
    """
    if os.name == "nt":
        os.rename(str(src), str(dst))
    else:
        os.link(str(src), str(dst))
        os.unlink(str(src))


# =============================================================================
# THE SENDER
# =============================================================================

class RecordingSender:
    """One sender per watcher. ``run_pass()`` does one pass and returns the report.

    Everything that touches the machine (clock, probe, open test, the check for
    a running recording program, the stop request) is injectable, so the tests
    run against temporary folders without a recording program or a real share.
    """

    def __init__(self, settings: OutboxSettings, nas_root: Optional[Path],
                 state_path: Optional[Path] = None,
                 transfer_log: Optional[Path] = None,
                 clock: Callable[[], float] = time.time,
                 local_now: Callable[[], datetime] = datetime.now,
                 assess: Optional[Callable[[Path], "recording_issue.Verdict"]] = None,
                 open_check: Callable[[Path], bool] = is_open_for_writing,
                 recorder_running: Optional[Callable[[], bool]] = None,
                 should_stop: Callable[[], bool] = lambda: False,
                 chunk_size: int = _CHUNK,
                 claim_lookup: Optional[Callable[[str], Optional[dict]]] = None,
                 read_index: Callable[[Path], "collage_index.Index"] = collage_index.read):
        self.settings = settings
        self.nas_root = Path(nas_root) if nas_root else None
        self.state_path = state_path
        self.transfer_log = transfer_log
        self._clock = clock
        self._local_now = local_now
        self._assess = assess or (lambda p: recording_issue.assess(p, settings.check))
        self._open_check = open_check
        self._recorder_running = recorder_running or self._default_recorder_running
        self._should_stop = should_stop
        self._chunk = max(1, int(chunk_size))
        # One row of the shared coordination table for one name (or None):
        # asked only when a name cannot be found by name anywhere, so a
        # recording the pipeline has already taken is never sent twice just
        # because the index has not caught up (see _send_one).
        self._claim_lookup = claim_lookup
        self._read_index = read_index
        self._pass_index: Optional["collage_index.Index"] = None
        self.state = _load_json(state_path)
        for key in ("seen", "failures", "partials", "conflicts", "probes", "unlocated"):
            if not isinstance(self.state.get(key), dict):
                self.state[key] = {}
        if not isinstance(self.state.get("sent"), list):
            self.state["sent"] = []

    # ------------------------------------------------------------------ helpers
    def _default_recorder_running(self) -> bool:
        names = self.settings.recorder_names
        if not names:
            return False
        try:
            from mousereach.watcher.recording_guard import running_programs
            return bool(running_programs(names))
        except Exception:
            return True      # could not look: assume it is running (safe side)

    @property
    def intake(self) -> Optional[Path]:
        return self.nas_root / "Unanalyzed" / "Multi-Animal" if self.nas_root else None

    def _abort_reason(self) -> Optional[str]:
        if self._should_stop():
            return "the watcher was asked to stop"
        return send_window_reason(self.settings, self._local_now())

    def _log(self, event: str, name: str, **extra) -> None:
        rec = {"at": datetime.now().isoformat(timespec="seconds"), "event": event, "name": name}
        rec.update(extra)
        logger.info(_ascii("Recording send: %s %s %s" % (
            event, name, " ".join("%s=%s" % kv for kv in sorted(extra.items())))))
        if self.transfer_log is None:
            return
        try:
            Path(self.transfer_log).parent.mkdir(parents=True, exist_ok=True)
            with open(self.transfer_log, "a", encoding="ascii", errors="replace") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except OSError as e:
            logger.debug("could not append to the transfer log: %s", e)

    def _candidates(self) -> List[Path]:
        out = self.settings.outbox
        files = []
        try:
            with os.scandir(out) as it:
                for e in it:
                    if not e.is_file(follow_symlinks=False):
                        continue
                    n = e.name
                    if n.startswith((".", "~")) or n.lower().endswith(
                            (".part", ".tmp", PARTIAL_SUFFIX)):
                        continue
                    if Path(n).suffix.lower() not in VIDEO_EXTENSIONS:
                        continue
                    files.append(Path(e.path))
        except OSError:
            return []

        # Newest last, by the later of creation and last change. WHY both: while
        # a program is still writing a file, Windows may not move its modified
        # time forward, but its creation time is already the newest.
        def newest(p):
            try:
                st = p.stat()
                return max(st.st_mtime, st.st_ctime)
            except OSError:
                return 0.0
        return sorted(files, key=newest)

    def _stable(self, path: Path, st) -> Optional[str]:
        """None when the size and time have not changed for long enough."""
        key = path.name.lower()
        rec = self.state["seen"].get(key)
        now = self._clock()
        sig = [st.st_size, st.st_mtime]
        if not rec or rec.get("sig") != sig:
            self.state["seen"][key] = {"sig": sig, "since": now}
            return "may still be written (its size or time changed since the last look)"
        waited = now - float(rec.get("since", now))
        need = self.settings.stable_minutes * 60.0
        if waited < need:
            return "may still be written (unchanged for %d of %d s)" % (int(waited), int(need))
        return None

    def _held_conflict(self, path: Path, st) -> Optional[str]:
        rec = self.state["conflicts"].get(path.name.lower())
        if not rec:
            return None
        if rec.get("sig") != [st.st_size, st.st_mtime]:
            self.state["conflicts"].pop(path.name.lower(), None)
            return None
        other = Path(rec.get("dest", ""))
        try:
            ost = other.stat()
            if [ost.st_size, ost.st_mtime] != rec.get("dest_sig"):
                self.state["conflicts"].pop(path.name.lower(), None)
                return None
        except OSError:
            self.state["conflicts"].pop(path.name.lower(), None)
            return None
        return rec.get("reason")

    def _known_hold(self, path: Path, st) -> Optional[str]:
        """Why this file is held, from what is already known, or None.

        Used while sending is not allowed: reads no file, runs no probe."""
        held = self._held_conflict(path, st)
        if held:
            return held
        rec = self.state["probes"].get(path.name.lower())
        if (rec and rec.get("sig") == [st.st_size, st.st_mtime]
                and rec.get("outcome") == "cannot_check"):
            return self._check_hold(rec)
        fail = self.state["failures"].get(path.name.lower()) or {}
        if int(fail.get("count", 0)) >= self.settings.warn_after_failures:
            return _ascii("%s (failed %d times in a row)" % (
                fail.get("last_error", "copy failed"), int(fail["count"])))
        return None

    def _verdict(self, path: Path, st) -> dict:
        """The recording check's verdict on this file ({outcome, reason}),
        remembered while its size and time are unchanged so a file is read
        once, not every pass."""
        key = path.name.lower()
        sig = [st.st_size, st.st_mtime]
        rec = self.state["probes"].get(key)
        if not (rec and rec.get("sig") == sig and rec.get("outcome")):
            v = self._assess(path)
            rec = {"sig": sig, "outcome": v.outcome, "reason": v.reason,
                   "measurements": v.measurements}
            self.state["probes"][key] = rec
        return rec

    @staticmethod
    def _check_hold(rec: dict) -> str:
        return _ascii("the recording check could not run here (%s); install ffmpeg "
                      "(or ffprobe) on this PC" % (rec.get("reason") or "unknown reason"))

    def _fail(self, name: str, why: str) -> None:
        rec = self.state["failures"].setdefault(name.lower(), {"count": 0})
        rec["count"] = int(rec.get("count", 0)) + 1
        rec["last_error"] = _ascii(why)
        self._log("copy_failed", name, reason=_ascii(why), attempt=rec["count"])

    def _delete_local(self, path: Path, sig) -> Optional[str]:
        """Delete the local original if it is still exactly what was verified."""
        try:
            st = path.stat()
        except OSError as e:
            return "could not look at the local file before deleting it: %s" % e
        if [st.st_size, st.st_mtime] != sig:
            return "the local file changed after it was checked; kept"
        try:
            path.unlink()
        except OSError as e:
            return ("verified on the shared drive, but the local file could not be "
                    "deleted (%s); tried again next pass" % e)
        return None

    def _record_sent(self, name: str, size: int, how: str) -> None:
        key = name.lower()
        for k in ("seen", "failures", "partials", "conflicts", "probes", "unlocated"):
            self.state[k].pop(key, None)
        self.state["sent"] = ([{"name": name, "bytes": size, "how": how,
                                "at": datetime.now().isoformat(timespec="seconds")}]
                              + self.state["sent"])[:_SENT_KEPT]

    def _conflict(self, path: Path, st, other: Path, why: str) -> str:
        try:
            ost = other.stat()
            dest_sig = [ost.st_size, ost.st_mtime]
        except OSError:
            dest_sig = None
        reason = _ascii("a different file of the same name is already on the shared drive "
                        "(%s; %s). Neither was changed; a person must decide which is right"
                        % (other, why))
        self.state["conflicts"][path.name.lower()] = {
            "sig": [st.st_size, st.st_mtime], "dest": str(other),
            "dest_sig": dest_sig, "reason": reason}
        self._log("conflict_held", path.name, dest=str(other), why=why)
        return reason

    # ------------------------------------------------- where is this name now?
    def _index(self) -> "collage_index.Index":
        """The server's collage index, read once per pass (one file)."""
        if self._pass_index is None:
            try:
                self._pass_index = self._read_index(self.nas_root)
            except Exception as e:      # an injected reader; the real one never raises
                self._pass_index = collage_index.Index(None, None, None, "unreadable", str(e))
        return self._pass_index

    def _whereabouts(self, name: str) -> Tuple[List[Path], List[Path]]:
        """(found, listed_but_gone) for ``name`` on the shared drive.

        ``found``: the named files that exist, from the two fixed places and
        every path the index lists. ``listed_but_gone``: paths the index lists
        that are not there now (moved after the index was built). Only single
        named paths are looked at -- see the section comment above."""
        fixed = [self.intake / name]
        problematic = recording_issue.problematic_dir(self.nas_root)
        if problematic is not None:
            fixed.append(problematic / name)
        listed = self._index().paths(self.nas_root, name)
        found: List[Path] = []
        gone: List[Path] = []
        seen = set()
        for p in fixed + listed:
            key = os.path.normcase(os.path.normpath(str(p)))
            if key in seen:
                continue
            seen.add(key)
            if _named_file_exists(p):
                found.append(p)
            elif p in listed:
                gone.append(p)
        return found, gone

    def _pipeline_record(self, name: str) -> Optional[dict]:
        """The shared claim row for ``name`` (one row), or None when there is
        none or it cannot be read. A lookup that fails does not stop a send:
        the cutting step refuses a collage another machine has already cut."""
        if self._claim_lookup is None:
            return None
        try:
            row = self._claim_lookup(name)
        except Exception as e:
            logger.debug("could not ask the coordination table about %s: %s", name, e)
            return None
        return row if isinstance(row, dict) and row else None

    def _unlocated(self, name: str, why: str) -> Tuple[str, str]:
        """Sent before, but not found by name now: never sent again, never
        deleted here. Waits for the server's next index; held (red) once an
        index built AFTER this was first noticed still does not show it, so a
        person looks instead of the file waiting for ever."""
        key = name.lower()
        first = float(self.state["unlocated"].setdefault(key, self._clock()))
        idx = self._index()
        if idx.usable and idx.built_at_epoch is not None and idx.built_at_epoch > first:
            return "held", _ascii(
                "%s; the server's collage index built since (%s) still does not show it on "
                "the shared drive, so a person must find where it went. Nothing was sent or "
                "deleted" % (why, idx.built_at))
        return "waiting", _ascii(
            "%s; nothing is sent or deleted until the server's next collage index (rebuilt "
            "every %d minutes) shows where it is" % (why, int(collage_index.REBUILD_EVERY_S // 60)))

    # ---------------------------------------------------------------- one file
    def _match_existing(self, path: Path, st, others: Iterable[Path],
                        src_hash: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
        """Compare the local file with every same-named file on the shared drive.

        Returns (outcome, detail, src_hash) with outcome 'same' (an identical
        copy exists), 'conflict' (a different one exists) or 'none'.
        """
        for other in others:
            try:
                ost = other.stat()
            except OSError:
                continue
            if ost.st_size != st.st_size:
                return "conflict", self._conflict(
                    path, st, other, "%d bytes there, %d here" % (ost.st_size, st.st_size)), src_hash
            if src_hash is None:
                src_hash = sha256_file(path, self._abort_reason)
            if sha256_file(other, self._abort_reason) != src_hash:
                return "conflict", self._conflict(
                    path, st, other, "same size, different content"), src_hash
            return "same", str(other), src_hash
        return "none", None, src_hash

    def _send_one(self, path: Path, st) -> Tuple[str, str]:
        """Send one checked file. Returns (outcome, text):
        'sent' | 'already' | 'held' | 'waiting' | 'failed'."""
        name = path.name
        sig = [st.st_size, st.st_mtime]
        intake = self.intake
        partial = intake / (name + PARTIAL_SUFFIX)
        final = intake / name
        if not intake.is_dir():
            return "failed", "the intake folder %s does not exist or cannot be reached" % intake
        created = False      # True once THIS call made the .partial; only then is it ours to remove
        renamed = False
        try:
            found, gone = self._whereabouts(name)
            outcome, detail, src_hash = self._match_existing(path, st, found, None)
            if outcome == "conflict":
                return "held", detail
            if outcome == "same":
                return self._already_there(path, st, sig, detail)
            # Not found by name. Two things still mean "sent before, moved
            # since", and then the file is neither sent again (that would cut
            # it twice) nor deleted (nothing confirmed a copy):
            if gone:
                return self._unlocated(name, (
                    "the server's collage index lists it at %s, but it is not there now "
                    "(moved since the index was built)" % gone[0]))
            record = self._pipeline_record(name)
            if record:
                return self._unlocated(name, (
                    "the pipeline already has this recording (claimed by %s, %s), but it "
                    "is not in the intake and the collage index does not list it"
                    % (record.get("hostname"), record.get("state"))))
            self.state["unlocated"].pop(name.lower(), None)

            # An unfinished copy left by an earlier pass that stopped. Removed
            # only once its size has not changed between two passes, so a copy
            # another machine is writing right now is never pulled away.
            if partial.exists():
                try:
                    psize = partial.stat().st_size
                except OSError:
                    psize = None
                if psize is not None and self.state["partials"].get(name.lower()) == psize:
                    partial.unlink()
                    self.state["partials"].pop(name.lower(), None)
                    self._log("removed_stale_partial", name, bytes=psize)
                else:
                    self.state["partials"][name.lower()] = psize
                    return "waiting", ("an unfinished copy is on the shared drive; it is removed "
                                       "on the next pass if nothing is still writing it")

            with low_io_priority():      # yields whether the low priority took
                h = hashlib.sha256()
                copied = 0
                with open(path, "rb") as src:
                    with open(partial, "xb") as dst:
                        created = True
                        while True:
                            why = self._abort_reason()
                            if why:
                                raise _Aborted(why)
                            block = src.read(self._chunk)
                            if not block:
                                break
                            h.update(block)
                            dst.write(block)
                            copied += len(block)
                        dst.flush()
                        os.fsync(dst.fileno())
                src_hash = h.hexdigest()
                now_st = path.stat()
                if [now_st.st_size, now_st.st_mtime] != sig or copied != st.st_size:
                    return "waiting", "the file changed while it was being copied; tried again later"
                # Verify by reading the copy back from the shared drive.
                psize = partial.stat().st_size
                if psize != st.st_size:
                    self._fail(name, "copy is %d bytes, the original %d" % (psize, st.st_size))
                    return "failed", "the copy on the shared drive has the wrong size"
                if sha256_file(partial, self._abort_reason) != src_hash:
                    self._fail(name, "SHA-256 of the copy read back from the shared drive differs")
                    return "failed", "the copy read back from the shared drive differs from the original"
                try:
                    _rename_no_overwrite(partial, final)
                    renamed = True
                except FileExistsError:
                    outcome, detail, _ = self._match_existing(path, st, [final], src_hash)
                    if outcome == "same":
                        return self._already_there(path, st, sig, str(final))
                    return "held", detail or "a file of this name appeared on the shared drive"
            fst = final.stat()
            if fst.st_size != st.st_size:
                # Not expected after a verified rename; the original is never deleted then.
                self._fail(name, "final file is %d bytes, the original %d" % (fst.st_size, st.st_size))
                return "failed", "the file on the shared drive has the wrong size after renaming"
            problem = self._delete_local(path, sig)
            self._log("sent", name, bytes=st.st_size, sha256=src_hash, to=str(final),
                      local_deleted=problem is None)
            self._record_sent(name, st.st_size, "copied and verified")
            if problem:
                return "waiting", problem
            return "sent", str(final)
        except _Aborted as e:
            self._log("copy_stopped", name, reason=str(e))
            return "waiting", "copy stopped (%s); it starts again next time" % e
        except Exception as e:
            self._fail(name, "%s: %s" % (type(e).__name__, e))
            return "failed", _ascii("%s: %s" % (type(e).__name__, e))
        finally:
            if created and not renamed:
                try:
                    partial.unlink()
                except OSError:
                    pass

    def _already_there(self, path: Path, st, sig, where: str) -> Tuple[str, str]:
        """An identical file is on the shared drive: delete the local copy only."""
        problem = self._delete_local(path, sig)
        if problem:
            return "waiting", problem
        self._log("deleted_local_already_on_share", path.name, on_share=where, bytes=st.st_size)
        self._record_sent(path.name, st.st_size, "was already on the shared drive")
        return "already", where

    def _preview_text(self, name: str, st) -> str:
        """What a real pass would do with this file, from the same lookups
        (named files and the index; sizes only -- a preview reads no content)."""
        if self.nas_root is None:
            return "ready: would be copied, verified, then deleted here"
        found, gone = self._whereabouts(name)
        same = []
        for other in found:
            try:
                same.append((other, os.stat(str(other)).st_size == st.st_size))
            except OSError:
                continue
        if any(not ok for _o, ok in same):
            return "a file of this name but another size is on the shared drive: would be HELD"
        if same:
            return ("already on the shared drive with the same size (%s): the local copy "
                    "would be deleted once a full content check agrees" % same[0][0])
        if gone:
            return ("the server's collage index lists it at %s, but it is not there now: would "
                    "WAIT (neither sent nor deleted) until the next index shows where it is"
                    % gone[0])
        record = self._pipeline_record(name)
        if record:
            return ("the pipeline already has this recording (claimed by %s, %s) but it cannot "
                    "be found by name: would WAIT (neither sent nor deleted)"
                    % (record.get("hostname"), record.get("state")))
        return "ready: would be copied, verified, then deleted here"

    # ---------------------------------------------------------------- one pass
    def run_pass(self, preview: bool = False) -> dict:
        """One look at the outbox; sends what may be sent unless ``preview``."""
        s = self.settings
        report = {"at": datetime.now().isoformat(timespec="seconds"),
                  "outbox": str(s.outbox) if s.outbox else None,
                  "not_sending": None, "sent_this_pass": [], "held": [],
                  "waiting": [], "waiting_for_name": [], "on_this_pc": 0,
                  "sent_recently": [], "recording_issues": []}
        if s.outbox is None:
            report["not_sending"] = "watcher.recording_outbox is not set (sending is off)"
            return report
        if not Path(s.outbox).is_dir():
            report["held"].append({"name": str(s.outbox),
                                   "reason": "the recording folder does not exist or cannot be read"})
            return report
        if self.nas_root is None:
            report["not_sending"] = "nas_root is not set, so there is nowhere to send to"
        gate = report["not_sending"] or self._abort_reason()
        report["not_sending"] = gate

        files = self._candidates()
        recorder_on = bool(files) and (gate is None or preview) and self._recorder_running()
        newest = files[-1] if files else None
        deadline = self._clock() + s.pass_minutes * 60.0
        present = set()
        # The server's collage index is read at most once per pass, and only
        # when a file is ready to be looked up (one file; never a listing).
        self._pass_index = None

        for path in files:
            name = path.name
            present.add(name.lower())
            if RAW_RECORDER_NAME.match(name):
                report["waiting_for_name"].append(name)
                continue
            v = validate_collage_filename(name)
            if not v.valid:
                report["held"].append({"name": name, "reason": _ascii(
                    "name is not a valid session name: %s" % v.error)})
                continue
            report["on_this_pc"] += 1
            try:
                st = path.stat()
            except OSError as e:
                report["waiting"].append({"name": name, "reason": "cannot be read: %s" % e})
                continue
            if gate and not preview:
                # Nothing is sent now, but what is held stays held -- and
                # red -- from facts already known: the size, the duration
                # read on an earlier pass, a remembered conflict, repeated
                # failures. Nothing new is read from the file meanwhile.
                held = self._known_hold(path, st)
                if held:
                    report["held"].append({"name": name, "reason": held})
                continue
            # A preview has no history of earlier looks, so it cannot apply the
            # unchanged-for-N-minutes rule; it says so in its heading instead.
            why = None if preview else self._stable(path, st)
            if why is None and recorder_on and path == newest:
                why = "may be the recording program's current file (it is running)"
            if why is None and self._open_check(path):
                why = "open in another program"
            if why:
                report["waiting"].append({"name": name, "reason": why})
                continue
            held = self._held_conflict(path, st)
            if held:
                report["held"].append({"name": name, "reason": held})
                continue
            verdict = self._verdict(path, st)
            if verdict.get("outcome") == "cannot_check":
                report["held"].append({"name": name, "reason": self._check_hold(verdict)})
                continue
            issue = verdict.get("reason") if verdict.get("outcome") == "issue" else None
            if preview:
                text = self._preview_text(name, st)
                if issue:
                    text += ("; it fails the recording check (%s), so the intake will set it "
                             "aside as a recording issue" % issue)
                report["waiting"].append({"name": name, "reason": text})
                continue
            if self._clock() >= deadline or self._abort_reason():
                report["waiting"].append({"name": name, "reason": "ready; sent on a later pass"})
                continue
            outcome, text = self._send_one(path, st)
            if outcome in ("sent", "already"):
                how = "copied and verified" if outcome == "sent" else "already on the shared drive"
                if issue:
                    how += "; a recording issue, set aside at the intake: " + issue
                    report["recording_issues"].append({"name": name, "reason": issue})
                    self._log("recording_issue", name, reason=issue)
                report["sent_this_pass"].append({"name": name, "bytes": st.st_size, "how": how})
                report["on_this_pc"] -= 1
            elif outcome == "held":
                report["held"].append({"name": name, "reason": text})
            else:
                count = int(self.state["failures"].get(name.lower(), {}).get("count", 0))
                entry = {"name": name, "reason": text}
                if count >= s.warn_after_failures:
                    entry["reason"] = _ascii("%s (failed %d times in a row)" % (text, count))
                    report["held"].append(entry)
                else:
                    report["waiting"].append(entry)

        # Forget files that are no longer in the folder.
        for k in ("seen", "failures", "conflicts", "probes", "unlocated"):
            for key in [x for x in self.state[k] if x not in present]:
                self.state[k].pop(key, None)
        report["sent_recently"] = list(self.state["sent"][:10])
        if self._pass_index is not None:
            # How current the server's index was (status 'missing' or
            # 'unreadable' is shown to a person: new files are still sent,
            # but a file sent long ago is only found again through it).
            report["collage_index"] = self._pass_index.describe()
        # Keep the status file small: a recording folder can hold years of
        # unnamed files. The counts stay exact; the lists are cut. A preview
        # is read by a person at the PC, so it keeps every name.
        for key, cap in (("waiting_for_name", _LISTED), ("held", _LISTED_HELD),
                         ("waiting", _LISTED)):
            report[key + "_count"] = len(report[key])
            if not preview:
                report[key] = report[key][:cap]
        if not preview:
            _save_json(self.state_path, self.state)
        return report


def held_lines(report: Optional[dict]) -> List[str]:
    """The red lines of a report: one per held file. [] when nothing is held."""
    if not isinstance(report, dict):
        return []
    return [_ascii("%s: %s" % (h.get("name"), h.get("reason")))
            for h in report.get("held") or [] if isinstance(h, dict)]


def summary_line(report: Optional[dict]) -> str:
    """One plain line for a person: how many sent, waiting, held, still on the PC."""
    if not isinstance(report, dict):
        return ""
    if report.get("outbox") is None:
        return "sending recordings: off"
    def count(key):
        n = report.get(key + "_count")
        return int(n) if isinstance(n, int) else len(report.get(key) or [])
    parts = ["%d sent last pass" % len(report.get("sent_this_pass") or []),
             "%d named still on this PC" % int(report.get("on_this_pc") or 0),
             "%d waiting" % count("waiting"),
             "%d HELD" % count("held")]
    if count("waiting_for_name"):
        parts.append("%d waiting for a person to name them" % count("waiting_for_name"))
    line = "recordings: " + ", ".join(parts)
    if report.get("not_sending"):
        line += " (%s)" % report["not_sending"]
    return _ascii(line)


# =============================================================================
# COMMAND LINE: mousereach-send-recordings
# =============================================================================

def build_sender_from_config(should_stop: Callable[[], bool] = lambda: False) -> RecordingSender:
    """A sender with this machine's settings (~/.mousereach/config.json)."""
    from mousereach.config import Paths, WatcherConfig, require_processing_root
    cfg = WatcherConfig.load()
    settings = OutboxSettings.from_config(cfg)
    try:
        root = require_processing_root()
        state_path = Path(root) / STATE_FILE_NAME
    except Exception:
        state_path = None
    try:
        log_path = cfg.get_log_dir() / TRANSFER_LOG_NAME
    except Exception:
        log_path = None
    coordinator = []

    def claim_lookup(name):
        # Built on first use, so a pass that finds every name never opens the
        # shared database at all. One row per call.
        if not coordinator:
            from mousereach.watcher.coordination import PipelineCoordinator
            coordinator.append(PipelineCoordinator())
        return coordinator[0].get_collage_claim(name)

    return RecordingSender(settings, Paths.NAS_ROOT, state_path=state_path,
                           transfer_log=log_path, should_stop=should_stop,
                           claim_lookup=claim_lookup)


def _print_report(report: dict) -> None:
    print(summary_line(report))
    for h in report.get("held") or []:
        print("  [!] HELD %s: %s" % (h.get("name"), h.get("reason")))
    for w in report.get("waiting") or []:
        print("  waiting  %s: %s" % (w.get("name"), w.get("reason")))
    for n in report.get("waiting_for_name") or []:
        print("  not named yet (never sent): %s" % n)
    for s in report.get("sent_this_pass") or []:
        print("  sent     %s (%s, %s bytes)" % (s.get("name"), s.get("how"), s.get("bytes")))
    for r in report.get("recording_issues") or []:
        print("  recording issue (sent; set aside at the intake) %s: %s"
              % (r.get("name"), r.get("reason")))


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="mousereach-send-recordings",
        description=("Send recordings a person has named from this PC's recording folder "
                     "(watcher.recording_outbox) to the shared intake, verify each copy, "
                     "then delete the local original. The running watcher does this on its "
                     "own outside recording hours; this command shows what it would do "
                     "(--preview) or does one pass now (--once)."))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true",
                   help="list what would be sent, held or left; copy and delete nothing")
    g.add_argument("--once", action="store_true",
                   help=("do one pass now (still never during recording time); for a PC "
                         "whose watcher is not running"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sender = build_sender_from_config()
    if args.preview:
        print("PREVIEW: nothing is copied or deleted. The real pass also waits until each "
              "file has been unchanged for %g minutes." % sender.settings.stable_minutes)
    report = sender.run_pass(preview=args.preview)
    _print_report(report)
    if args.once and any(str(w.get("reason", "")).startswith("may still be written")
                         for w in report.get("waiting") or []):
        print("A file is sent only after it has been seen unchanged for %g minutes. "
              "Run this command again after that time (the running watcher does this "
              "on its own)." % sender.settings.stable_minutes)
    return 1 if report.get("held") else 0


if __name__ == "__main__":
    raise SystemExit(main())
