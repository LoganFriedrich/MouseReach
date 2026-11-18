"""Filesystem hardening shared by the pipeline stages and the watcher.

WHY THIS EXISTS
---------------
A reviewer GUI (or an indexer/backup pass) briefly holding a pipeline file
surfaces as PermissionError winerror 5/32 (errno 13) and clears within
seconds. Work items died on exactly this, three ways:
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


# ---------------------------------------------------------------------------
# Atomic replacement of a whole file
# ---------------------------------------------------------------------------
#
# WHY: review files, ground truth, routing records and stage outputs were
# written in place (open "w", then write). Opening for writing empties the file
# first, so a crash, a killed window or a lost share connection part-way
# through leaves a truncated file. Readers treat an unreadable review copy as
# "no answers" and an unreadable ground-truth file as "no ground truth", so a
# torn write silently loses a person's work instead of failing.
#
# The writers below never open the real file for writing. They write the whole
# content to a temporary file in the SAME folder, flush it to disk, and then
# swap it in with one os.replace. A reader sees either the old complete file or
# the new complete file, never a mix. If anything fails, the old file is left
# exactly as it was and the temporary file is removed.
#
# The temporary name starts with a dot: ".<name>.<random>.tmp". WHY: the review
# router and the archiver take every file whose name STARTS WITH the video's
# stem; a temporary file named "<stem>....tmp" left by a crash would be carried
# along as if it were one of the video's outputs. A dot-name never matches a
# stem, and remove_leftover_temps finds it again by the same pattern.
#
# os.replace is retried: on Windows (and over SMB) it is refused with a sharing
# violation while another program has the destination open, which clears in
# seconds. After the retries the error is raised, so a stuck file fails loudly.

REPLACE_ATTEMPTS = 5
REPLACE_BASE_DELAY = 1.0


def _temp_prefix(path: Path) -> str:
    return "." + path.name + "."


def _fsync_file(p) -> None:
    """Flush a closed file's content to disk (best effort on shares that do
    not support it -- the replace is still atomic)."""
    import os
    # Read-write, not read-only: on Windows flushing a file needs write access
    # (a read-only handle's flush fails and would be silently skipped).
    try:
        fd = os.open(str(p), os.O_RDWR | getattr(os, "O_BINARY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _keep_mode(tmp: Path, path: Path) -> None:
    """Give the temporary file the permissions the replaced file had (or the
    usual 0644 for a new file). WHY: mkstemp creates its file readable by its
    owner only, and after the rename that would be the real file's mode -- on a
    POSIX share other users could no longer read a review or ground truth. (On
    Windows chmod only sets or clears read-only; this leaves it writable.)"""
    import os
    import stat
    try:
        mode = stat.S_IMODE(os.stat(str(path)).st_mode)
    except OSError:
        mode = 0o644
    try:
        os.chmod(str(tmp), mode)
    except OSError:
        pass


def _unlink_temp(p) -> bool:
    """Delete one of our temporary files; True when it is gone.

    WHY the second try: _keep_mode gives the temporary file the replaced
    file's permissions, so when that file is read-only (and the replace is
    therefore refused) the temporary file is read-only too, and on Windows a
    read-only file cannot be deleted. It is made writable and deleted again,
    so a refused write never leaves a temporary file behind."""
    import os
    import stat
    try:
        os.unlink(str(p))
        return True
    except FileNotFoundError:
        return True
    except OSError:
        pass
    try:
        os.chmod(str(p), stat.S_IREAD | stat.S_IWRITE)
        os.unlink(str(p))
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _replace_into_place(tmp: Path, path: Path, attempts: int,
                        base_delay: float) -> None:
    import os
    _keep_mode(tmp, path)
    try:
        retry_transient(lambda: os.replace(str(tmp), str(path)),
                        attempts=attempts, base_delay=base_delay,
                        what=str(path))
    except BaseException:
        _unlink_temp(tmp)
        raise


def write_text_atomic(path, text: str, encoding: str = "utf-8",
                      attempts: int = REPLACE_ATTEMPTS,
                      base_delay: float = REPLACE_BASE_DELAY) -> Path:
    """Replace ``path`` with ``text`` in one step (see the block comment above).

    The folder must exist. On failure the old file is untouched, no temporary
    file is left, and the error is raised."""
    import os
    import tempfile
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=_temp_prefix(path), suffix=".tmp",
                                    dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    except BaseException:
        _unlink_temp(tmp)
        raise
    _replace_into_place(tmp, path, attempts, base_delay)
    return path


def write_json_atomic(path, data, attempts: int = REPLACE_ATTEMPTS,
                      base_delay: float = REPLACE_BASE_DELAY,
                      trailing_newline: bool = False, **json_kwargs) -> Path:
    """Replace ``path`` with ``data`` as JSON in one step.

    The JSON text is built BEFORE anything touches the disk, so data that
    cannot be serialised raises with the old file untouched. ``json_kwargs``
    go to ``json.dumps`` (indent, default, ...)."""
    text = json.dumps(data, **json_kwargs)
    if trailing_newline:
        text += "\n"
    return write_text_atomic(path, text, attempts=attempts,
                             base_delay=base_delay)


def copy_file_atomic(src, dest, attempts: int = REPLACE_ATTEMPTS,
                     base_delay: float = REPLACE_BASE_DELAY) -> Path:
    """Copy ``src`` over ``dest`` in one step, keeping ``src``'s modified time
    (as shutil.copy2 does). ``dest``'s folder must exist."""
    import os
    import shutil
    import tempfile
    dest = Path(dest)
    fd, tmp_name = tempfile.mkstemp(prefix=_temp_prefix(dest), suffix=".tmp",
                                    dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(str(src), str(tmp))
        _fsync_file(tmp)
    except BaseException:
        _unlink_temp(tmp)
        raise
    _replace_into_place(tmp, dest, attempts, base_delay)
    return dest


def stage_copy(src, dest) -> Path:
    """Copy ``src`` to a temporary file beside ``dest`` (same folder, named
    like every temporary file here: ".<name>.<random>.tmp") and return that
    file's path; ``dest`` itself is not touched. The copy keeps ``src``'s
    modified time and is checked to be the same size. On failure nothing is
    left behind and the error is raised.

    WHY: a mover that must put several files in place together (filing a
    video's whole generation, archive/core.py) copies every file in first
    under a name no reader takes for the real one, and only then swaps them
    in with rename_with_retry (as archive_video does) -- a rename inside one folder, which is quick and
    cannot leave half a file."""
    import os
    import shutil
    import tempfile
    dest = Path(dest)
    fd, tmp_name = tempfile.mkstemp(prefix=_temp_prefix(dest), suffix=".tmp",
                                    dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(str(src), str(tmp))
        _fsync_file(tmp)
        if tmp.stat().st_size != Path(src).stat().st_size:
            raise OSError(f"the copy of {Path(src).name} came out a different size")
    except BaseException:
        _unlink_temp(tmp)
        raise
    return tmp


def aside_path(path) -> Path:
    """A free temporary name beside ``path`` (".<name>.<random>.tmp") to move
    the current file to while a new one takes its place (the renames in archive_video, archive/core.py). Only the
    name is chosen; nothing is created."""
    import uuid
    path = Path(path)
    return path.with_name(f"{_temp_prefix(path)}{uuid.uuid4().hex[:12]}.old.tmp")


def rename_with_retry(src, dst, attempts: int = REPLACE_ATTEMPTS,
                      base_delay: float = REPLACE_BASE_DELAY) -> None:
    """``os.replace(src, dst)``, retried while another program briefly holds
    one of them (retry_transient). Raises the last error. Unlike the atomic
    writers it removes nothing when it fails: the caller decides."""
    import os
    retry_transient(lambda: os.replace(str(src), str(dst)),
                    attempts=attempts, base_delay=base_delay, what=str(dst))


def remove_leftover_temps(path) -> int:
    """Delete temporary files a crashed atomic write of ``path`` left behind.

    Only for a writer that knows nobody else is writing ``path`` at the same
    moment (a temporary file in use by a live writer would be deleted under
    it). Returns how many were removed."""
    path = Path(path)
    n = 0
    try:
        for p in path.parent.glob(_glob_escape(_temp_prefix(path)) + "*.tmp"):
            if _unlink_temp(p):
                n += 1
    except OSError:
        pass
    return n


def remove_temp_file(p) -> bool:
    """Delete one temporary file found by is_leftover_temp (read-only
    included); True when it is gone."""
    return _unlink_temp(p)


def is_leftover_temp(name: str) -> bool:
    """True for a file name this module's writers use for a temporary file
    (".<name>.<random>.tmp"). A mover that takes a whole folder (the review
    return) uses it to leave such a crash leftover out of what it carries."""
    return name.startswith(".") and name.endswith(".tmp")


def _glob_escape(s: str) -> str:
    import glob
    return glob.escape(s)


# ---------------------------------------------------------------------------
# WHICH FILES IN A FOLDER BELONG TO ONE VIDEO
#
# A video's files are named <stem>.mp4, <stem>_<output>.json and
# <stem>DLC_<scorer>.h5. A bare "starts with the stem" test also takes the
# files of every video whose stem merely BEGINS with this one: tray numbers
# are any number of digits, so <date>_<animal>_P1 is a prefix of
# <date>_<animal>_P10, and filing, staging or cleaning up the first would
# move or delete the second's files -- leaving that video in no place. So the
# character after the stem must be one a file of THIS video can carry.

# Names copy programs (and transfer.safe_copy) give a file while it is still
# being written, renaming it once complete. Such a file is unfinished by
# definition and is never taken for one of a video's files.
TEMP_SUFFIXES = ('.part', '.tmp', '.partial')

# '.' -> <stem>.mp4 / <stem>.<ext>; '_' -> <stem>_<output>.json;
# 'D' -> <stem>DLC_<scorer>.h5 (the pose tool appends its scorer directly).
VIDEO_NAME_BOUNDARY = (".", "_", "D")


def is_file_of_video(name: str, video_id: str, include_temp: bool = False) -> bool:
    """True when the file name ``name`` belongs to ``video_id`` (see above)."""
    if not video_id or not name.startswith(video_id):
        return False
    n = len(video_id)
    if len(name) > n and name[n] not in VIDEO_NAME_BOUNDARY:
        return False
    if not include_temp and name.lower().endswith(TEMP_SUFFIXES):
        return False
    return True


def files_of_video(directory, video_id: str, include_temp: bool = False) -> list:
    """Every file in ``directory`` that belongs to ``video_id`` -- the one
    rule every mover uses (filing, staging, clean-up). A missing folder holds
    nothing. Unfinished copies (TEMP_SUFFIXES) are left out unless asked for.

    The name is tested before asking whether the entry is a file, and the
    listing's own entry answers that (no extra round trip on Windows): the
    folders scanned can hold tens of thousands of entries on a share."""
    import os
    out = []
    try:
        with os.scandir(directory) as entries:
            for e in entries:
                if is_file_of_video(e.name, video_id, include_temp) and e.is_file():
                    out.append(Path(e.path))
    except (FileNotFoundError, NotADirectoryError):
        return []
    return out
