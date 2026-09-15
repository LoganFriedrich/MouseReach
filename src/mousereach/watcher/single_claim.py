"""Claim a video in the shared singles folder before posing it.

WHY THIS EXISTS
---------------
People drop cut single-animal videos into Unanalyzed/Single_Animal
(``Paths.SINGLE_ANIMAL_OUTPUT``) whenever they like -- the point is to get them
off the recording PCs at once -- while several GPU nodes poll that folder.
Every node that saw a video used to copy it into its own DLC queue and pose
it. Two nodes, one video: two ~14-minute GPU runs, two staged poses, and
duplicate processing downstream. Nothing in the folder said "taken".

THE CLAIM IS A RENAME
---------------------
A node takes a video by renaming it, inside the same share folder, from

    <Single_Animal>/<stem>.mp4
to  <Single_Animal>/.inflight/<hostname>/<stem>.mp4

A rename within one share is one operation on the file server: exactly one
node's rename finds the file, and every other node's fails with "file not
found" and leaves the video to the winner. A marker file is NOT atomic. The
processing server's ``Processing/Posed/.claims`` markers
(``ProcessingOrchestrator._claim_video``) are check-then-write, and need a
half-second sleep and a read-back to narrow -- never close -- the window in
which two nodes both write a marker and both believe they won. The re-pose
queue (watcher/repose.py) claims its requests by rename into its own
``.inflight`` for the same reason.

A rename is also refused on Windows while another program still has the file
open for writing, so a video still being copied in normally cannot be claimed
even if its size happened to hold still for the stability wait
(state.WatcherStateManager.discover_new_singles). The claim then simply fails
this poll and is tried again.

WHY ``.inflight`` IS A DOT-FOLDER UNDER THE SINGLES FOLDER
----------------------------------------------------------
Under the same folder so the rename never crosses a volume (a cross-volume
"rename" is a copy, and a copy is not a claim). A dot-folder so that readers
which skip dot-folders never mistake a claimed video for one still waiting.
The intake scan only lists files at the top of the singles folder, so a
claimed video is never registered again.

HOW LONG A CLAIM LIVES
----------------------
The claimed file stays in ``.inflight/<hostname>/`` until that node has handed
the video on (staged it, archived it, or routed it to a review queue) and
confirmed the next copy is there; it is then removed as a duplicate
(``DLCOrchestrator._retire_claimed_single``). Never before: until then it may
be the only copy on the share.

While its node is still working the video (or has handed it on but could not
confirm the next copy), the node touches it every poll, paused or not
(``heartbeat_claims``). A claim the node has given up on -- a failed pose, no
local copy left -- is released back to the folder at once where the failure is
seen, and otherwise simply stops being touched. A claim nobody has touched for
``STALE_S`` (a day, the re-pose request rule) means its node is gone or has
given up, and ``reclaim_stale`` moves the video back into the singles folder
for any node to take. Nothing here
ever deletes a video. The trade-off is the re-pose queue's: a node that comes
back after more than a day may pose a video another node has since taken.

The claim also sets the file's modified time to now. WHY: a rename keeps the
modified time, and for a video copied onto the share last week that is
already older than a day, so another node's stale sweep would hand the claim
straight back before this node's first heartbeat.

ASCII-only log text (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mousereach.config import Paths
# Same day-long rule as a claimed re-pose request, and for the same reason: a
# pose can wait behind a long queue or a long recording pause, and handing a
# live node's work to another node costs a duplicate pose.
from mousereach.watcher.repose import STALE_S

logger = logging.getLogger(__name__)

INFLIGHT_DIR = ".inflight"

# The start of the reason a GPU node records ('unresolvable') when a video it
# registered from the singles folder is gone by the time it tries to take it:
# another node claimed it, or somebody removed it. The intake scan
# (state.WatcherStateManager.discover_new_singles) re-validates ONLY rows
# carrying this reason once the file is back at the top of the folder --
# released after a failed copy, or returned by reclaim_stale. WHY only these:
# 'unresolvable' is also recorded for other reasons, and a row parked for one
# of those must not be re-driven just because a same-named file exists.
LEFT_FOLDER_REASON = "gone from the shared singles folder before this node took it"

# The start of the reason a GPU node records ('unresolvable') when a file in
# the singles folder has the name of a video ANOTHER node already holds in
# .inflight -- normally someone copied the same batch in again while the first
# copy was out being posed. The node does not claim it and does not pose it.
# WHY not LEFT_FOLDER_REASON: rows with that reason are re-driven when a file
# of the name is in the folder, which is exactly the double pose this refuses.
# A person removes the second copy; the holder carries on with the first.
DUPLICATE_OF_CLAIM_REASON = ("a second copy of a video another node already "
                             "holds for pose")


# ------------------------------------------------------------------ locations

def front_door() -> Optional[Path]:
    """The shared singles folder, read from Paths when called (never cached,
    so a changed configuration and the tests' patched Paths both apply)."""
    d = Paths.SINGLE_ANIMAL_OUTPUT
    return Path(d) if d else None


def inflight_root() -> Optional[Path]:
    """``<Single_Animal>/.inflight``, or None when no singles folder is set."""
    door = front_door()
    return door / INFLIGHT_DIR if door else None


def _valid_host(hostname) -> bool:
    # A host name becomes a folder name: refuse anything that would put the
    # claim somewhere other than one folder directly under .inflight.
    return (isinstance(hostname, str) and hostname not in ("", ".", "..")
            and not any(c in hostname for c in "/\\:"))


def host_dir(hostname: str) -> Optional[Path]:
    """``<Single_Animal>/.inflight/<hostname>``, or None."""
    root = inflight_root()
    if root is None or not _valid_host(hostname):
        return None
    return root / hostname


def claimed_path(video_id: str, hostname: str) -> Optional[Path]:
    """Where ``hostname``'s claim on ``video_id`` lives (it may not exist)."""
    d = host_dir(hostname)
    return d / f"{video_id}.mp4" if d else None


def _key(p) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def in_front_door(path) -> bool:
    """True when ``path`` sits directly in the singles folder (not in a
    subfolder such as ``.inflight``)."""
    door = front_door()
    if door is None or not path:
        return False
    try:
        return _key(Path(path).parent) == _key(door)
    except (TypeError, ValueError, OSError):
        return False


def claim_holder(path) -> Optional[str]:
    """The host whose claim folder ``path`` is in, or None when it is not a
    claimed copy (``<Single_Animal>/.inflight/<host>/<file>``)."""
    root = inflight_root()
    if root is None or not path:
        return None
    try:
        p = Path(path)
        if _key(p.parent.parent) == _key(root):
            return p.parent.name
    except (TypeError, ValueError, OSError):
        pass
    return None


# ---------------------------------------------------------------- the claim

def claim_single(src, hostname: str) -> Optional[Path]:
    """Take ``<Single_Animal>/<stem>.mp4`` for ``hostname`` by renaming it to
    ``<Single_Animal>/.inflight/<hostname>/<stem>.mp4``.

    Returns the claimed path, or None when the claim was not taken. None is
    never an error to raise, because a claim not taken is always safe (the
    video stays where every node can see it):

      * the file is gone -- another node's rename won, or somebody removed
        it (logged at DEBUG; the caller says what it does about it);
      * the rename was refused -- normally because a program still has the
        file open, e.g. it is still being copied in; tried again next poll;
      * ``src`` is not directly in the singles folder, the host name cannot
        be a folder name, or no singles folder is configured;
      * this host already holds a claimed copy of that name (never
        overwritten: it may be the only copy of an earlier drop).
    """
    src = Path(src)
    door = front_door()
    if door is None:
        logger.warning(f"{src.name}: no shared singles folder is configured; not claimed")
        return None
    if not in_front_door(src):
        logger.warning(f"{src.name}: not directly in the shared singles folder "
                       f"({door}); not claimed")
        return None
    hdir = host_dir(hostname)
    if hdir is None:
        logger.warning(f"{src.name}: host name {hostname!r} cannot name a claim "
                       f"folder; not claimed")
        return None
    dest = hdir / src.name
    try:
        hdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"{src.name}: could not create the claim folder {hdir} ({e}); "
                       f"not claimed this poll")
        return None
    try:
        if dest.exists():
            logger.warning(f"{src.stem}: {hostname} already holds a claimed copy at "
                           f"{dest}; the new file at {src} is left where it is and "
                           f"nothing is overwritten")
            return None
    except OSError as e:
        logger.warning(f"{src.stem}: could not check {dest} ({e}); not claimed this poll")
        return None

    # Touch BEFORE the rename as well as after: the rename keeps the modified
    # time, and between the rename and the second touch another node's stale
    # sweep could otherwise see a week-old claim (see the module docstring).
    try:
        os.utime(src, None)
    except FileNotFoundError:
        logger.debug(f"{src.stem}: gone from the singles folder before the claim")
        return None
    except OSError as e:
        logger.debug(f"{src.stem}: could not touch before claiming ({e}); "
                     f"the rename decides")

    try:
        os.replace(src, dest)
    except FileNotFoundError:
        logger.debug(f"{src.stem}: gone from the singles folder before the claim "
                     f"(another node took it, or it was removed)")
        return None
    except OSError as e:
        # DEBUG, not INFO: a refusal can repeat for as long as a program keeps
        # the file open, and the orchestrator reports it once itself (and
        # escalates to a WARNING if it lasts; DLCOrchestrator._note_single_claim_refused).
        logger.debug(f"{src.stem}: could not be claimed ({type(e).__name__}: {e}); "
                     f"normally a program still has it open (still being copied "
                     f"in). Left where it is.")
        return None

    try:
        os.utime(dest, None)
    except OSError:
        pass
    logger.info(f"{src.stem}: claimed for {hostname} ({dest})")
    return dest


def release_single(claimed) -> bool:
    """Move a claimed copy back to ``<Single_Animal>/<stem>.mp4``.

    True when it moved. False (logged) when it is not a claimed copy, has
    already gone, or a file of that name already sits in the singles folder:
    a release never overwrites. WHY: that file may be a fresh drop of the same
    recording, or a second copy a person put there; which one to keep is a
    person's call, and both are left in place for it.

    On Windows the move is ``os.rename``, which refuses an existing
    destination by itself, so a file dropped between the check and the move
    is not overwritten either. Elsewhere ``os.replace`` after the check.
    """
    claimed = Path(claimed)
    if claimed.parent.parent.name != INFLIGHT_DIR:
        logger.warning(f"{claimed}: not a claimed copy (not in a {INFLIGHT_DIR} host "
                       f"folder); not released")
        return False
    target = claimed.parent.parent.parent / claimed.name
    try:
        # Gone already -- normally another node's sweep released it a moment
        # earlier. Checked BEFORE the target: otherwise the file that node just
        # put back reads as a clash, and a WARNING asks a person to choose
        # between two copies when there is only one.
        if not claimed.exists():
            logger.debug(f"{claimed.stem}: claimed copy already gone (released by "
                         f"another node, or handed on); nothing to release")
            return False
        if target.exists():
            logger.warning(f"{claimed.stem}: not released -- {target} already exists. "
                           f"The claimed copy stays at {claimed}; a person should "
                           f"decide which copy to keep.")
            return False
    except OSError as e:
        logger.warning(f"{claimed.stem}: could not check {target} ({e}); not released")
        return False
    try:
        if os.name == "nt":
            os.rename(claimed, target)
        else:
            os.replace(claimed, target)
    except FileNotFoundError:
        logger.debug(f"{claimed.stem}: claimed copy already gone; nothing to release")
        return False
    except FileExistsError:
        logger.warning(f"{claimed.stem}: not released -- a file appeared at {target} "
                       f"meanwhile. The claimed copy stays at {claimed}.")
        return False
    except OSError as e:
        logger.warning(f"{claimed.stem}: could not release {claimed} back to the "
                       f"singles folder ({type(e).__name__}: {e})")
        return False
    logger.info(f"{claimed.stem}: released back to the shared singles folder "
                f"({target}) for any node to take")
    return True


# --------------------------------------------------------- keeping claims alive

def host_claims(hostname: str) -> Dict[str, Path]:
    """``{video_id: claimed path}`` for this host's claimed videos. Never
    raises; empty when there are none or the folder cannot be read."""
    hdir = host_dir(hostname)
    out: Dict[str, Path] = {}
    if hdir is None:
        return out
    try:
        for p in sorted(hdir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".mp4":
                out[p.stem] = p
    except OSError:
        pass
    return out


def heartbeat_claims(hostname: str, only: Optional[Iterable[str]] = None) -> int:
    """Touch this host's claimed videos so no other node's stale sweep hands
    them back. Returns how many were touched. Never raises.

    ``only`` (a set of video ids) limits the touch to those claims; None
    touches every file in the host's claim folder. The orchestrator passes the
    claims whose rows are still being worked or have been handed on
    (DLCOrchestrator._single_claim_upkeep). WHY: a claim this node has given up
    on -- a failed pose, no local file any more -- must NOT be kept alive, or
    the video is held from every other node for as long as this node runs.
    Left untouched, it goes stale after STALE_S and any running GPU node
    returns it to the folder.
    """
    hdir = host_dir(hostname)
    if hdir is None:
        return 0
    try:
        entries = list(hdir.iterdir())
    except FileNotFoundError:
        return 0
    except OSError as e:
        logger.warning(f"Could not read this node's claim folder {hdir} ({e}); its "
                       f"claims were not refreshed this poll")
        return 0
    keep = None if only is None else set(only)
    n = 0
    for p in entries:
        try:
            if keep is not None and p.stem not in keep:
                continue
            if p.is_file():
                os.utime(p, None)
                n += 1
        except OSError as e:
            logger.debug(f"could not touch claimed copy {p} ({e})")
    return n


# A stale claim that cannot go back because a same-named file already sits in
# the folder is reported once per claimed file per day, not on every sweep.
# WHY: every running GPU node sweeps every few minutes, and hundreds of
# identical WARNINGs a day bury the real ones. In memory: a restart may repeat
# it once, which is harmless.
_UNRELEASABLE_WARN_S = 24 * 3600
_unreleasable_warned: Dict[str, float] = {}


def reclaim_stale(now: Optional[float] = None) -> List[Path]:
    """Return to the singles folder every claimed video (any host's) whose
    modified time is more than ``STALE_S`` old. Returns the paths the videos
    are back at. Never deletes, never overwrites (``release_single``), never
    raises. Each return is logged.
    """
    root = inflight_root()
    if root is None:
        return []
    now = time.time() if now is None else now
    try:
        hosts = sorted(h for h in root.iterdir() if h.is_dir())
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning(f"Could not read the singles claim folder {root} ({e})")
        return []
    returned: List[Path] = []
    for h in hosts:
        try:
            files = sorted(p for p in h.iterdir()
                           if p.is_file() and p.suffix.lower() == ".mp4")
        except OSError as e:
            logger.debug(f"could not read claim folder {h} ({e})")
            continue
        for p in files:
            try:
                age = now - p.stat().st_mtime
            except OSError:
                continue
            if age <= STALE_S:
                continue
            target = h.parent.parent / p.name
            try:
                clash = target.exists()
            except OSError:
                clash = False
            if clash:
                key = _key(p)
                last = _unreleasable_warned.get(key)
                if last is None or now - last >= _UNRELEASABLE_WARN_S:
                    _unreleasable_warned[key] = now
                    logger.warning(
                        f"{p.stem}: claimed by {h.name} but not touched for "
                        f"{age / 3600:.0f} h, and cannot go back: {target} already "
                        f"exists (the video was dropped again). Nothing is "
                        f"overwritten; a person should keep one copy and delete "
                        f"the other. (Repeated at most once a day.)")
                else:
                    logger.debug(f"{p.stem}: stale claim still blocked by {target}")
                continue
            if release_single(p):
                back = h.parent.parent / p.name
                returned.append(back)
                logger.warning(f"{p.stem}: claimed by {h.name} but not touched for "
                               f"{age / 3600:.0f} h; returned to the shared singles "
                               f"folder ({back}) for any node to take")
    return returned


def inflight_claims() -> Dict[str, Tuple[str, float]]:
    """``{video_id: (hostname, last refreshed as epoch seconds)}`` for every
    claimed video. The time is the claimed file's modified time, which the
    holder's heartbeat keeps current. WHY readers want it: a claim held by a
    stopped node looks exactly like a live one by name alone; its age is what
    tells them apart. Never raises."""
    root = inflight_root()
    out: Dict[str, Tuple[str, float]] = {}
    if root is None:
        return out
    try:
        hosts = sorted(h for h in root.iterdir() if h.is_dir())
    except OSError:
        return out
    for h in hosts:
        try:
            for p in sorted(h.iterdir()):
                if p.is_file() and p.suffix.lower() == ".mp4" and p.stem not in out:
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    out[p.stem] = (h.name, mtime)
        except OSError:
            continue
    return out


def inflight_ids() -> Dict[str, str]:
    """``{video_id: hostname}`` for every claimed video, for readers (the
    reconciler, the census, the dashboard) that must count a claimed video as
    being worked, not as lost. Never raises; empty when nothing is claimed."""
    return {vid: host for vid, (host, _) in inflight_claims().items()}
