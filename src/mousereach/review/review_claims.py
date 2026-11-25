"""One review window per video: the reviewer's claim on a queue bundle.

WHY THIS EXISTS
-------------------------
Several people review at once, each in their own review window (on the
processing server, each in their own login). Each window picked its next video
by the lab's priority order and a random draw, with nothing telling it that
another window already had that video open. Two reviewers could be handed the
same video and answer the same pellets: the merged review keeps both people's
statements, but the work is done twice and, on a pellet both answered, the
later save wins.

THE CLAIM
---------
A window claims the bundle it shows: a marker file
``<review root>/.review_claims/<video>.review`` (the review root is the folder
that holds the triage and re-segmentation queues, ``Processing/Review``;
one marker per open review window). It
is created with the same exclusive create the watcher's claims use
(``watcher/claim_marker.take``): when two windows race for one video, exactly
one gets it. The marker names the window's machine and login
(``<host>/<user>``) and its process.

- A window picking its next video skips one another window holds.
- The claim is let go when the window moves to another video, opens a video
  by hand (Browse), loses the bundle, or is closed -- and, as a last resort,
  when the window's program exits.
- A claim is NOT held for ever by a window that crashed or was left open: a
  claim of the SAME login whose process has ended is taken over at once (the
  person reopened the tool); any other claim is broken once nobody has touched
  it for IDLE_AFTER_S by the file server's clock. Every saved answer and every
  load touches it.
- Opening a video another window holds, deliberately (from the dashboard or
  a worklist), is not refused: the person is told who has it open.

Nothing here moves, releases or changes a video. Removing the folder loses
nothing but the claims (open windows simply claim again on their next pick).

ASCII-only log text (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import atexit
import getpass
import logging
import os
import socket
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CLAIMS_DIR_NAME = ".review_claims"
SUFFIX = ".review"

# A claim nobody has touched (a load or a saved answer) for this long, by the
# file server's clock, is a window that crashed or was left open: it is broken
# so the video can be reviewed. WHY one hour: a reviewer saves an answer at
# least every few minutes while working a video; an hour without one is a
# window nobody is using. Breaking a live but idle claim costs at most a
# repeated look at the same video; keeping a dead one costs a video nobody can
# be handed.
IDLE_AFTER_S = 3600

_held = set()          # (claims dir, stem) this process holds


def who() -> str:
    """This window's machine and login, as written in its claims."""
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    return "%s/%s" % (socket.gethostname(), user)


def claims_dir(bundle: Path) -> Path:
    """``<review root>/.review_claims`` for a bundle in a queue folder
    (``<review root>/<queue>/<video>``). The review root is the configured
    ``Paths.REVIEW_ROOT`` when the bundle lies under it, else the bundle's
    grandparent (the same place on a configured machine)."""
    bundle = Path(bundle)
    try:
        from mousereach.config import Paths
        root = getattr(Paths, "REVIEW_ROOT", None)
        if root and Path(root) in bundle.parents:
            return Path(root) / CLAIMS_DIR_NAME
    except Exception:
        pass
    return bundle.parent.parent / CLAIMS_DIR_NAME


def _marker(bundle: Path) -> Path:
    return claims_dir(bundle) / (Path(bundle).name + SUFFIX)


def holder(bundle: Path) -> Optional[str]:
    """Who holds the claim on this bundle (``host/user``), None when nobody."""
    from mousereach.watcher.claim_marker import holder as _holder
    return _holder(claims_dir(bundle), Path(bundle).name, SUFFIX)


def _idle(bundle: Path) -> bool:
    """True when the claim's marker was last touched more than IDLE_AFTER_S
    ago by the file server's clock (a fresh probe file in the claims folder).
    False when that cannot be told -- a claim is never broken without
    evidence."""
    from mousereach.review.causal_review_io import _share_now
    d = claims_dir(bundle)
    try:
        touched = _marker(bundle).stat().st_mtime
    except OSError:
        return False
    now = _share_now(d)
    return now is not None and now - touched > IDLE_AFTER_S


def claim(bundle: Path) -> bool:
    """Claim ``bundle`` for this window. True when this window holds it
    afterwards (also when it already did); False while another window holds
    it. A claim that cannot be made at all (no claims folder can be created
    or written) is True: the claim prevents duplicate work, and refusing every
    video because the share is read-only would stop review altogether --
    logged once per video."""
    from mousereach.watcher import claim_marker as cm
    bundle = Path(bundle)
    d, stem = claims_dir(bundle), bundle.name
    me, proc = who(), cm.this_process()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("Review claims: cannot create %s (%s); %s is opened without "
                       "a claim", d, e, stem)
        return True
    if cm.take(d, stem, me, SUFFIX, process=proc):
        _held.add((str(d), stem))
        touch(bundle)
        return True
    if _idle(bundle):
        other = holder(bundle)
        try:
            os.remove(str(_marker(bundle)))
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning("Review claims: could not break the idle claim on %s "
                           "(%s)", stem, e)
            return False
        logger.info("Review claims: broke %s's claim on %s (untouched for over "
                    "%d minutes)", other or "(unknown)", stem, IDLE_AFTER_S // 60)
        if cm.take(d, stem, me, SUFFIX, process=proc):
            _held.add((str(d), stem))
            return True
    return False


def touch(bundle: Path) -> None:
    """Mark this window's claim as in use (a load or a saved answer). Never
    raises; does nothing for a claim this window does not hold."""
    bundle = Path(bundle)
    if (str(claims_dir(bundle)), bundle.name) not in _held:
        return
    try:
        os.utime(str(_marker(bundle)))
    except OSError:
        pass


def release(bundle: Optional[Path]) -> None:
    """Let go of this window's claim on ``bundle`` (another window's claim is
    never removed here). Never raises."""
    if bundle is None:
        return
    from mousereach.watcher import claim_marker as cm
    bundle = Path(bundle)
    d, stem = claims_dir(bundle), bundle.name
    try:
        cm.release(d, stem, who(), SUFFIX, process=cm.this_process())
    except Exception as e:
        logger.debug("Review claims: could not release %s (%s)", stem, e)
    _held.discard((str(d), stem))


def release_all() -> None:
    """Let go of every claim this process holds (registered to run at exit,
    so a window closed with the program still lets its video go)."""
    from mousereach.watcher import claim_marker as cm
    for d, stem in list(_held):
        try:
            cm.release(d, stem, who(), SUFFIX, process=cm.this_process())
        except Exception:
            pass
        _held.discard((d, stem))


atexit.register(release_all)
