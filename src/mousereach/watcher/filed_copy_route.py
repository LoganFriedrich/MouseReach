"""Put a FILED video into a review queue while its filed copy stays filed.

THE RULE ("filed is final")
----------------------------
Once a video is filed its results are the final answer. Nothing automatic ever
takes a filed video back into a queue. A PERSON may: ``mousereach-route-to-
queue`` run by hand, or a 'wrong pellet' answer a person saved on that filed
video (the version scan's mislabel divert). Even then the filed copy STAYS in
Analyzed until the new run is filed and replaces it (archive/core.archive_video
supersedes the old generation at that moment), exactly as for a filed video
whose re-run is held in a queue.

WHY a copy and not a move: a move emptied the filed folder the moment the
question was asked. Until a person answered it, the video had no filed result
anywhere, so anything that reads Analyzed (the database integrator, a status
count, a person looking for the video) found nothing -- for a video whose
filed result was still the best answer anyone had.

HOW
---
1. The video's files in its filed folder are COPIED (checksum-verified) into
   ``<queue root>/.filed_copies/<video>/``. A dot-folder: the queue readers
   that judge, retire or release bundles skip names starting with "."
   (review_routing.INCOMING_DIR_NAME says which), so a half-made copy is never
   taken for a bundle. Same volume as the queue, so step 2 is a rename.
2. That copy is routed into the queue by the ordinary router
   (review_gate.route_to_queue, all or nothing), which writes the routing
   record and the watcher state exactly as for any other route.
3. The copy folder is removed. Nothing in the filed folder is moved, renamed
   or written at any step.

If anything fails, the copy folder is deleted (it holds copies only) and the
error is raised: nothing was routed, and the filed folder is byte for byte as
it was.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Under each queue root; see the module docstring (step 1) for why a dot-folder.
FILED_COPIES_DIR_NAME = ".filed_copies"


class FiledCopyError(RuntimeError):
    """The filed folder's files could not all be copied; nothing was routed."""


def route_filed_copy(video_id: str, filed_dir, dest_root, reason: str, *,
                     db=None, db_state: Optional[str] = None,
                     held_at_step: Optional[str] = None) -> Path:
    """Copy ``video_id``'s files from ``filed_dir`` into ``dest_root`` (a
    review queue) as a new bundle, leaving the filed folder untouched.
    Returns the bundle folder. Raises (nothing routed) on any failure,
    including the router's own refusals (AlreadyInQueueError,
    CrystallizedError ...), which pass through unchanged."""
    from mousereach.watcher.review_gate import route_to_queue
    from mousereach.watcher.review_routing import _bundle_files
    from mousereach.watcher.transfer import safe_copy

    filed_dir = Path(filed_dir)
    dest_root = Path(dest_root)
    files = _bundle_files(filed_dir, video_id)
    if not files:
        raise FiledCopyError(f"{video_id}: no files of the video in {filed_dir}")
    staging = dest_root / FILED_COPIES_DIR_NAME / video_id
    if staging.exists():
        # A copy left by an earlier attempt that stopped part-way: copies
        # only, never the filed files themselves, so it is made again.
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        failed = [f.name for f in files if not safe_copy(f, staging / f.name, verify=True)]
        if failed:
            raise FiledCopyError(
                f"{video_id}: could not copy {', '.join(sorted(failed))} out of "
                f"{filed_dir}; nothing routed, the filed folder is unchanged")
        bundle = route_to_queue(video_id, staging, dest_root, reason=reason, db=db,
                                db_state=db_state, held_at_step=held_at_step,
                                all_or_nothing=True)
        logger.info("%s: filed copy put into %s; the filed results stay in %s "
                    "until the new run is filed", video_id, bundle, filed_dir)
        return Path(bundle)
    finally:
        # Empty after a successful route (every file was moved into the
        # bundle); holds copies only after a failed one.
        shutil.rmtree(staging, ignore_errors=True)
        try:
            staging.parent.rmdir()       # only when empty
        except OSError:
            pass
