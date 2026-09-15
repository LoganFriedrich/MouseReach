"""Scan the whole MouseReach_Pipeline tree and report every video by the folder
it physically lives in.

The dashboard's source of truth is the pipeline folders themselves: a video's
stage is simply *where it sits*. This walks every known pipeline folder (across
the Y: canonical side and the C: working side), finds every video, and -- when a
video appears in more than one folder -- keeps the furthest-along one.

It is intentionally coarse + fast-ish: it identifies videos by cheap markers
(media files, ``_reaches.json``, review-bundle dirs, quarantine notes) and does
NOT open each video's JSONs. Per-step / version detail comes from the Version
check and the File Details dialog (the selection-gated button on Pipeline
Overview). The ``Analyzed`` rglob is the slow part, so the caller runs this on a
background thread.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_REACHES = "_reaches.json"
_QUAR = ".quarantine.json"

# Dedup priority: if a stem shows up in several folders, the highest wins.
# A video sitting in a review queue (or quarantine/failed) is the ACTIONABLE
# state and wins even over 'analyzed' -- e.g. legacy triage bundles were staged
# from Analyzed, so the same stem lives in both; the operator needs to see the
# review-queue membership, not "done".
_PRIORITY = {
    "triage": 120, "deep_review": 120,
    "quarantined": 115, "failed": 110,
    "analyzed": 100,
    "processing": 50, "dlc_complete": 40,
    "cropped": 30, "raw_collage": 20,
}

_VALIDATED = {"analyzed"}
_NEEDS_REVIEW = {"triage", "deep_review"}
_BAD = {"failed", "quarantined"}

# Which pipeline steps are complete at each stage, so the per-step columns (DLC,
# Seg, Reach, Outcome) are meaningful under the folder scan.
_STEPS_DONE = {
    "raw_collage":  (),
    "cropped":      (),
    "dlc_complete": ("dlc",),
    "processing":   ("dlc",),                             # DLC done, MouseReach in progress
    "analyzed":     ("dlc", "seg", "reach", "outcome"),
    "triage":       ("dlc", "seg", "reach", "outcome"),   # ran the algos, held on a question
    "deep_review":  ("dlc",),                             # seg failed / escalated -> seg not trusted
    "quarantined":  (),
    "failed":       (),
}


def _bucket(state: str) -> str:
    if state in _VALIDATED:
        return "validated"
    if state in _NEEDS_REVIEW:
        return "needs_review"
    if state in _BAD:
        return "failed"
    return "in_progress"


def scan_pipeline_folders(progress: Optional[Callable[[str], None]] = None) -> Dict[str, Dict]:
    """Return ``{stem: dashboard_dict}`` for every video found across the pipeline
    folders. ``progress(message)`` is called before each phase."""
    from mousereach.config import Paths, WatcherConfig, parse_tray_type

    best: Dict[str, tuple] = {}   # stem -> (priority, state, path, mtime)

    def add(stem: str, state: str, path):
        if not stem:
            return
        pr = _PRIORITY.get(state, 0)
        cur = best.get(stem)
        if cur is None or pr > cur[0]:
            try:
                mt = Path(path).stat().st_mtime
            except OSError:
                mt = 0.0
            best[stem] = (pr, state, str(path), mt)

    def _glob_media(folder, state):
        if not folder or not Path(folder).exists():
            return
        folder = Path(folder)
        for ext in ("*.mp4", "*.mkv"):
            for f in folder.glob(ext):
                add(f.stem, state, f)

    if progress:
        progress("Scanning raw + cropped videos...")
    _glob_media(Paths.MULTI_ANIMAL_SOURCE, "raw_collage")
    _glob_media(Paths.SINGLE_ANIMAL_OUTPUT, "cropped")
    # A single a GPU node has claimed for pose was MOVED into
    # Single_Animal/.inflight/<host>/, which the top-level glob above never
    # sees. It is still a cut video waiting for its pose, so it goes in the
    # same "cropped" row -- otherwise it would vanish from the dashboard for
    # as long as the node holds it. The host is kept for the File Details view.
    claimed: Dict[str, str] = {}
    try:
        from mousereach.census.runner import claimed_singles, single_inflight_dir_name
        claimed = claimed_singles(Paths.SINGLE_ANIMAL_OUTPUT)
        inflight = Path(Paths.SINGLE_ANIMAL_OUTPUT or ".") / single_inflight_dir_name()
        for stem, host in sorted(claimed.items()):
            add(stem, "cropped", inflight / host / f"{stem}.mp4")
    except Exception as e:
        # The scan must still show everything else; say why claims are missing.
        logger.warning("could not read singles claimed for pose: %s", e)
    _glob_media(Paths.DLC_STAGING, "dlc_complete")

    if progress:
        progress("Scanning the working folder...")
    proc = Paths.PROCESSING
    if proc and Path(proc).exists():
        proc = Path(proc)
        for f in proc.glob(f"*{_REACHES}"):
            add(f.name[: -len(_REACHES)], "processing", f)
        for f in proc.glob("*.mp4"):
            add(f.stem, "processing", f)

    failed = Paths.FAILED
    if failed and Path(failed).exists():
        for f in Path(failed).glob("*.mp4"):
            add(f.stem, "failed", f)

    if progress:
        progress("Scanning the review queues...")
    for root, state in ((Paths.TRIAGE_REVIEW, "triage"), (Paths.DEEP_REVIEW, "deep_review")):
        if root and Path(root).exists():
            for d in Path(root).iterdir():
                # Every subdir here is taken to BE a review bundle (named for its
                # video), so tool/OS scratch dirs would otherwise be listed as
                # phantom videos awaiting review -- e.g. a stray ".omc/" (agent
                # state) showed up in the dashboard as a triage entry. Dot-dirs
                # are never review bundles; skip them.
                if d.is_dir() and not d.name.startswith("."):
                    add(d.name, state, d)

    try:
        qdir = WatcherConfig.load().get_quarantine_dir()
    except Exception:
        qdir = None
    if qdir and Path(qdir).exists():
        for j in Path(qdir).glob(f"*{_QUAR}"):
            add(j.name[: -len(_QUAR)].rsplit(".", 1)[0], "quarantined", j)

    # The big, slow one -- the final-output tree.
    if progress:
        progress("Scanning the final output (Analyzed) -- this is the big one, ~30s...")
    analyzed = Paths.ANALYZED_OUTPUT
    if analyzed and Path(analyzed).exists():
        # WHY iter_files and not rglob: superseded outputs sit under
        # Analyzed/Archive/ with their ORIGINAL names, so rglob listed an old
        # generation's _reaches.json as "analyzed" -- and since analyzed
        # outranks processing, a video back in Processing for a reprocess
        # showed as done. iter_files never enters a superseded folder.
        from mousereach.pipeline.analyzed_tree import iter_files
        for f in iter_files(analyzed, f"*{_REACHES}"):
            add(f.name[: -len(_REACHES)], "analyzed", f)

    # Roll each collage's offspring state up to the collage: a "raw_collage" that
    # already has its single-animal children downstream has in fact been cropped,
    # even though its file still sits in the Multi-Animal folder. Derive that from
    # the deterministic offspring names so the dashboard stops telling a collage it
    # "needs cropping" when 8/8 children are already in the pipeline. Prefer a saved
    # crop manifest for the offspring SET; read stages live from the scan.
    if progress:
        progress("Rolling collage offspring up to their collages...")
    downstream_index = {s: st for s, (pr, st, p, m) in best.items()}
    crop_rollup: Dict[str, Dict] = {}
    try:
        from mousereach.video_prep.core.collage_provenance import derive_offspring_status
        for stem, (pr, state, path, mt) in best.items():
            if state != "raw_collage":
                continue
            st = derive_offspring_status(stem, downstream_index)
            if st["n_expected"]:
                crop_rollup[stem] = st
    except Exception:
        pass

    if progress:
        progress(f"Building the list ({len(best)} videos)...")
    out: Dict[str, Dict] = {}
    for stem, (pr, state, path, mt) in best.items():
        tray = None
        try:
            tray = parse_tray_type(f"{stem}.mp4").get("tray_type")
        except Exception:
            pass
        ts = {}
        if mt:
            ts["updated"] = datetime.fromtimestamp(mt).isoformat()
        review = state if state in ("triage", "deep_review") else "none"
        steps = _STEPS_DONE.get(state, ())
        roll = crop_rollup.get(stem)
        out[stem] = {
            "locations": [{"stage": state, "path": path}],
            "versions": {},
            "timestamps": ts,
            "ground_truths": [],
            "status": _bucket(state),
            "current_stage": state,
            "metadata": {"state": state, "path": path},
            "dlc_status": "validated" if "dlc" in steps else "pending",
            "seg_status": "validated" if "seg" in steps else "pending",
            "reach_status": "validated" if "reach" in steps else "pending",
            "outcome_status": "validated" if "outcome" in steps else "pending",
            "archive_ready": state == "analyzed",
            "review_status": review,
            "tray_type": tray,
            "tray_supported": (tray not in ("E", "F")) if tray else True,
        }
        if state == "cropped" and stem in claimed:
            out[stem]["metadata"]["claimed_by"] = claimed[stem]
        if roll:
            # crop_state in {cropped, partial, uncropped}; counts for the note.
            out[stem]["crop_state"] = roll["crop_state"]
            out[stem]["offspring_present"] = roll["n_present"]
            out[stem]["offspring_expected"] = roll["n_expected"]
            out[stem]["offspring_complete"] = roll.get("n_complete", 0)
            # all_complete == every offspring reached the final Analyzed output ->
            # the collage is retirement-eligible (move to cold storage + the backup NAS).
            out[stem]["offspring_all_complete"] = roll.get("all_complete", False)
            # A collage whose children are all downstream is done with cropping --
            # don't flag it as needing attention.
            if roll["crop_state"] == "cropped":
                out[stem]["status"] = "in_progress"
    return out
