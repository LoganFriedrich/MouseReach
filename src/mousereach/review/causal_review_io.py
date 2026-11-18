"""
Causal Review I/O -- save format and corpus index for the causal review tool.

Three outputs per reviewed video:
  1. DURABLE copy at ``{NAS_ROOT}/review_records/reviews/{video}_causal_review.json``
  2. Working copy ``{video}_causal_review.json`` beside whatever the reviewer
     had open (a triage bundle, a processing dir, the canonical results dir)
  3. Corpus index (append/update) at
     ``{NAS_ROOT}/review_records/causal_review_index.json``

The durable copy exists because the working copy is not durable and was, for a
long time, the only copy. A review is written into a triage bundle; the bundle
is transient -- reprocessing regenerates it, and returning a cleared bundle
MOVES its files onto one node's local disk and removes the bundle. A review only
reached shared storage that survives when its video was archived, so whether a
reviewer's work outlived a reprocess came down to timing: many reviews existed
only on one machine's local disk, and some only inside a triage bundle -- one
reprocess away from gone, and not reconstructable from the index.

So the durable copy is written FIRST and never depends on the bundle's
lifecycle. The working copy may still be written wherever the reviewer is; it
must never be the only one.

The save is idempotent: re-reviewing a (video, segment) pair
overwrites the prior entry in both per-video file and index.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_username() -> str:
    return os.environ.get("USERNAME", os.environ.get("USER", "unknown"))


def bundle_manifest_path(bundle_dir, video_stem: Optional[str] = None) -> Path:
    """Path to a review bundle's manifest.

    Honors the project naming convention -- every file associated with an origin
    source carries that video's full title first: ``{stem}_manifest.json`` -- with
    a fallback to the legacy bare ``manifest.json`` for bundles staged before the
    convention. When neither exists, returns the stem-named path (the write
    target)."""
    bundle_dir = Path(bundle_dir)
    stem = video_stem or bundle_dir.name
    named = bundle_dir / f"{stem}_manifest.json"
    if named.exists():
        return named
    legacy = bundle_dir / "manifest.json"
    if legacy.exists():
        return legacy
    return named


def durable_review_dir() -> Optional[Path]:
    """The directory that holds the copy of record for every review.

    Beside the corpus index, on shared storage: the index is already the one
    place every node agrees to look, and a backup of the NAS covers it. Returns
    None only when neither a NAS nor a processing root is configured, in which
    case the caller keeps its working copy and says so.
    """
    base = _default_index_dir()
    return None if base is None else base / "reviews"


def durable_review_path(video_stem: str) -> Optional[Path]:
    """Where this video's copy of record lives, or None if unconfigured."""
    d = durable_review_dir()
    return None if d is None else d / f"{video_stem}_causal_review.json"


def resolve_review_path(video_stem: str, primary_dir=None) -> Optional[Path]:
    """Locate a saved ``{video_stem}_causal_review.json`` across the canonical
    save locations, NEWEST first: an explicit ``primary_dir`` (e.g. a processing
    dir), the routine Pending bundle, and next to the canonical video. Returns the
    Path (which ``load_and_apply_review`` accepts directly) or None.

    The review tool saves the per-video review either in the bundle dir (routine
    manifest mode, Pending/<stem>/) or next to the video -- neither of which is
    the pipeline's processing dir. This resolver lets any (re)processing find and
    auto-apply the reviewer's resolution wherever it was saved. It NEVER raises;
    a missing review yields None and the extractor no-ops.

    The durable store is searched too, and it is the reason a reprocess can no
    longer strand kinematics without the reviewer's answers: the bundle can be
    regenerated, the local processing dir cleared, the video not yet archived,
    and the review is still found."""
    candidates: List[Path] = []
    if primary_dir is not None:
        candidates.append(Path(primary_dir))
    try:
        from mousereach.review.staging import (
            DEFAULT_PENDING_DIR, resolve_canonical_paths, DEFAULT_CONNECTOME_ROOT)
        candidates.append(Path(DEFAULT_PENDING_DIR) / video_stem)
        try:
            candidates.append(Path(
                resolve_canonical_paths(video_stem, DEFAULT_CONNECTOME_ROOT)["mp4"]).parent)
        except Exception:
            pass
    except Exception:
        pass
    try:
        d = durable_review_dir()
        if d is not None:
            candidates.append(d)
    except Exception:
        pass
    found: List[Path] = []
    for d in candidates:
        try:
            p = d / f"{video_stem}_causal_review.json"
            if p.is_file():
                found.append(p)
        except Exception as err:
            logger.warning('Could not look for a review in %s: %s', d, err)
            pass
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def review_path_index(stems) -> Dict[str, tuple]:
    """``{stem: (path, mtime, size)}``: for every stem, the review
    ``resolve_review_path(stem)`` would return, found from a few folder
    listings instead of several network checks per video.

    Same places, same order, same rule (newest modified time wins; on a tie
    the earlier place): the routine Pending bundle, next to the canonical video
    (only where ``resolve_canonical_paths`` resolves), the durable store.
    WHY: the archive's version scan asks this for every archived video every
    half hour; per video it cost several missed lookups on the share --
    minutes of every scan on a large archive, while the work loop waited. Stems with no review are absent. Never raises; a place that
    cannot be listed contributes nothing, exactly as a failed check does.
    """
    suffix = "_causal_review.json"
    stems = set(stems)
    places: Dict[str, List[tuple]] = {}

    def add(stem, path):
        try:
            st = Path(path).stat()
        except OSError:
            return
        places.setdefault(stem, []).append((Path(path), st.st_mtime, st.st_size))

    try:
        from mousereach.review.staging import (
            DEFAULT_PENDING_DIR, resolve_canonical_paths, DEFAULT_CONNECTOME_ROOT,
            cohort_dir_for_stem)
    except Exception:
        DEFAULT_PENDING_DIR = DEFAULT_CONNECTOME_ROOT = None
    # 1. the routine Pending bundle: <pending>/<stem>/<stem>_causal_review.json
    if DEFAULT_PENDING_DIR:
        try:
            with os.scandir(DEFAULT_PENDING_DIR) as it:
                bundles = {e.name for e in it if e.is_dir()}
        except OSError:
            bundles = set()
        for stem in stems & bundles:
            add(stem, Path(DEFAULT_PENDING_DIR) / stem / f"{stem}{suffix}")
    # 2. beside the canonical video. resolve_canonical_paths needs the video in
    # <analyzed root>/<cohort folder>; a cohort folder that does not exist
    # cannot resolve for any of its stems, so it is checked once per cohort,
    # and the per-stem resolution runs only where the folder exists.
    if DEFAULT_CONNECTOME_ROOT:
        by_cohort: Dict[str, List[str]] = {}
        for stem in stems:
            try:
                by_cohort.setdefault(cohort_dir_for_stem(stem), []).append(stem)
            except Exception:
                continue
        for cohort, members in by_cohort.items():
            try:
                if not (Path(DEFAULT_CONNECTOME_ROOT) / cohort).is_dir():
                    continue
            except OSError:
                continue
            for stem in members:
                try:
                    folder = Path(resolve_canonical_paths(stem, DEFAULT_CONNECTOME_ROOT)["mp4"]).parent
                except Exception:
                    continue
                if (folder / f"{stem}{suffix}").is_file():
                    add(stem, folder / f"{stem}{suffix}")
    # 3. the durable store: one listing.
    try:
        d = durable_review_dir()
    except Exception:
        d = None
    if d is not None:
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.endswith(suffix):
                        stem = e.name[: -len(suffix)]
                        if stem in stems and e.is_file():
                            st = e.stat()
                            places.setdefault(stem, []).append((Path(e.path), st.st_mtime, st.st_size))
        except OSError:
            pass
    return {stem: max(found, key=lambda f: f[1]) for stem, found in places.items()}


def _get_timestamp() -> str:
    return datetime.now().isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Replace ``path`` with ``data`` in one step (fsutil.write_json_atomic).

    Every review copy, clear marker, the corpus index and flagged_sessions.json
    go through here. WHY not ``write_text``: that empties the file before
    writing it, so a crash mid-save left a truncated review, and
    ``review_copies`` skips an unreadable copy -- the next save then had nothing
    to carry forward from it and the person's answers were gone. Now the old
    file stays complete until the new one is swapped in; a write that still
    fails after the retries raises, with the old file unchanged.
    """
    from mousereach.pipeline.fsutil import write_json_atomic
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, data, indent=2, default=str, trailing_newline=True)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def collect_provenance(video_dir: Path, video_stem: str) -> Dict[str, Any]:
    """Read version stamps from the algo JSON files next to the video.

    Pulls ``version`` / ``detector`` from _reaches.json,
    _pellet_outcomes.json, _segments.json, and _reach_assignments.json
    when they exist. Also reads pipeline_versions.json if present.
    """
    prov: Dict[str, Any] = {}

    for suffix, key in [
        ("_segments.json", "segmenter"),
        ("_reaches.json", "reach_detector"),
        ("_pellet_outcomes.json", "outcome_detector"),
        ("_reach_assignments.json", "assignment"),
    ]:
        path = video_dir / f"{video_stem}{suffix}"
        if not path.exists():
            # Try _segmentation.json as alternate
            if suffix == "_segments.json":
                path = video_dir / f"{video_stem}_segmentation.json"
            if not path.exists():
                continue
        try:
            data = _read_json(path)
            entry: Dict[str, Any] = {}
            if "version" in data:
                entry["version"] = data["version"]
            if "detector" in data:
                entry["detector"] = data["detector"]
            if "segmenter_version" in data:
                entry["version"] = data["segmenter_version"]
            prov[key] = entry
        except Exception:
            pass

    # DLC model/scorer from HDF5 column header (best-effort)
    try:
        import pandas as pd
        from mousereach.pipeline.manifest import select_pose_file
        h5_files = list(video_dir.glob(f"{video_stem}*.h5"))
        if h5_files:
            df = pd.read_hdf(select_pose_file(h5_files), stop=0)
            if hasattr(df.columns, "levels"):
                scorer = df.columns.get_level_values(0)[0]
                prov["dlc_scorer"] = scorer
    except Exception:
        pass

    return prov


# ---------------------------------------------------------------------------
# Per-segment review record
# ---------------------------------------------------------------------------

def build_segment_record(
    segment_num: int,
    pellet_num: Optional[int],
    algo_outcome: Optional[str],
    algo_causal_reach: Optional[Dict[str, int]],
    algo_interaction_frame: Optional[int],
    human_outcome: Optional[str],
    human_causal_reach: Optional[Dict[str, int]],
    is_phantom: bool,
    agreed: bool,
    answers: Dict[str, Any],
    notes: str = "",
    segment_span: Optional[Dict[str, int]] = None,
    true_segment_num: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a single segment's review record.

    ``segment_span`` is the segment's frame range ({"start": .., "end": ..}) as
    it stood when the human looked at it, and it is what makes the record
    durable. A human review is a fact about what happens in a stretch of video,
    so it stays true no matter which algo or DLC version runs next. But the
    record was anchored only to ``segment_num`` -- an index the segmenter
    reassigns every time it re-cuts the video. Re-segment, and segment 7's
    review silently starts describing a different stretch of footage.

    With the span stored, a review can always be re-attached to the frames it
    was actually about, and a reviewer's note like "this boundary is wrong"
    survives the fix to that boundary instead of being invalidated by it.

    Optional because reviews written before this field existed do not have it;
    those were backfilled in place from the segmentation each was made against
    (see scripts/backfill_causal_review_spans.py).

    ``true_segment_num`` is for when the segmentation is wrong about WHICH pellet
    presentation this stretch is -- boundaries offset by one, so what it calls
    segment 1 is really segment 2. That is a fact about the segmentation rather
    than about the reach or the outcome, and a reviewer needs to be able to say
    it in one action instead of writing prose nothing can read. It is recorded
    only when it differs from what the segmenter said; the frame span still says
    which footage was judged, so the review stays anchored either way.
    """
    rec: Dict[str, Any] = {
        "segment_num": segment_num,
        "pellet_num": pellet_num,
        "segment_span": segment_span,
        "algo": {
            "outcome": algo_outcome,
            "causal_reach": algo_causal_reach,
            "interaction_frame": algo_interaction_frame,
        },
        "human": {
            "outcome": human_outcome,
            "causal_reach": human_causal_reach,
            "is_phantom": is_phantom,
            "agreed": agreed,
        },
        "answers": answers,
        "notes": notes,
    }
    if true_segment_num is not None and true_segment_num != segment_num:
        rec["true_segment_num"] = int(true_segment_num)
        rec["segmentation_wrong"] = True
    return rec


# ---------------------------------------------------------------------------
# Save per-video review file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Who answered each segment
# ---------------------------------------------------------------------------
# WHY per segment: a review file used to carry ONE "reviewer" and
# "reviewed_at" for the whole file. One person saving a single answer in a
# video another person had answered earlier then made the file credit the
# second person with every answer in it, the first person's included. Each
# segment's record now carries its own "reviewer" and "reviewed_at", set only
# when that segment's answer is new or changed in a save; an unchanged answer
# keeps the values it had. The file-level fields stay as they were and mean
# "who saved the file last, and when". Readers that need "who answered
# segment N" use segment_reviewer(), which falls back to the file-level
# fields for files written before this (2026-10-28).

# The parts of a record that ARE the person's answer. Not "algo" (it changes
# when the algorithm is re-run), not human.agreed (it is derived from the
# algorithm's call), not segment_num / pellet_num / segment_span (a re-cut
# renumbers them without anyone answering anything).
def _answer_signature(rec: Dict[str, Any]) -> str:
    human = rec.get("human") or {}
    return json.dumps({
        "outcome": human.get("outcome"),
        "causal_reach": human.get("causal_reach"),
        "is_phantom": human.get("is_phantom"),
        "answers": rec.get("answers"),
        "notes": rec.get("notes"),
        "true_segment_num": rec.get("true_segment_num"),
    }, sort_keys=True, default=str)


def _previous_review(video_stem: str, output_dir: Path) -> Optional[Dict[str, Any]]:
    """The review as last saved -- the working copy about to be replaced, or
    else the durable copy -- or None. Never raises: without it every answer
    in this save is simply treated as given now."""
    for path in (Path(output_dir) / f"{video_stem}_causal_review.json",
                 durable_review_path(video_stem)):
        try:
            if path is not None and Path(path).is_file():
                return _read_json(Path(path))
        except Exception as err:
            logger.warning('Could not read the saved review %s, so every answer counts as given now: %s', path, err)
            continue
    return None


def _previous_reviews(docs) -> Optional[Dict[str, Any]]:
    """Every earlier copy's records in one document, each carrying its own
    reviewer and time, for stamp_segment_reviewers: an answer kept from ANY
    copy keeps the name and time of the person who gave it. None when there
    is no earlier copy."""
    docs = [d for d in (docs or []) if isinstance(d, dict)]
    if not docs:
        return None
    recs = [_with_own_reviewer(r, d) for d in docs
            for r in (d.get("segments") or []) if isinstance(r, dict)]
    return {"reviewer": docs[0].get("reviewer"),
            "reviewed_at": docs[0].get("reviewed_at"), "segments": recs}


def segment_reviewer(rec: Dict[str, Any], doc: Optional[Dict[str, Any]] = None):
    """(reviewer, reviewed_at) of ONE segment's answer: the record's own
    values, or -- for a file saved before records carried them -- the
    file-level ones."""
    rec, doc = rec or {}, doc or {}
    return (rec.get("reviewer") or doc.get("reviewer"),
            rec.get("reviewed_at") or doc.get("reviewed_at"))


def stamp_segment_reviewers(
    segments: List[Dict[str, Any]],
    previous: Optional[Dict[str, Any]],
    reviewer: str,
    timestamp: str,
    recut: Optional[datetime] = None,
    carried=(),
) -> List[Dict[str, Any]]:
    """Give every answered record its own "reviewer" and "reviewed_at"
    (see the comment block above for why). For each record:

    - not answered (a ``reviewed: False`` placeholder): no reviewer -- nobody
      answered it; any left over from an earlier answer is removed;
    - the same answer as the previous save's record for that segment (matched
      by segment number, else by frame span): keeps that record's reviewer
      and time, or, if the previous file had none per record, the previous
      file's own -- the person who saved that file gave that answer;
    - otherwise (new or changed): this save's reviewer and time.

    ``recut``: when a person set the current cuts. A person's answer in THIS
    save never inherits a time from before then -- it was given now, on the
    new cuts. WHY: an answer re-given after a re-cut is often the same as the
    voided one (same pellet number, same outcome); inheriting the old time
    left it void (answer_survives_recut), so the pellet was asked again on
    every pass for ever. ``carried``: segment numbers whose record this save
    did not answer but took back from an earlier copy (carry_earlier_answers)
    -- those keep their own time, so a voided answer carried along as a record
    is never made to look as if it had been given on the new cuts.

    Returns new record dicts; the caller's are not changed."""
    prev_doc = previous or {}
    by_num: Dict[int, List[Dict[str, Any]]] = {}
    by_span: Dict[str, List[Dict[str, Any]]] = {}
    for p in prev_doc.get("segments") or []:
        if p.get("segment_num") is not None:
            by_num.setdefault(int(p["segment_num"]), []).append(p)
        if p.get("segment_span"):
            by_span.setdefault(json.dumps(p["segment_span"], sort_keys=True), []).append(p)
    out = []
    for rec in segments:
        rec = dict(rec)
        if (rec.get("answers") or {}).get("reviewed") is False:
            rec.pop("reviewer", None)
            rec.pop("reviewed_at", None)
            out.append(rec)
            continue
        sig = _answer_signature(rec)
        cands = []
        if rec.get("segment_num") is not None:
            cands.extend(by_num.get(int(rec["segment_num"])) or [])
        if rec.get("segment_span"):
            cands.extend(by_span.get(json.dumps(rec["segment_span"], sort_keys=True)) or [])
        same = next((p for p in cands if p is not None
                     and (p.get("answers") or {}).get("reviewed") is not False
                     and _answer_signature(p) == sig), None)
        if same is not None:
            who, when = segment_reviewer(same, prev_doc)
        elif not prev_doc and rec.get("reviewer") and rec.get("reviewed_at"):
            # No previous file to compare with (moved, or not readable) and the
            # record already says who answered it: it came back from a saved
            # review unchanged -- keep it.
            who, when = rec["reviewer"], rec["reviewed_at"]
        else:
            who, when = reviewer, timestamp
        if (recut is not None and when and rec.get("segment_num") is not None
                and int(rec["segment_num"]) not in carried
                and not answer_survives_recut({"reviewed_at": when}, None, recut)):
            who, when = reviewer, timestamp
        rec["reviewer"] = who or reviewer
        rec["reviewed_at"] = when or timestamp
        out.append(rec)
    return out


def save_causal_review(
    video_stem: str,
    output_dir: Path,
    segments: List[Dict[str, Any]],
    provenance: Dict[str, Any],
    reviewer: Optional[str] = None,
    manual_segmentation: Optional[Dict[str, Any]] = None,
    unmatched_answers: Optional[List[Dict[str, Any]]] = None,
    other_copies=(),
    problems: Optional[List[str]] = None,
    cuts_set_at: Optional[str] = None,
) -> Path:
    """Write (or overwrite) the per-video causal review JSON.

    ``cuts_set_at``: when a person set the cuts this save's answers were given
    on (the segments file's ``corrected_at``, causal_review_io.cuts_set_at);
    an answer given in this save is then never stamped with an earlier time
    (stamp_segment_reviewers says why).

    A save can add or change answers; it never drops one. Before writing, every
    existing copy is read -- the working copy about to be replaced, the durable
    copy, and any folder in ``other_copies`` -- and for each segment this save
    does not answer, the newest answer (or recheck) an earlier copy holds for
    its frames is kept in its place; an earlier answer matching no current
    segment is kept in ``unmatched_answers`` (carry_earlier_answers). WHY: the
    durable copy used to be overwritten with whatever the window held, so an
    answer the window had not loaded was replaced by a "not reviewed"
    placeholder and lost from every copy.

    Parameters
    ----------
    video_stem : str
        Video identifier (without extension).
    output_dir : Path
        Directory to write the WORKING copy into -- normally wherever the
        reviewer is looking. This may be a transient triage bundle, so it is
        never the only copy written (see ``durable_review_dir``).
    segments : list of dict
        Per-segment review records (from ``build_segment_record``). Each
        answered record is saved with its own "reviewer" / "reviewed_at"
        (``stamp_segment_reviewers``): this save's for an answer that is new
        or changed since the file was last saved, the earlier ones otherwise.
    provenance : dict
        Algo version stamps (from ``collect_provenance``).
    reviewer : str, optional
        Reviewer name; defaults to OS username. Saved at file level as "who
        saved the file last"; per segment only for the answers this save
        gives or changes.
    manual_segmentation : dict, optional
        History only: cuts an old version of the review window stored in the
        review file. When not given, the block an earlier copy holds is kept
        unchanged. No reader uses it; the segments file holds the cuts.
    unmatched_answers : list of dict, optional
        Earlier answers whose frames match no current segment
        (``restore_review_for_segments``). Kept under ``UNMATCHED_ANSWERS_KEY``
        so a re-cut never deletes what a person decided; nothing applies them.
    problems : list, optional
        A failure this save survives -- the durable copy could not be
        written, so the review is saved only in ``output_dir`` -- is appended
        here as a sentence for the person, besides being logged. WHY: the
        review window shows these; a log line alone was never seen, and the
        reviewer went on believing the review was safely kept.

    Returns
    -------
    Path to the working copy (the durable copy is written first, and its
    location is available from ``durable_review_path``).
    """
    reviewer = reviewer or _get_username()
    timestamp = _get_timestamp()
    earlier = review_copies(video_stem, [output_dir] + [d for d in (other_copies or ()) if d])
    segments, unmatched_answers, kept = carry_earlier_answers(
        segments, earlier, unmatched_answers)
    if kept:
        logger.warning(
            "%s: segment(s) %s were not answered in this save, but an earlier "
            "copy holds a person's answer for their frames; that answer is kept",
            video_stem, ", ".join(str(k) for k in kept))
    segments = stamp_segment_reviewers(
        segments, _previous_reviews(earlier), reviewer, timestamp,
        recut=_when(cuts_set_at) if cuts_set_at else None, carried=set(kept))

    doc: Dict[str, Any] = {
        "type": "causal_review",
        "schema_version": "1.1",   # 1.1 adds segment_span to every record
        "video_stem": video_stem,
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "provenance": provenance,
        "segments": segments,
    }
    # Cuts an old version of the review window saved inside the review file.
    # Nothing reads them any more (the segments file holds the cuts), but they
    # are a record of what a person did, so they ride along UNCHANGED: the
    # block given here, else the first one an earlier copy holds. WHY: a save
    # from a window that had not loaded the block wrote the review without
    # it, and the record was gone from every copy.
    if not manual_segmentation:
        manual_segmentation = next(
            (d["manual_segmentation"] for d in earlier
             if isinstance(d, dict) and d.get("manual_segmentation")), None)
    if manual_segmentation:
        doc["manual_segmentation"] = manual_segmentation
    if unmatched_answers:
        doc[UNMATCHED_ANSWERS_KEY] = list(unmatched_answers)

    # The durable copy first. If the process dies between the two writes, the
    # surviving one is the copy nothing else can delete -- the working copy sits
    # in a directory the pipeline regenerates.
    durable = durable_review_path(video_stem)
    if durable is None:
        logger.warning(
            "%s: no NAS or processing root configured, so this review has only "
            "the copy in %s. That directory is not durable -- a reprocess can "
            "regenerate it. Configure a root (mousereach-setup) so reviews are "
            "kept.", video_stem, output_dir)
    else:
        try:
            _write_json(durable, doc)
        except Exception as e:
            # Loud, and not fatal: refusing to save would lose the reviewer's
            # work outright, which is worse than saving it in one place.
            logger.error(
                "%s: DURABLE review copy could not be written to %s (%s). The "
                "review is saved only at %s, which a reprocess can destroy. "
                "Copy it somewhere safe.", video_stem, durable, e, output_dir)
            if problems is not None:
                problems.append(
                    "The lasting copy of this review could not be written (%s: "
                    "%s). It is saved only in %s, which reprocessing can "
                    "replace. Save again; if this keeps happening, tell whoever "
                    "looks after the pipeline." % (durable.parent, e, output_dir))

    out_path = output_dir / f"{video_stem}_causal_review.json"
    _write_json(out_path, doc)
    return out_path


def load_causal_review(video_stem: str, output_dir: Path):
    """Load a saved per-video causal review, if present.

    Returns ``(doc, {segment_num: record})``; ``(None, {})`` if no file. The
    per-segment records are exactly what ``_collect_answers`` produced, so they
    can be dropped straight back into the widget's ``_review_records`` to restore
    a prior review session (resume + re-populate the panel).
    """
    path = Path(output_dir) / f"{video_stem}_causal_review.json"
    if not path.exists():
        return None, {}
    doc = _read_json(path)
    by_seg: Dict[int, Any] = {}
    for rec in doc.get("segments", []):
        sn = rec.get("segment_num")
        if sn is not None:
            by_seg[int(sn)] = rec
    return doc, by_seg


# ---------------------------------------------------------------------------
# Review -> outcomes bridge: apply a reviewer's corrections onto the algo's
# per-segment outcomes so a correction made in the Causal Review tool actually
# flows FORWARD into kinematics (which reads outcome + causal_reach_id from the
# outcomes). This is the missing link between review and the data product.
# ---------------------------------------------------------------------------

_TOUCHED_OUTCOMES = ("retrieved", "displaced_sa", "displaced_outside")

# A reach a reviewer TYPES (the review window's "Undetected reach (enter frames
# manually)" row) is the same as a detected reach only when that detected reach
# covers at least this share of the typed frames. WHY one rule, in one place:
# the window used to swap a typed reach for any detected reach touching it by a
# single frame, while the results step matched by half the frames; a typed
# reach between the two rules was recorded as one reach and resolved to another
# (the causal reach was lost, or pointed to another reach). Both now call typed_reach_match.
TYPED_REACH_COVER = 0.5

# The longest a TYPED reach may be. A typed reach the detector missed is added
# to the reach list with exactly these frames and measured over them, so a typo
# in a frame number becomes a "reach" of hundreds of frames with nonsense
# kinematics. Real reaches, detected or typed, are short (a median of roughly
# a dozen frames, almost all under a hundred); a typed span of many hundreds of
# frames reads like a slip of the keys. 300 frames is far past any real reach
# and still catches that.
MAX_TYPED_REACH_FRAMES = 300


def typed_reach_match(start, end, reaches):
    """The detected reach a typed [start, end] reach IS, or None when no detected
    reach covers at least TYPED_REACH_COVER of its frames (then it is a reach
    the detector missed, and is kept with the frames the person typed).

    ``reaches`` are dicts with start/end frames under ``start``/``end`` (the
    review window's list) or ``start_frame``/``end_frame`` (reaches.json).
    Frames are inclusive. The largest overlap wins."""
    s, e = int(start), int(end)
    span = max(1, e - s + 1)
    best, best_ov = None, 0
    for r in reaches or []:
        rs = r.get("start", r.get("start_frame"))
        re_ = r.get("end", r.get("end_frame"))
        if rs is None or re_ is None:
            continue
        ov = min(e, int(re_)) - max(s, int(rs)) + 1
        if ov > best_ov:
            best, best_ov = r, ov
    if best is not None and best_ov >= TYPED_REACH_COVER * span:
        return best
    return None


def _span_of(rec):
    """The frames a reviewer actually looked at, or None."""
    sp = rec.get("segment_span")
    if not isinstance(sp, dict):
        return None
    a, b = sp.get("start"), sp.get("end")
    if a is None or b is None:
        return None
    a, b = int(a), int(b)
    return (a, b) if b >= a else (b, a)


def _seg_span(seg):
    """The frames a current segment covers, or None."""
    a, b = seg.get("start_frame"), seg.get("end_frame")
    if a is None or b is None:
        return None
    a, b = int(a), int(b)
    return (a, b) if b >= a else (b, a)


def _overlap(x, y):
    lo, hi = max(x[0], y[0]), min(x[1], y[1])
    return max(0, hi - lo + 1)


# A review is only re-attached to a segment that covers most of the frames the
# reviewer saw. Below this the two are different stretches of video and applying
# the judgement would be inventing one.
MIN_SPAN_OVERLAP = 0.5

# How much the best candidate must beat the runner-up by. Below this the review
# straddles two segments and there is no honest way to choose.
MIN_MARGIN = 0.15

# A span shorter than this many frames is not footage: the tray advance alone
# takes ~15 frames and the shortest real pellet trial runs hundreds. The only
# sources of shorter spans are the duplicate-boundary defect (a zero-length
# segment) and a review recorded against one. Such a span must never be a
# matching target OR a matchable review: overlap is normalised by the SHORTER
# span, so a 1-frame segment inside any review span scores 100%, ties the true
# target, and the tie then drops the human's answer (e.g. a segment whose
# start and end are the same frame).
MIN_REAL_SEGMENT_FRAMES = 5


def _too_short(span, human_segmentation: bool = False) -> bool:
    """A span no pellet presentation fits in (see MIN_REAL_SEGMENT_FRAMES); a
    person's own cuts are taken as made, however short."""
    return (span is not None and not human_segmentation
            and span[1] - span[0] + 1 < MIN_REAL_SEGMENT_FRAMES)


def _reach_outside(reach, span) -> bool:
    """The reach lies wholly outside the span."""
    return reach is not None and span is not None and (reach[1] < span[0] or reach[0] > span[1])


def answer_problem(rec, human_segmentation: bool = False) -> Optional[str]:
    """Why an answer cannot count for the segment it was given for, in words
    for the reviewer -- or None when it can.

    The span and reach-position tests are the SAME tests the matching applies
    (index_review_by_segment): an answer failing them is dropped there. Two
    tests are asked only here, when answering: a touched outcome needs its
    causal reach, and an ignore window may not cover that whole reach. A
    stored record of either shape is still matched; the results then use the
    reach assignment's causal reach for it (truth_resolver). WHY the review tool asks this at
    the moment of answering: until 2026-10-23 it recorded such an answer, the
    pipeline dropped it without a word, and the video came back asking the same
    question -- a reviewer could answer one segment again and again. Part of
    triage is checking the event happened to the pellet shown; when it did
    not, the answer is not saved and the video goes to deep review (the
    Escalate button), where the cuts are fixed."""
    if not _is_answered(rec) or ((rec or {}).get("answers") or {}).get("is_boundary"):
        return None
    span = _span_of(rec)
    if span is None:
        return None
    n = rec.get("segment_num")
    if _too_short(span, human_segmentation):
        return ("Segment %s is only %d frame(s) long (frames %d-%d): two cuts almost "
                "on top of each other, not a pellet presentation. No pellet can be "
                "answered here -- the cuts need fixing."
                % (n, span[1] - span[0] + 1, span[0], span[1]))
    cr = _decided_reach(rec)
    if _reach_outside(cr, span):
        return ("The reach you chose (frames %d-%d) is outside segment %s, which runs "
                "from frame %d to %d. As this video is cut, that reach cannot have "
                "touched this segment's pellet." % (cr[0], cr[1], n, span[0], span[1]))
    human = rec.get("human") or {}
    if human.get("outcome") in _TOUCHED_OUTCOMES and cr is None:
        # WHY: a touched pellet with no causal reach was saved as a complete
        # answer, and the results then shipped the pellet with no causal reach
        # at all (the person's empty pick blocked the reach assignment's).
        # Saying which reach touched it is part of answering a touched pellet.
        return ("Pellet %s is marked %s, but no reach is picked as the one that "
                "touched it. Pick that reach, or type its frames under 'Undetected "
                "reach'. If no reach touched the pellet, choose 'untouched', or "
                "mark the frames where something else moved it under 'Ignore "
                "windows'." % (rec.get("pellet_num") or n, human.get("outcome")))
    for w in (rec.get("answers") or {}).get("abnormal_ranges") or []:
        # WHY: an ignore window takes its frames out of the measurements, so
        # one over the whole causal reach leaves the pellet's own reach
        # unmeasured -- the two answers contradict each other.
        try:
            ws, we = int(w["start_frame"]), int(w["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if cr is not None and ws <= cr[0] and we >= cr[1]:
            return ("The ignore window at frames %d-%d covers the whole reach you "
                    "chose as the one that touched the pellet (frames %d-%d), so "
                    "that reach would not be measured at all. If something other "
                    "than a reach moved the pellet, leave the reach unpicked; "
                    "otherwise make the window cover only the frames that are not "
                    "the reach." % (ws, we, cr[0], cr[1]))
    typed = (human.get("causal_reach") or {}).get("typed")
    if typed and typed[1] - typed[0] + 1 > MAX_TYPED_REACH_FRAMES:
        return ("The reach you typed runs %d frames (frames %d-%d); no reach lasts that "
                "long. Check the two frame numbers and type them again."
                % (typed[1] - typed[0] + 1, typed[0], typed[1]))
    return None


# Each matching note (index_review_by_segment), in a reviewer's words, and
# which of its numbers are the frames the note is about -- so the triage banner
# can show a note only on the segment it concerns: a reviewer needs only the
# note for the segment being looked at. Words None: nothing to act on.
_PLAIN_NOTES = [
    (re.compile(r"^segment (\S+) answer \(frames (\d+)-(\d+)\) was given before a person set"),
     "earlier answer for frames {1}-{2} was given before the re-cut -- answer again", (1, 2)),
    (re.compile(r"^segment (\S+) answer \(frames not recorded\) was given before a person set"),
     "an earlier answer was given before the re-cut -- answer again", None),
    (re.compile(r"^segment (\S+) review names causal reach (\d+)-(\d+), outside the frames "
                r"it records \((\d+)-(\d+)\)"),
     "earlier answer's reach {1}-{2} is outside this segment", (3, 4)),
    (re.compile(r"^current segment (\S+) \(frames (\d+)-(\d+)\) is shorter than any real"),
     "too short to be a pellet -- Escalate", (1, 2)),
    (re.compile(r"^segment (\S+) review covers frames .* shorter than any real footage"),
     None, None),
    (re.compile(r"^segment (\S+) review \(frames (\d+)-(\d+)\) matches no current segment"),
     "earlier answer for frames {1}-{2} (video re-cut since)", (1, 2)),
    (re.compile(r"^segment (\S+) review \(frames (\d+)-(\d+)\) splits across current "
                r"segments (\d+) and (\d+)"),
     "earlier answer for frames {1}-{2} (video re-cut since)", (1, 2)),
    (re.compile(r"^segment (\d+): two answers for frames (\d+)-(\d+) disagree"),
     "two earlier answers disagree -- answer again", (1, 2)),
    (re.compile(r"^review of segment (\S+) now applies to segment (\d+)"),
     None, None),
    (re.compile(r"^segment (\S+) review \(frames (\d+)-(\d+)\) overlaps current segment "
                r"(\d+) less than another review"),
     None, None),
]


def plain_review_note(note: str, span=None) -> Optional[str]:
    """A matching note (index_review_by_segment) in words for the reviewer, or
    None when it says nothing they need to act on. With ``span`` (start, end),
    also None unless the note is about frames inside that span. A note in no
    known form is returned as written."""
    for rx, words, about in _PLAIN_NOTES:
        m = rx.match(note or "")
        if not m:
            continue
        if not words:
            return None
        if span is not None and about is not None:
            g = m.groups()
            a, b = int(g[about[0]]), int(g[about[1]])
            if b < span[0] or a > span[1]:
                return None
        return words.format(*m.groups())
    return note


def pellet_span(boundaries, i) -> Tuple[int, int]:
    """THE frames of pellet ``i + 1``: from cut ``i`` to the frame before cut
    ``i + 1``, both ends included.

    The one definition of where a pellet starts and ends. Every reader that
    matches a person's answer to a pellet builds its pellets with it -- the
    review window (which stamps these frames on each answer it saves), the
    review gate, the release check, the results step (truth_resolver, through
    current_pellets) and the ground-truth readers.

    WHY one definition: the reach detector writes its own segment list into
    the reaches file with a different end -- a segment there ends AT the next
    cut, and it adds a last segment from the last cut to the end of the video
    (reach_detector.py). The results step used to match answers against that
    list while the window and the gate used this one, so the two disagreed by
    one frame on every pellet; under the re-cut rule (answer_survives_recut)
    the gate passed a video as answered while the results step dropped every
    one of its answers and filed the algorithm's calls, with only a log line.
    The reaches file's list is still where each pellet's detected reaches are
    found (by number); never where a pellet's frames come from."""
    return int(boundaries[i]), int(boundaries[i + 1]) - 1


def _cut_frames(seg_doc) -> Optional[List[int]]:
    """The cut frames of a ``*_segments.json`` document, or None when one of
    them cannot be read."""
    bounds = []
    for b in (seg_doc or {}).get("boundaries") or []:
        try:
            bounds.append(int(b.get("frame", b.get("index", 0))) if isinstance(b, dict) else int(b))
        except (TypeError, ValueError):
            return None
    return bounds


def segments_from_segmentation(seg_doc) -> List[Dict[str, Any]]:
    """The current pellets ``[{segment_num, start_frame, end_frame}]`` from a
    ``*_segments.json`` document's boundaries, framed by pellet_span (the one
    definition). Each carries ``cuts_set_at`` when a person set these cuts.
    Empty when the document has no usable boundaries."""
    bounds = _cut_frames(seg_doc)
    if not bounds or len(bounds) < 2:
        return []
    when = cuts_set_at(seg_doc)
    segs = []
    for i in range(len(bounds) - 1):
        start, end = pellet_span(bounds, i)
        segs.append({"segment_num": i + 1, "start_frame": start, "end_frame": end})
    if when:
        for s in segs:
            s[CUTS_SET_AT_KEY] = when
    return segs


def current_pellets(seg_doc, reaches_doc=None) -> List[Dict[str, Any]]:
    """The current pellets for the results step: segments_from_segmentation's
    frames (pellet_span), each with the reach detector's ``reaches`` of the
    same-numbered segment of ``reaches_doc`` attached (a causal pick is placed
    on a reach by frames, truth_resolver). The reaches file's own segment
    frames are NOT used (pellet_span says why). Empty when the segments
    document has no usable cuts: the caller then has no person's cuts and no
    re-cut to honour."""
    pellets = segments_from_segmentation(seg_doc)
    if not pellets:
        return []
    reaches_by_num = {}
    for s in (reaches_doc or {}).get("segments") or []:
        if isinstance(s, dict) and s.get("segment_num") is not None:
            try:
                reaches_by_num[int(s["segment_num"])] = s.get("reaches") or []
            except (TypeError, ValueError):
                continue
    for p in pellets:
        p["reaches"] = reaches_by_num.get(p["segment_num"], [])
    return pellets


def segmentation_is_human(seg_doc) -> bool:
    """True when a person made (or corrected) this segmentation."""
    return (seg_doc or {}).get("boundary_source") == "human"


# Carried on each current segment (segments_from_segmentation, and every
# caller that builds its own list of current segments): when a person set the
# current cuts. index_review_by_segment uses it to void answers given before.
CUTS_SET_AT_KEY = "cuts_set_at"


def cuts_set_at(seg_doc) -> Optional[str]:
    """When a person saved the current cuts (the fixer's ``corrected_at``), or
    None for the segmenter's own cuts or a file that does not record it."""
    if not segmentation_is_human(seg_doc):
        return None
    return (seg_doc or {}).get("corrected_at") or None


def recut_time(current_segments) -> Optional[datetime]:
    """When a person set the current cuts (the newest ``cuts_set_at`` the
    current pellets carry), or None for the segmenter's own cuts."""
    return max((t for t in (_when((s or {}).get(CUTS_SET_AT_KEY))
                            for s in current_segments or [])
                if t is not None), default=None)


def answer_survives_recut(rec, review_doc, recut) -> bool:
    """Does a person's saved answer still stand on the current cuts?

    THE re-cut rule, decided here and nowhere else: the review window, the
    review gate, the release check, the hand-score check and the results step
    all reach it through index_review_by_segment.

    The rule (owner ruling R2, 2026-11-15): re-segmentation voids everything
    after it, a person's earlier outcome and causal-reach answers included,
    and they are asked again. Never carry, re-home or restore an answer across
    a re-cut, by frames or by number; answers persist only through re-runs on
    the SAME cuts. So once a person has set the current cuts (``recut``, from
    the segments file's ``corrected_at``), an answer given before then does
    not stand -- on any pellet, including one whose frames happen to be the
    same: the person judged it as part of a cutting that turned out wrong.
    WHY not "unchanged pellets keep theirs" (built 2026-11-18 in 0bd388d):
    it decided survival by comparing frames, and the results step and the
    gate framed pellets differently by one frame (pellet_span), so the gate
    kept answers the results step then dropped, unasked.

    True: no person's re-cut (``recut`` None), or the answer was given at or
    after it. An answer whose time cannot be read counts as given before."""
    if recut is None:
        return True
    given = _when((rec or {}).get("reviewed_at") or (review_doc or {}).get("reviewed_at"))
    return given is not None and given >= recut


def answered_segments(review_doc, current_segments=None,
                      human_segmentation: bool = False) -> set:
    """CURRENT segment numbers a person has answered: their review's frames
    match the segment (index_review_by_segment), and the record is an answer
    (``is_answered``: not a ``reviewed: False`` placeholder, not a pellet left
    undecided).

    WHY this is the one definition: the release check, the review gate and the
    queue views counted an answer by the segment NUMBER it was written against,
    while the results step matches it by FRAMES. After a person re-cut a video,
    an old answer for "segment 7" -- about other footage -- counted as answering
    today's segment 7: the video left review, the results step correctly ignored
    the answer, and today's segment 7 reached the data with nobody having
    watched it. A nightly check then sent the same videos back, every night.
    Without ``current_segments`` this falls back to numbers, the
    only handle a video without a segmentation file has.
    """
    mapping, _ = index_review_by_segment(review_doc, current_segments,
                                         human_segmentation=human_segmentation)
    return {int(sn) for sn, rec in mapping.items() if is_answered(rec)}


def index_review_by_segment(review_doc, current_segments=None,
                            human_segmentation: bool = False,
                            voided: Optional[list] = None):
    """Map each CURRENT segment number to the review record describing it.

    Reviews used to be found by segment number alone. That number is an index
    the segmenter hands out fresh on every re-cut, so after a re-segmentation
    the review of "segment 7" was applied to whatever the new segment 7 happened
    to be -- a human judgement stamped onto footage nobody looked at, recorded as
    outcome_source "human_review" and indistinguishable from a real one.

    Every record written since review schema 1.1 carries ``segment_span``: the
    frames the reviewer was actually shown. When both sides carry frames this
    matches on overlap and ignores the numbers entirely. A review whose frames
    no longer correspond to any segment is DROPPED rather than guessed at, and
    said so in the returned notes.

    Records with no span -- written before 1.1, or ones whose original
    segmentation could not be recovered -- fall back to matching by number,
    which is all that has ever been available for them.

    A RE-CUT VOIDS THE ANSWERS GIVEN BEFORE IT (answer_survives_recut, the
    one place the rule is decided). When a person set the current cuts (a
    current segment carries ``cuts_set_at``, from the segments file's
    ``corrected_at``), no answer given before that time is applied, whatever
    its frames; the pellet is asked again (recut_voided_segments) and the
    answer stays in the file as a record (restore_review_for_segments keeps it
    under ``unmatched_answers``). An answer given after the cuts were set is
    matched as below.

    A segment shorter than MIN_REAL_SEGMENT_FRAMES is treated as a segmenter
    artifact -- unless ``human_segmentation`` is set: a person who cut the video
    decided where its segments are, and their cuts are logged as made, however
    short: if a human segmented the video, that is what the algorithm logs,
    even when the range is short.

    Args:
        review_doc: a loaded causal review document
        current_segments: the segments of the CURRENT segmentation, each with
            start_frame / end_frame. When absent, behaviour is the old
            number-matching, so callers that have no segmentation still work.
        human_segmentation: the current segmentation was made by a person
            (``boundary_source == "human"`` in its segments file).
        voided: a list; when given, every record a re-cut voided is appended
            to it (recut_voided_segments uses them to ask those pellets again).

    Returns:
        (mapping, notes) -- mapping is {current segment_num: review record};
        notes is a list of human-readable strings about anything that moved or
        was dropped, for logging.
    """
    notes = []
    records = [r for r in (review_doc or {}).get("segments", []) if isinstance(r, dict)]

    by_num = {}
    for rec in records:
        sn = rec.get("segment_num")
        if sn is not None:
            by_num[int(sn)] = rec

    if not current_segments:
        return by_num, notes

    spans = {}
    for seg in current_segments:
        sn, sp = seg.get("segment_num"), _seg_span(seg)
        if sn is None or sp is None:
            continue
        if _too_short(sp, human_segmentation):
            notes.append(
                "current segment %s (frames %d-%d) is shorter than any real "
                "footage; excluded as a review-matching target"
                % (sn, sp[0], sp[1]))
            continue
        spans[int(sn)] = sp
    if not spans:
        return by_num, notes

    recut = recut_time(current_segments)

    mapping = {}
    claimed = {}
    conflicted = {}
    for rec in records:
        rspan = _span_of(rec)
        old_num = rec.get("segment_num")
        if _too_short(rspan, human_segmentation):
            notes.append(
                "segment %s review covers frames %d-%d, shorter than any real "
                "footage (a zero-length-segment artifact); dropped"
                % (old_num, rspan[0], rspan[1]))
            continue
        if not answer_survives_recut(rec, review_doc, recut):
            notes.append(
                "segment %s answer (frames %s) was given before a person set the "
                "current cuts (%s); a re-cut voids it -- not applied, the pellet "
                "is unanswered again"
                % (old_num, "%d-%d" % rspan if rspan else "not recorded",
                   recut.isoformat(timespec="seconds")))
            if voided is not None:
                voided.append(rec)
            continue
        if rspan is None:
            # No frames recorded; the number is the only handle there is.
            if old_num is not None and int(old_num) in spans:
                mapping.setdefault(int(old_num), rec)
            continue

        # An answer whose causal reach lies wholly outside the frames it records
        # is about other footage: the review tool once put an old answer back on
        # the wrong segment and the next save stamped that segment's frames on
        # it (typically shifted by one segment). Applied, it would make a neighbouring segment's reach this
        # segment's causal reach. Not guessed at: dropped, and the segment goes
        # back to a person.
        cr = _decided_reach(rec)
        if _reach_outside(cr, rspan):
            notes.append(
                "segment %s review names causal reach %d-%d, outside the frames it "
                "records (%d-%d); not applied -- it is about other footage"
                % (old_num, cr[0], cr[1], rspan[0], rspan[1]))
            continue

        scored = []
        for sn, sspan in spans.items():
            ov = _overlap(rspan, sspan)
            if not ov:
                continue
            frac = ov / max(1, min(rspan[1] - rspan[0] + 1, sspan[1] - sspan[0] + 1))
            scored.append((frac, sn))
        scored.sort(reverse=True)
        best_frac, best_num = scored[0] if scored else (0.0, None)

        # A re-cut can split the reviewed stretch across two new segments, and
        # then "best" is a coin flip. Picking one silently is the same quiet
        # guess this matching exists to stop, so an ambiguous match is dropped.
        if len(scored) > 1 and (scored[0][0] - scored[1][0]) < MIN_MARGIN:
            notes.append(
                "segment %s review (frames %d-%d) splits across current segments "
                "%d and %d (%.0f%% vs %.0f%%); dropped rather than assigned to "
                "one of them"
                % (old_num, rspan[0], rspan[1], scored[0][1], scored[1][1],
                   scored[0][0] * 100, scored[1][0] * 100))
            continue

        if best_num is None or best_frac < MIN_SPAN_OVERLAP:
            notes.append(
                "segment %s review (frames %d-%d) matches no current segment; "
                "dropped rather than applied to different footage"
                % (old_num, rspan[0], rspan[1]))
            continue

        # Two records for the SAME frames: the review tool used to restore a
        # saved answer by segment number, so after a re-cut it filed an old
        # answer under the wrong number and the person's new answer for those
        # frames beside it. File order is not time order,
        # so "first wins" could apply the older answer -- or a left-blank
        # 'triaged' record over the person's real one. A decision beats a
        # placeholder or a blank; two decisions that agree are one; two that
        # DISAGREE are neither, and the segment goes back to a person.
        held = mapping.get(best_num)
        if held is not None and _span_of(held) == rspan:
            if not _is_decided(rec):
                continue
            if not _is_decided(held):
                mapping[best_num] = rec
                continue
            if _applied_call(held) != _applied_call(rec):
                conflicted[best_num] = (old_num, held.get("segment_num"), rspan)
            continue

        prev = claimed.get(best_num)
        if prev is not None and prev >= best_frac:
            notes.append(
                "segment %s review (frames %d-%d) overlaps current segment %d "
                "less than another review; dropped"
                % (old_num, rspan[0], rspan[1], best_num))
            continue

        mapping[best_num] = rec
        claimed[best_num] = best_frac
        if old_num is not None and int(old_num) != best_num:
            notes.append(
                "review of segment %s now applies to segment %d "
                "(frames %d-%d, %.0f%% overlap) -- the segmentation changed"
                % (old_num, best_num, rspan[0], rspan[1], best_frac * 100))

    for sn, (a, b, rspan) in sorted(conflicted.items()):
        mapping.pop(sn, None)
        notes.append(
            "segment %d: two answers for frames %d-%d disagree (saved as segments "
            "%s and %s); neither is applied -- a person must answer it again"
            % (sn, rspan[0], rspan[1], b, a))
    return mapping, notes


def recut_voided_segments(review_doc, current_segments=None,
                          human_segmentation: bool = False) -> set:
    """CURRENT segment numbers whose person's answer a re-cut voided and that
    nobody has answered since: the pellets that must be ASKED AGAIN.

    index_review_by_segment voids an answer given before a person set the
    current cuts (answer_survives_recut). Voiding alone only
    makes the pellet unanswered, and an unanswered pellet the algorithm did not
    flag takes the algorithm's call -- so a person's answer would silently be
    replaced by the very call they had overruled. This names those pellets so
    the review gate sends them to triage (review_gate.evaluate_gate) and a
    person answers them on the new cuts.

    Which current pellet a voided answer is about: the one its frames overlap
    most (ties: each of them); an answer that recorded no frames, the current
    pellet with its number. Asking one pellet too many costs a person a look;
    asking one too few files the algorithm's call over a person's answer.
    Only an ANSWER counts (is_answered): a voided placeholder asks nothing.
    The answers a save kept aside (``unmatched_answers``) are read too: a save
    after the re-cut may keep a voided answer there, and it must still be
    asked. Empty when the current cuts are the segmenter's, or nothing was
    voided."""
    if not current_segments:
        return set()
    voided: list = []
    mapping, _ = index_review_by_segment(review_doc, current_segments,
                                         human_segmentation=human_segmentation,
                                         voided=voided)
    aside = [r for r in (review_doc or {}).get("unmatched_answers") or []
             if isinstance(r, dict)]
    if aside:
        # Only what the re-cut voids among them; an aside answer that is not
        # void was never applied by any reader and is not asked about here.
        index_review_by_segment({"segments": aside,
                                 "reviewed_at": (review_doc or {}).get("reviewed_at")},
                                current_segments, human_segmentation=human_segmentation,
                                voided=voided)
    if not voided:
        return set()
    answered = {int(sn) for sn, rec in mapping.items() if is_answered(rec)}
    spans = {}
    for seg in current_segments:
        sn, sp = seg.get("segment_num"), _seg_span(seg)
        if sn is not None and sp is not None and not _too_short(sp, human_segmentation):
            spans[int(sn)] = sp
    out = set()
    for rec in voided:
        if not is_answered(rec):
            continue
        rspan = _span_of(rec)
        if rspan is None:
            num = rec.get("segment_num")
            try:
                if num is not None and int(num) in spans:
                    out.add(int(num))
            except (TypeError, ValueError):
                pass
            continue
        scored = [(_overlap(rspan, sp), sn) for sn, sp in spans.items()]
        best = max((ov for ov, _ in scored), default=0)
        if best:
            out |= {sn for ov, sn in scored if ov == best}
    return out - answered


# ---------------------------------------------------------------------------
# Ground truth matched to the CURRENT cuts by frames
# ---------------------------------------------------------------------------

def gt_segments(gt) -> List[Dict[str, Any]]:
    """The segments a ground-truth file was made on, ``[{segment_num,
    start_frame, end_frame}]``, from its own ``segmentation.boundaries``
    (pellet_span, the rule every other reader uses). Empty when the file records no usable
    boundaries.

    Each boundary is placed by its own ``index`` (0 = the first cut), not by
    its position in the list: segment N exists only when the boundaries with
    index N-1 and N are both in the file. WHY: "Save as Ground Truth" writes
    only the boundaries a person determined, so a file can hold cuts 3, 4 and
    9; numbered by position they made "segment 1" of cuts 3-4 and "segment 2"
    of cuts 4-9, and the outcomes of pellets 1 and 2 were matched to the
    footage of pellets 4 and 5-9. A segment whose two cuts are not both in
    the file has no frames here; its outcome then falls back to its number
    (index_gt_outcomes_by_segment). A file whose boundaries carry no index
    (an older format) is numbered by position, as before."""
    bounds = ((gt or {}).get("segmentation") or {}).get("boundaries") or []
    by_index = {}
    for b in bounds:
        if not isinstance(b, dict) or b.get("index") is None or b.get("frame") is None:
            return segments_from_segmentation((gt or {}).get("segmentation") or {})
        try:
            by_index[int(b["index"])] = int(b["frame"])
        except (TypeError, ValueError):
            return []
    out = []
    for i in sorted(by_index):
        if i + 1 in by_index:
            start, end = pellet_span(by_index, i)
            out.append({"segment_num": i + 1, "start_frame": start, "end_frame": end})
    return out


def index_gt_outcomes_by_segment(gt, current_segments=None,
                                 human_segmentation: bool = False):
    """Map each CURRENT segment number to the ground-truth outcome entry about
    its frames. Returns ``(mapping, notes)``.

    WHY by frames: ground truth was applied by segment NUMBER, while every
    review answer is matched by the frames it was given for
    (index_review_by_segment). After a re-cut renumbers the segments, the
    ground truth's "segment 7" is other footage than today's segment 7, and
    its outcome would be stamped on a pellet nobody judged. Each outcome entry
    takes the frames of its segment in the ground truth's OWN segmentation and
    goes through the same matching as a review answer: an entry whose frames
    match no current segment, or split across two, is dropped and said so.
    A file without its own segmentation, or a call without
    ``current_segments``, falls back to the number -- the only handle there is.
    """
    spans = {s["segment_num"]: s for s in gt_segments(gt)}
    records = []
    for seg in (((gt or {}).get("outcomes") or {}).get("segments") or []):
        sn = seg.get("segment_num") if isinstance(seg, dict) else None
        if sn is None:
            continue
        own = spans.get(int(sn))
        records.append({
            "segment_num": int(sn),
            "segment_span": ({"start": own["start_frame"], "end": own["end_frame"]}
                             if own else None),
            "human": {"outcome": seg.get("outcome")},
            "answers": {},
            "_gt": seg,
        })
    mapping, notes = index_review_by_segment(
        {"segments": records}, current_segments if spans else None,
        human_segmentation=human_segmentation)
    return {int(sn): rec["_gt"] for sn, rec in mapping.items()}, \
        ["ground truth " + n for n in notes]


# Outcomes that are not a decision: the algorithm's own "triaged", an older
# tool's "uncertain", and none at all.
UNDECIDED_OUTCOMES = (None, "triaged", "uncertain")


def is_answered(rec) -> bool:
    """True when a saved review record is a person's ANSWER for its segment.

    Not answered: a record marked ``reviewed: False`` (a segment the reviewer
    never reached), and a pellet record whose outcome is still undecided. The
    review window saves a pellet whose outcome question was left on
    "-- select outcome --" with the algorithm's "triaged" still in it; until
    2026-10-26 that counted as an answer, so a video with a pellet nobody
    decided left triage and was FILED (reaching the exports as
    "Undetermined"). A
    boundary record carries no pellet and needs no outcome.

    The ONE definition: every check of whether a person answered a segment --
    the release from triage, the review gate, the hand-score check, routing,
    restoring answers in the review window -- comes here."""
    rec = rec or {}
    answers = rec.get("answers") or {}
    if answers.get("reviewed") is False:
        return False
    if answers.get("is_boundary"):
        return True
    return (rec.get("human") or {}).get("outcome") not in UNDECIDED_OUTCOMES


_is_answered = is_answered


def _decided_reach(rec):
    """(start, end) of a decided touched outcome's causal reach, else None."""
    human = (rec or {}).get("human") or {}
    hc = human.get("causal_reach")
    if (not _is_answered(rec) or human.get("outcome") not in _TOUCHED_OUTCOMES
            or not isinstance(hc, dict)):
        return None
    try:
        start = int(hc["start"])
        return start, int(hc.get("end", start))
    except (KeyError, TypeError, ValueError):
        return None


def _is_decided(rec) -> bool:
    """Answered AND carrying an outcome: a reviewer who leaves an element open
    saves it as 'triaged', which is not a decision about the footage."""
    ho = ((rec or {}).get("human") or {}).get("outcome")
    return _is_answered(rec) and ho not in (None, "triaged")


def _applied_call(rec):
    """What apply_review_overrides would take from a record -- the parts two
    answers must share to be the same answer."""
    human = (rec or {}).get("human") or {}
    ho = human.get("outcome")
    hc = human.get("causal_reach") or {}
    reach = None
    if ho in _TOUCHED_OUTCOMES and isinstance(hc, dict):
        reach = (hc.get("reach_id"), hc.get("start"), hc.get("end"))
    ranges = ((rec or {}).get("answers") or {}).get("abnormal_ranges") or []
    return (ho, reach, json.dumps(ranges, sort_keys=True, default=str))


# Earlier answers whose frames match no current segment. A re-cut must never
# delete what a person decided, and applying them to other footage would be
# worse, so they ride along in the file under this key and nothing applies them.
UNMATCHED_ANSWERS_KEY = "unmatched_answers"


def restore_review_for_segments(review_doc, current_segments,
                                human_segmentation: bool = False):
    """What the review tool restores when it reopens a video.

    Returns ``(restored, unmatched, notes)``: ``restored`` maps each CURRENT
    segment number to the saved answer whose frames it covers (renumbered to
    the current number, frames kept as reviewed); ``unmatched`` holds answered
    records that match no current segment, plus any carried from before.

    WHY by frames: the tool restored by segment number. After a re-cut, an
    answer about other footage sat on today's segment N, the tool counted N as
    done and never showed it, while the release check -- which matches by
    frames -- counted N unanswered and kept the video held: a video nobody
    could finish. Both now use index_review_by_segment.
    """
    mapping, notes = index_review_by_segment(
        review_doc, current_segments, human_segmentation=human_segmentation)
    restored = {}
    for sn, rec in mapping.items():
        if not _is_answered(rec):
            continue
        r = dict(rec)
        r["segment_num"] = int(sn)
        restored[int(sn)] = r
    used = {id(r) for r in mapping.values()}
    records = [r for r in (review_doc or {}).get("segments", []) if isinstance(r, dict)]
    taken = [r for r in records if id(r) in used]
    unmatched = []
    for r in records:
        if id(r) in used or not _is_answered(r):
            continue
        # An exact copy of an answer that was restored adds nothing.
        if any(_span_of(r) == _span_of(k) and _applied_call(r) == _applied_call(k)
               for k in taken):
            continue
        unmatched.append(r)
    unmatched.extend((review_doc or {}).get(UNMATCHED_ANSWERS_KEY) or [])
    return restored, unmatched, notes


# ---------------------------------------------------------------------------
# Several copies of one review: the newest statement per pellet
# ---------------------------------------------------------------------------
# A video's review exists in more than one place: the durable copy, the copy
# in the bundle being reviewed, a copy beside the video. They can differ: a
# copy can be older (written before a later save elsewhere), and a recheck
# ("are you sure?") reopens a pellet in one copy only. The review window used
# to read the FIRST copy that had any records and never the durable one, and
# a save wrote only what the window held over every copy. When the window
# opened a bundle without a working copy, the reviewer answered what was open
# and the save overwrote the durable copy's earlier answers with "not
# reviewed" placeholders: answers a person gave were gone from every live
# copy. So both the window and the save now read every copy and, for each
# current segment, take the NEWEST statement a person made about its frames.

def _when(value) -> Optional[datetime]:
    """An ISO time as a comparable local time, or None when unreadable."""
    try:
        t = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is not None:
        t = t.astimezone().replace(tzinfo=None)
    return t


def _statement(rec, doc) -> Optional[tuple]:
    """("answer" | "recheck", when) for a record that is a person's statement
    about its pellet, else None. An answer (is_answered) was given at its own
    ``reviewed_at`` (the file's for records saved before records carried one).
    A recheck -- a pellet sent back with "are you sure?" -- is a statement too:
    it reopened an answer at ``answers.recheck.asked_at``. A placeholder or a
    pellet left undecided says nothing and never replaces an answer."""
    if is_answered(rec):
        return "answer", _when(segment_reviewer(rec, doc)[1])
    recheck = ((rec or {}).get("answers") or {}).get("recheck")
    if isinstance(recheck, dict):
        return "recheck", _when(recheck.get("asked_at"))
    return None


def _with_own_reviewer(rec, doc) -> Dict[str, Any]:
    """A copy of ``rec`` carrying its own reviewer and time, so they survive
    being moved out of the file that held them at file level."""
    out = dict(rec)
    if is_answered(rec):
        who, when = segment_reviewer(rec, doc)
        if who and not out.get("reviewer"):
            out["reviewer"] = who
        if when and not out.get("reviewed_at"):
            out["reviewed_at"] = when
    return out


def newest_statements(docs, current_segments=None, human_segmentation: bool = False):
    """For each CURRENT segment, the newest statement a person made about its
    frames in ANY of ``docs`` (several copies of one video's review).

    Returns ``(chosen, unmatched, notes)``: ``chosen`` maps a current segment
    number to the newest answer or recheck matched to it by frames
    (index_review_by_segment, the one matching rule); ``unmatched`` holds the
    answers that match no current segment, from every copy, each once, plus
    every copy's carried ``unmatched_answers``. A newer answer replaces an
    older one, a newer recheck reopens an older answer, a newer answer ends an
    older recheck. Placeholders and undecided records are never chosen: they
    say nothing, so they can never replace an answer. On equal times the copy
    earlier in ``docs`` wins; a time that cannot be read counts as oldest.

    ``docs`` without ``current_segments`` match by segment number."""
    best: Dict[int, tuple] = {}
    unmatched: List[Dict[str, Any]] = []
    notes: List[str] = []
    seen_unmatched = set()

    def keep_unmatched(rec):
        key = (json.dumps(rec.get("segment_span"), sort_keys=True, default=str),
               json.dumps(_applied_call(rec), sort_keys=True, default=str))
        if key not in seen_unmatched:
            seen_unmatched.add(key)
            unmatched.append(rec)

    for rank, doc in enumerate(d for d in (docs or []) if isinstance(d, dict)):
        stated = [r for r in doc.get("segments") or []
                  if isinstance(r, dict) and _statement(r, doc) is not None]
        mapping, n = index_review_by_segment(
            {"segments": stated}, current_segments, human_segmentation=human_segmentation)
        notes.extend(n)
        used = set()
        for sn, rec in mapping.items():
            used.add(id(rec))
            kind, when = _statement(rec, doc)
            order = (when or datetime.min, -rank)
            held = best.get(int(sn))
            if held is None or order > held[0]:
                best[int(sn)] = (order, _with_own_reviewer(rec, doc))
        for rec in stated:
            if id(rec) not in used and is_answered(rec):
                keep_unmatched(_with_own_reviewer(rec, doc))
        for rec in doc.get(UNMATCHED_ANSWERS_KEY) or []:
            if isinstance(rec, dict):
                keep_unmatched(rec)
    chosen = {sn: rec for sn, (_, rec) in best.items()}
    # An answer that lost to a newer statement about the same frames in one
    # copy can be "unmatched" in another copy only by accident of numbering;
    # drop any unmatched entry that IS a chosen record.
    chosen_keys = {(json.dumps(r.get("segment_span"), sort_keys=True, default=str),
                    json.dumps(_applied_call(r), sort_keys=True, default=str))
                   for r in chosen.values()}
    unmatched = [r for r in unmatched
                 if (json.dumps(r.get("segment_span"), sort_keys=True, default=str),
                     json.dumps(_applied_call(r), sort_keys=True, default=str))
                 not in chosen_keys]
    return chosen, unmatched, notes


def merge_review_copies(docs, current_segments=None, human_segmentation: bool = False):
    """One review document built from several copies of the same video's
    review (newest_statements says which record wins). Returns
    ``(doc, notes)``; ``doc`` is None when no copy was given. The result
    holds only statements (answers and rechecks) and the carried unmatched
    answers -- what restoring, releasing and counting need."""
    docs = [d for d in (docs or []) if isinstance(d, dict)]
    if not docs:
        return None, []
    chosen, unmatched, notes = newest_statements(docs, current_segments, human_segmentation)
    merged: Dict[str, Any] = {k: v for k, v in docs[0].items()
                              if k not in ("segments", UNMATCHED_ANSWERS_KEY)}
    merged["segments"] = [chosen[sn] for sn in sorted(chosen)]
    if unmatched:
        merged[UNMATCHED_ANSWERS_KEY] = unmatched
    return merged, notes


def merge_review_into(doc, others):
    """``doc`` (one copy of a video's review) with every segment record
    replaced by the newest statement -- answer or recheck -- that ``doc`` or
    any of ``others`` holds about that record's frames (newest_statements),
    and every copy's unmatched answers carried. Records no copy has a
    statement for (placeholders) stay as ``doc`` has them, and ``doc``'s other
    fields are kept, so the result is still a complete copy of the review.

    WHY: a machine that keeps a second copy of a review (the durable copy made
    when a bundle leaves its queue, the copy put back into a re-staged bundle)
    used to REPLACE the older-looking file with the newer one by file time. An
    "are you sure?" recheck, or an answer, that only the other copy held was
    lost. Merging statement by statement keeps the newest word a person gave
    about every pellet, whichever copy it is in.

    NON-STATEMENT FIELDS ARE KEPT TOO: a record's other fields -- above all a
    "this segment is really pellet N" record (``segmentation_wrong`` with
    ``true_segment_num``), which is usually saved with the pellet left open and
    so is no statement -- are filled into the result's record for the same
    frames from whichever copy holds them, when the chosen record has no value
    of its own; a record about frames the result has no record for at all is
    added. Without this a mislabel only one copy held was dropped by the merge,
    and the release check and the hand-score check act on exactly those
    records."""
    docs = [d for d in [doc] + list(others or []) if isinstance(d, dict)]
    if not docs:
        return None
    base = docs[0]
    recs = [r for r in (base.get("segments") or []) if isinstance(r, dict)]
    current = []
    for rec in recs:
        sp = _span_of(rec)
        if sp is not None and rec.get("segment_num") is not None:
            current.append({"segment_num": int(rec["segment_num"]),
                            "start_frame": sp[0], "end_frame": sp[1]})
    if len(current) != len(recs):
        current = None
    chosen, unmatched, _ = newest_statements(docs, current, human_segmentation=True)
    out = dict(base)
    segs, used = [], set()
    for rec in recs:
        sn = rec.get("segment_num")
        pick = chosen.get(int(sn)) if sn is not None else None
        if pick is not None:
            pick = dict(pick)
            pick["segment_num"] = int(sn)
            used.add(int(sn))
            segs.append(pick)
        else:
            segs.append(rec)
    # A statement about a segment the base copy has no record for at all.
    for sn in sorted(set(chosen) - used):
        segs.append(chosen[sn])
    _carry_record_fields(segs, docs)
    out["segments"] = segs
    if unmatched:
        out[UNMATCHED_ANSWERS_KEY] = unmatched
    else:
        out.pop(UNMATCHED_ANSWERS_KEY, None)
    return out


# The fields of a review record that make up a person's STATEMENT (the answer
# or recheck that merge_review_into chooses per pellet) and the record's
# identity. Every other field is filled across copies by _carry_record_fields.
_STATEMENT_FIELDS = frozenset({"segment_num", "segment_span", "human", "answers",
                               "reviewer", "reviewed_at"})


def _same_frames(a, b) -> bool:
    """Do two review records describe the same segment? By frames when both
    carry them (segment numbers can differ between two cuts), else by number."""
    sa, sb = _span_of(a), _span_of(b)
    if sa is not None and sb is not None:
        return sa == sb
    return (a.get("segment_num") is not None
            and a.get("segment_num") == b.get("segment_num"))


def _record_when(rec, doc) -> datetime:
    """When ``rec`` (a record of copy ``doc``) was written, for deciding which
    of two records about the same frames is the newer word. A statement's own
    time (_statement: an answer's ``reviewed_at``, a recheck's ``asked_at``);
    any other record's own ``reviewed_at``, else its copy's -- the save that
    wrote it. A time that cannot be read counts as oldest."""
    st = _statement(rec, doc or {})
    if st is not None:
        return st[1] or datetime.min
    when = (rec or {}).get("reviewed_at") or (doc or {}).get("reviewed_at")
    return _when(when) or datetime.min


def _is_mislabel(rec) -> bool:
    return (isinstance(rec, dict) and bool(rec.get("segmentation_wrong"))
            and rec.get("true_segment_num") is not None)


def _mislabel_superseded(rec, doc, docs) -> bool:
    """True when some copy in ``docs`` holds a NEWER statement (answer or
    recheck) about the same frames as the mislabel record ``rec`` that does
    not repeat the mislabel. WHY: a person's later word about those frames
    is their current view; an older "this is really pellet N" must not be
    brought back over it from a copy that happens to be older -- that sent
    the video to re-segmentation and steered the hand-score check on a call
    the person had since replaced. A later placeholder (pellet left open)
    says nothing about the pellet, so it never supersedes a mislabel."""
    when = _record_when(rec, doc)
    for d in docs:
        for other in (d or {}).get("segments") or []:
            if (isinstance(other, dict) and other is not rec
                    and _statement(other, d) is not None
                    and _same_frames(other, rec)
                    and not (_is_mislabel(other)
                             and other.get("true_segment_num") == rec.get("true_segment_num"))
                    and _record_when(other, d) > when):
                return True
    return False


def _carry_record_fields(segs, docs) -> None:
    """Fill, in place, every field of ``segs``' records that the record lacks
    (or holds as None) from another copy's record about the same frames, the
    copies read in order; then add any record that carries a mislabel
    (``segmentation_wrong`` + ``true_segment_num``) about frames none of
    ``segs`` describes. Statement fields are never touched: which answer counts
    is decided by newest_statements alone.

    RECENCY: a field is carried only from a record that is not older than the
    chosen one (_record_when), or onto a chosen record that is no statement at
    all (a placeholder says nothing to override); and a mislabel is never
    carried or added when a newer statement about its frames does not repeat
    it (_mislabel_superseded). Without this, a later answer that no longer
    says "this is really pellet N" had the older copy's mislabel grafted back
    and written to disk."""
    for d in docs:
        for other in d.get("segments") or []:
            if not isinstance(other, dict):
                continue
            if _is_mislabel(other) and _mislabel_superseded(other, d, docs):
                other = {k: v for k, v in other.items()
                         if k not in ("segmentation_wrong", "true_segment_num")}
            match = [r for r in segs if isinstance(r, dict) and _same_frames(r, other)]
            if not match:
                if _is_mislabel(other):
                    rec = dict(other)
                    rec.setdefault("reviewer", d.get("reviewer"))
                    rec.setdefault("reviewed_at", d.get("reviewed_at"))
                    segs.append(rec)
                continue
            other_when = _record_when(other, d)
            for i, r in enumerate(segs):
                if not any(r is m for m in match):
                    continue
                st = _statement(r, {})
                if st is not None and other_when < (st[1] or datetime.min):
                    continue   # an older record never fills a newer statement
                fill = {k: v for k, v in other.items()
                        if k not in _STATEMENT_FIELDS and v is not None and r.get(k) is None}
                if fill:
                    r = dict(r)
                    r.update(fill)
                    segs[i] = r


def mislabel_records(video_stem: str, dirs=()) -> Dict[str, Any]:
    """Every record, in EVERY copy of the review (review_copies), in which a
    person said the segment is really another pellet (``segmentation_wrong``
    with ``true_segment_num``) and no newer statement about the same frames
    replaced it (_mislabel_superseded), as one document ``{"segments": [...]}``.
    Each record carries its copy's reviewer and time when it has none of its
    own.

    WHY separate from effective_review: the merged review keeps only answers
    and rechecks (merge_review_copies), and a mislabel record is usually saved
    with the pellet left open -- so it is not a statement and the merge drops
    it. The checks that act on a mislabel (the release check's seg_corrected,
    the hand-score check) must still see it, from whichever copy holds it --
    but not after a person's later answer about those frames left it out."""
    copies = review_copies(video_stem, dirs)
    out: List[Dict[str, Any]] = []
    seen = set()
    for doc in copies:
        for rec in doc.get("segments") or []:
            if not _is_mislabel(rec) or _mislabel_superseded(rec, doc, copies):
                continue
            rec = dict(rec)
            rec.setdefault("reviewer", doc.get("reviewer"))
            rec.setdefault("reviewed_at", doc.get("reviewed_at"))
            key = json.dumps(rec, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(rec)
    return {"segments": out}


def review_copies(video_stem: str, dirs=(), include_durable: bool = True,
                  unreadable: Optional[list] = None,
                  strict: bool = False) -> List[Dict[str, Any]]:
    """Every readable copy of a video's review: ``{stem}_causal_review.json``
    in each of ``dirs`` (in that order), then the durable copy. A copy that
    cannot be read is logged and skipped -- never treated as "no answers"
    silently. When ``unreadable`` is a list, each such copy is also added to
    it as words for a person: the kinematics step refuses to write features
    while a copy of the person's answers cannot be read
    (truth_resolver.resolve_truth_layers), because the answers in it would be
    missing from the result.

    ``strict`` True RAISES instead of skipping, when a copy exists but cannot
    be read (cut off, not JSON, a read error on the share). WHY: a caller that
    acts on "this pellet is not answered" -- routing a video back to a person,
    reopening answers -- must not decide that from the copies it happened to
    be able to read: the unreadable copy may be the only one holding the
    person's answer, and skipping it would ask the person again."""
    paths: List[Path] = [Path(d) / f"{video_stem}_causal_review.json" for d in dirs if d]
    if include_durable:
        try:
            dp = durable_review_path(video_stem)
        except Exception:
            dp = None
        if dp is not None:
            paths.append(Path(dp))
    out, seen = [], set()
    for p in paths:
        key = os.path.normcase(str(p))
        if key in seen:
            continue
        seen.add(key)
        try:
            if p.is_file():
                out.append(_read_json(p))
        except Exception as err:
            if strict:
                raise
            logger.warning("Could not read the review copy %s: %s", p, err)
            if unreadable is not None:
                unreadable.append(f"the review copy {p} could not be read ({err})")
    return out


def review_dirs(video_stem: str, primary=None) -> List[Path]:
    """Every folder a copy of this video's review can sit in, besides the
    durable store (which review_copies always adds): ``primary`` (a folder, or
    the review file itself -- its folder is used), the triage and
    re-segmentation queue bundles, and the video's filed folder. Folders that
    do not exist are harmless (nothing is read from them). Never raises."""
    out: List[Path] = []
    if primary is not None:
        p = Path(primary)
        try:
            out.append(p.parent if p.is_file() else p)
        except OSError:
            out.append(p)
    try:
        from mousereach.config import Paths
        for q in (getattr(Paths, "TRIAGE_REVIEW", None), getattr(Paths, "DEEP_REVIEW", None)):
            if q:
                out.append(Path(q) / video_stem)
    except Exception:
        pass
    try:
        from mousereach.review.bundle_media import filed_media_dirs
        out += [Path(d) for d in filed_media_dirs(video_stem)[:1]]
    except Exception:
        pass
    return out


def effective_review(video_stem: str, dirs=(), seg_doc=None, current_segments=None,
                     human_segmentation=None, unreadable: Optional[list] = None,
                     strict: bool = False):
    """THE way to read a video's review: every copy (``{stem}_causal_review.json``
    in each of ``dirs``, then the durable copy -- review_copies), merged per
    current segment to the newest answer or recheck a person gave about its
    frames (merge_review_copies), matched against the segmentation in
    ``seg_doc`` -- or against ``current_segments`` (``[{segment_num,
    start_frame, end_frame}]``) when the caller has them already. Returns the
    merged document, or None when no copy exists.

    WHY one function: the review window's restore, its "already reviewed"
    check and the bulk release check each picked their own copies -- the
    first copy with records, a union over two folders, the bundle copy alone
    -- so one could call a pellet answered that another called open, or miss
    an answer only the durable copy still held.

    ``strict`` is passed to review_copies: True raises when a copy exists but
    cannot be read, for callers that would act on a pellet looking open."""
    current = current_segments or (segments_from_segmentation(seg_doc) if seg_doc else None)
    human = (segmentation_is_human(seg_doc) if human_segmentation is None
             else bool(human_segmentation))
    doc, _ = merge_review_copies(review_copies(video_stem, dirs, unreadable=unreadable,
                                              strict=strict),
                                 current or None,
                                 human_segmentation=human)
    return doc


def carry_earlier_answers(segments, earlier_docs, unmatched_answers=None):
    """What a save must still hold of the earlier copies it is about to
    replace. Returns ``(segments, unmatched_answers, kept)``.

    For each current segment whose new record is NOT an answer (a placeholder
    or a pellet left open) while an earlier copy's newest statement about its
    frames is an answer or a recheck, that earlier record takes its place: a
    save can add or change answers, never drop one. An earlier answer whose
    frames match no current segment is added to ``unmatched_answers`` unless it
    is there already. ``kept`` lists the segment numbers whose earlier answer
    was put back, for the log. The current segments are read from the new
    records' own ``segment_span``; records without one match by number."""
    current = []
    for rec in segments:
        sp = _span_of(rec)
        if sp is not None and rec.get("segment_num") is not None:
            current.append({"segment_num": int(rec["segment_num"]),
                            "start_frame": sp[0], "end_frame": sp[1]})
    if len(current) != len(segments):
        current = None
    chosen, unmatched, _ = newest_statements(earlier_docs, current,
                                             human_segmentation=True)
    out, kept = [], []
    for rec in segments:
        sn = rec.get("segment_num")
        prior = chosen.get(int(sn)) if sn is not None else None
        if prior is not None and not is_answered(rec):
            prior = dict(prior)
            prior["segment_num"] = int(sn)
            if is_answered(prior):
                kept.append(int(sn))
            out.append(prior)
            continue
        out.append(rec)
    carried = list(unmatched_answers or [])
    have = {(json.dumps(r.get("segment_span"), sort_keys=True, default=str),
             json.dumps(_applied_call(r), sort_keys=True, default=str))
            for r in list(carried) + out if isinstance(r, dict)}
    for rec in unmatched:
        key = (json.dumps(rec.get("segment_span"), sort_keys=True, default=str),
               json.dumps(_applied_call(rec), sort_keys=True, default=str))
        if key not in have:
            have.add(key)
            carried.append(rec)
    return out, carried, kept


def apply_review_overrides(
    outcomes_data: Dict[str, Any],
    review_by_seg: Dict[int, Dict[str, Any]],
    reviewer: Optional[str] = None,
) -> Dict[str, Any]:
    """Override the algo's per-segment ``outcome`` + ``causal_reach_id`` with the
    human review, for segments that were actually reviewed.

    NON-DESTRUCTIVE: returns a deep copy. The algo's originals are preserved on
    each segment as ``algo_outcome`` / ``algo_causal_reach_id``; provenance is
    recorded as ``outcome_source`` ("human_review" or "algo").

    - A touched human outcome (retrieved / displaced_sa / displaced_outside)
      sets ``causal_reach_id`` to the human's causal-reach id.
    - untouched / abnormal_exception carry NO causal reach (``causal_reach_id``
      = None), so kinematics excludes them from causal-reach kinematics.
    - abnormal ignore-windows are carried onto the segment as ``abnormal_ranges``.
    - Segments without an answer (``is_answered``: unreviewed, ``reviewed:
      False`` placeholders, or a pellet saved with its outcome left undecided)
      are untouched.
    """
    import copy
    out = copy.deepcopy(outcomes_data)
    by_seg = {int(k): v for k, v in (review_by_seg or {}).items()}
    for seg in out.get("segments", []):
        seg["outcome_source"] = "algo"
        sn = seg.get("segment_num")
        if sn is None:
            continue
        rec = by_seg.get(int(sn))
        if not rec:
            continue
        ans = rec.get("answers") or {}
        # Not an answer (is_answered): a placeholder, or a pellet left undecided.
        # Applying the latter stamped the algorithm's "triaged" onto the result
        # as outcome_source "human_review" -- a non-answer recorded as a person's.
        if not is_answered(rec):
            continue
        human = rec.get("human") or {}
        ho = human.get("outcome")
        if ho is None:
            continue
        hc = human.get("causal_reach") or {}
        seg["algo_outcome"] = seg.get("outcome")
        seg["algo_causal_reach_id"] = seg.get("causal_reach_id")
        seg["outcome"] = ho
        seg["causal_reach_id"] = hc.get("reach_id") if ho in _TOUCHED_OUTCOMES else None
        seg["outcome_source"] = "human_review"
        if seg["causal_reach_id"] is not None:
            # Who chose the causal reach (the kinematics read this to keep a
            # person's pick; a touched pellet with no pick is left to the
            # reach assignment -- feature_extractor.causal_reach_owned_by_person).
            seg["causal_reach_source"] = "human_review"
        # Who answered this segment; the file-level reviewer only for files
        # saved before records carried their own (segment_reviewer).
        who = rec.get("reviewer") or reviewer
        if who:
            seg["reviewed_by"] = who
        if ans.get("abnormal_ranges"):
            seg["abnormal_ranges"] = ans["abnormal_ranges"]
    return out


def load_and_apply_review(
    outcomes_data: Dict[str, Any],
    review_path: Path,
    video_stem: Optional[str] = None,
    current_segments: Optional[List[Dict[str, Any]]] = None,
    human_segmentation: bool = False,
) -> Dict[str, Any]:
    """Load a causal-review file and apply its corrections onto ``outcomes_data``.

    Pass ``current_segments`` (the segments of the segmentation being applied to,
    each with start_frame / end_frame) so reviews are matched to the frames the
    reviewer actually saw rather than to a segment number the segmenter may have
    reassigned. Without it the old number-matching is used.

    ``review_path`` may be the review JSON itself OR a directory containing
    ``{video_stem}_causal_review.json``. Returns ``outcomes_data`` unchanged if
    no review is found (safe no-op)."""
    review_path = Path(review_path)
    doc = None
    if review_path.is_file():
        doc = _read_json(review_path)
    elif review_path.is_dir():
        stem = video_stem or outcomes_data.get("video_name") or ""
        cand = review_path / f"{stem}_causal_review.json"
        if cand.exists():
            doc = _read_json(cand)
    if not doc:
        return outcomes_data
    by_seg, notes = index_review_by_segment(doc, current_segments,
                                            human_segmentation=human_segmentation)
    for n in notes:
        logger.warning("%s: %s", video_stem or "", n)
    return apply_review_overrides(outcomes_data, by_seg, reviewer=doc.get("reviewer"))


# ---------------------------------------------------------------------------
# Ground-truth detection -- a video that's already GT'd shouldn't be reviewed;
# we already know the answer. GT files are chained to a video by stem and may
# live anywhere (improvement snapshots, Processing, next to the video, or in the
# bundle), so we index them by stem across the known roots.
# ---------------------------------------------------------------------------

def _gt_index_roots():
    """Where ground-truth files may live: the configured Processing tree, plus
    any extra roots in MOUSEREACH_GT_ROOTS (path-separator-joined). No
    lab-specific default -- an unconfigured install indexes nothing."""
    import os
    roots = []
    try:
        from mousereach.config import Paths
        if Paths.PROCESSING:
            roots.append(Path(Paths.PROCESSING))
    except Exception:
        pass
    for extra in (os.environ.get("MOUSEREACH_GT_ROOTS") or "").split(os.pathsep):
        if extra.strip():
            roots.append(Path(extra.strip()))
    return roots


GT_INDEX_ROOTS = _gt_index_roots()

_GT_INDEX_CACHE: Optional[Dict[str, Path]] = None


def build_gt_index(roots=None) -> Dict[str, Path]:
    """Scan the GT roots for ``*_unified_ground_truth.json`` -> {stem: path}
    (most-recently-modified wins on duplicates)."""
    roots = roots or GT_INDEX_ROOTS
    idx: Dict[str, Path] = {}
    for r in roots:
        r = Path(r)
        if not r.exists():
            continue
        for p in r.rglob("*_unified_ground_truth.json"):
            stem = p.name.split("_unified_ground_truth.json")[0]
            try:
                if stem not in idx or p.stat().st_mtime > idx[stem].stat().st_mtime:
                    idx[stem] = p
            except OSError:
                idx.setdefault(stem, p)
    return idx


def gt_index(refresh: bool = False) -> Dict[str, Path]:
    """Cached stem -> unified-GT-path index (built once per process)."""
    global _GT_INDEX_CACHE
    if _GT_INDEX_CACHE is None or refresh:
        _GT_INDEX_CACHE = build_gt_index()
    return _GT_INDEX_CACHE


def find_gt(video_stem: str, extra_dirs=()) -> Optional[Path]:
    """Path to the video's unified GT if it exists ANYWHERE, else None.

    Checks co-located dirs first (bundle / next-to-video -- "chained at the hip"),
    then the corpus-wide GT index.
    """
    for d in extra_dirs:
        p = Path(d) / f"{video_stem}_unified_ground_truth.json"
        if p.exists():
            return p
    return gt_index().get(video_stem)


def has_gt(video_stem: str, extra_dirs=()) -> bool:
    return find_gt(video_stem, extra_dirs) is not None


# ---------------------------------------------------------------------------
# Session flags -- "this mouse+day has a physical issue; review ALL its videos"
# ---------------------------------------------------------------------------

def session_key(video_stem: str) -> str:
    """Group key for videos of the same mouse on the same day: {date}_{mouse}.

    e.g. ``20250624_CNT0101_P1`` -> ``20250624_CNT0101``. A session-level issue
    (a cage artifact that day, a misplaced pellet, etc.) affects every ``P#`` of
    that mouse+day, so they should all get human review.
    """
    parts = str(video_stem).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(video_stem)


# ---------------------------------------------------------------------------
# Shared corpus files: one writer at a time
# ---------------------------------------------------------------------------
#
# The corpus index (review_records/causal_review_index.json) and
# flagged_sessions.json are each ONE file that every review window on every
# machine reads, changes and writes back. Two reviewers saving at the same
# moment both read the old file, each added their own entries, and the second
# write threw away the first one's. So the read-change-write is done while
# holding a lock file beside the shared file:
#
#   <file>.lock   created with O_CREAT | O_EXCL -- the file server creates it
#                 only if it does not exist, so exactly one window holds it;
#                 it records a random token, the host and the process.
#
# Inside the lock the file is read AGAIN (never a copy read before waiting),
# changed, and replaced in one step (_write_json). The lock is removed only by
# the holder whose token it carries.
#
# WHY not watcher.claim_marker: a claim there is held per HOST, and a host that
# already holds a claim "wins" again -- two windows on the same machine (two
# reviewers logged on to one server) would both get in. WHY not one index file
# per video: readers outside this package read the single file.
#
# A lock whose holder died (a window killed mid-save) is broken once it is
# INDEX_LOCK_STALE_S old, measured by the file server's own clock: its age is
# the modified time of a probe file written now minus the lock's modified time,
# both stamped by the server, so machines whose clocks disagree cannot break a
# live lock early. A waiter does not write a probe file on every poll: it
# probes once to learn how far its own clock is from the server's, estimates
# from that, and probes again only when the estimate says the lock is stale or
# the last probe is SHARE_CLOCK_REFRESH_S old (_ShareClock).
#
# Breaking a stale lock must never remove a LIVE one. Between looking at the
# stale lock and removing it, another waiter may break it too and a new
# holder take a fresh lock under the same name. So the breaker reads the
# lock's token BEFORE looking at its age, moves the lock aside under a name of
# its own (one rename: only one breaker can win it), and then checks that what
# it moved is still the file it judged stale -- same token, same modified
# time, same size. If not, it moved a fresh holder's lock and renames it back.
# Residual limit, accepted: if in that instant a THIRD writer has created a
# new lock, the rename back is refused, the fresh holder has lost its lock and
# two writers can be inside at once (logged at ERROR). That needs a lock left
# by a dead window, two windows breaking it at the same instant and a third
# arriving within the same few milliseconds.
#
# A writer that cannot get the lock within its wait gives up with
# IndexLockBusy, naming the lock file. Background callers wait up to
# INDEX_LOCK_WAIT_S. The review window waits only GUI_LOCK_WAIT_S: it waits on
# the window's own thread, so a long wait froze the window ("Not responding")
# on every save while a dead window's lock was younger than
# INDEX_LOCK_STALE_S -- and a frozen window invites being killed, which is how
# stale locks are made. For the index, the review itself is already saved by
# then (both copies are written before the index) and the window says the
# index was not updated and asks for one more Save: the video's index
# entries are written again only when it is saved again (nothing retries).

INDEX_LOCK_WAIT_S = 60.0
GUI_LOCK_WAIT_S = 3.0
INDEX_LOCK_STALE_S = 600.0
INDEX_LOCK_POLL_S = 0.25
# How often (seconds) a waiter re-learns the server clock's offset even when
# its estimate says the lock is still fresh.
SHARE_CLOCK_REFRESH_S = 30.0


class IndexLockBusy(RuntimeError):
    """Another writer held a shared corpus file's lock for longer than the
    caller's wait (and it was not stale). The message names the lock file."""


def _lock_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + ".lock")


def _lock_token(lock: Path) -> Optional[str]:
    try:
        return json.loads(Path(lock).read_text(encoding="utf-8")).get("token")
    except Exception:
        return None


def _share_now(folder: Path) -> Optional[float]:
    """'Now' by the file server's clock: the modified time of a probe file
    written in ``folder``. None when it cannot be written."""
    import tempfile
    try:
        fd, name = tempfile.mkstemp(prefix=".clock.", suffix=".tmp", dir=str(folder))
    except OSError:
        return None
    try:
        os.write(fd, b"0")
        os.close(fd)
        return os.stat(name).st_mtime
    except OSError:
        return None
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


class _ShareClock:
    """The file server's 'now', without a probe file on every poll.

    One probe (_share_now) gives the offset between this machine's clock and
    the server's; later readings are this machine's clock plus that offset.
    A decision to break a lock is always made on a FRESH probe (confirm), so
    a drifting offset can delay a break but never cause an early one."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.offset: Optional[float] = None
        self.probed_at: Optional[float] = None   # time.monotonic() of last probe

    def confirm(self) -> Optional[float]:
        now = _share_now(self.folder)
        if now is not None:
            self.offset = now - time.time()
            self.probed_at = time.monotonic()
        return now

    def just_probed(self) -> bool:
        return (self.probed_at is not None
                and time.monotonic() - self.probed_at < 1.0)

    def estimate(self) -> Optional[float]:
        if (self.offset is None or self.probed_at is None
                or time.monotonic() - self.probed_at > SHARE_CLOCK_REFRESH_S):
            return self.confirm()
        return time.time() + self.offset


def _break_if_stale(lock: Path, stale_s: float,
                    clock: Optional[_ShareClock] = None) -> bool:
    """Remove ``lock`` if its holder is gone (see the block comment above).
    True when the lock is gone afterwards, so the caller may try again."""
    # First a look at the lock's age from its metadata only, WITHOUT opening
    # it. WHY: on Windows a file opened for reading (read_text) cannot be
    # deleted while it is open, so a waiter that read the lock on every poll
    # made the holder's release fail whenever the two met -- and the released
    # lock then stood for INDEX_LOCK_STALE_S. A live lock is never opened by
    # a waiter now; only one old enough to be stale is.
    try:
        st0 = os.stat(str(lock))
    except FileNotFoundError:
        return True
    except OSError:
        return False
    clock = clock or _ShareClock(lock.parent)
    est = clock.estimate()
    if est is None or est - st0.st_mtime < stale_s:
        return False
    # It may be stale. The token is read BEFORE the age that decides. Read
    # after it, a fresh holder's token (taken in the gap) would be compared
    # with itself below and its live lock removed.
    stale_token = _lock_token(lock)
    try:
        st = os.stat(str(lock))
    except FileNotFoundError:
        return True
    except OSError:
        return False
    # Decide on a fresh reading of the server clock (the estimate itself,
    # when it was just probed).
    now = est if clock.just_probed() else clock.confirm()
    if now is None or now - st.st_mtime < stale_s:
        return False
    aside = lock.with_name(".%s.broken.%s.tmp" % (lock.name, uuid.uuid4().hex))
    try:
        os.rename(str(lock), str(aside))
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        ast = os.stat(str(aside))
        same = (ast.st_mtime == st.st_mtime and ast.st_size == st.st_size
                and _lock_token(aside) == stale_token)
    except OSError:
        same = False
    if not same:
        # What we moved is not the lock we judged stale: another waiter broke
        # that one and a new holder took a fresh lock in between. Put it back.
        try:
            os.rename(str(aside), str(lock))
        except OSError as e:
            # The residual limit in the block comment: a third writer already
            # holds a new lock, and the holder whose lock we moved cannot be
            # told. Its moved lock is deleted (it guards nothing now).
            logger.error("Lock on %s: moved a live lock aside while breaking a "
                         "stale one and could not put it back (%s); two writers "
                         "may be updating it at once",
                         lock.name[:-len(".lock")], e)
            if not _unlink_lock_file(aside):
                logger.error("Could not delete the moved-aside lock %s", aside)
        return False
    if not _unlink_lock_file(aside):
        # It no longer guards anything (its name is not the lock's), so it is
        # only clutter -- but said, so a stray file is explicable.
        logger.warning("Could not delete the moved-aside stale lock %s", aside)
    logger.warning("Broke a stale lock on %s (held %.0f s; its holder is gone)",
                   lock.name[:-len(".lock")], now - st.st_mtime)
    return True


# How long (seconds) removing a lock file keeps retrying while something has
# it open for a moment (Windows refuses to delete an open file).
LOCK_UNLINK_RETRY_S = 2.0


def _unlink_lock_file(path: Path, retry_s: Optional[float] = None) -> bool:
    """Delete a lock file; True when it is gone.

    WHY the retry: on Windows (and over an SMB share) a delete is refused
    with a sharing violation while any other handle has the file open -- a
    waiter reading a possibly-stale lock's token, a virus scanner, an indexer.
    Those holds last milliseconds. Giving up on the first refusal left a
    released lock standing for INDEX_LOCK_STALE_S, and for those ten minutes
    every save skipped the index and every Flag Session failed."""
    from mousereach.pipeline.fsutil import is_transient_lock
    retry_s = LOCK_UNLINK_RETRY_S if retry_s is None else retry_s
    deadline = time.monotonic() + retry_s
    while True:
        try:
            os.unlink(str(path))
            return True
        except FileNotFoundError:
            return True
        except OSError as e:
            if not is_transient_lock(e) or time.monotonic() >= deadline:
                return False
            time.sleep(0.02)


# How many times a refused create of a lock that does NOT exist is retried
# (LOCK_CREATE_REFUSED_SLEEP_S apart) before the refusal is raised.
LOCK_CREATE_REFUSED_TRIES = 8
LOCK_CREATE_REFUSED_SLEEP_S = 0.05


def _take_lock(lock: Path, token: str, wait_s: Optional[float] = None) -> None:
    """Create ``lock`` holding ``token``; wait at most ``wait_s`` for it.

    Every pass of the loop ends at the deadline -- including a pass where
    the lock vanished (a stale break, a holder's release) -- so a caller's
    wait is always honoured. WHY: the review window takes this lock on its
    own thread; a loop that could go round without looking at the clock
    froze the window for good."""
    wait_s = INDEX_LOCK_WAIT_S if wait_s is None else wait_s
    deadline = time.monotonic() + wait_s
    clock = _ShareClock(lock.parent)
    body = json.dumps({"token": token, "host": socket.gethostname(),
                       "pid": os.getpid(), "taken_at": _get_timestamp()}).encode("utf-8")
    from mousereach.pipeline.fsutil import is_transient_lock
    refused = 0
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError) as e:
            # A PermissionError here is Windows' answer for a lock file that
            # is being deleted at this instant (a holder releasing it) or a
            # passing sharing violation on the share: the lock is busy for a
            # moment, so it is waited out like a held lock, not raised.
            if not isinstance(e, FileExistsError) and not is_transient_lock(e):
                raise
            if isinstance(e, PermissionError) and not os.path.lexists(str(lock)):
                # The create was refused and there is no lock to wait for:
                # most likely this account may not create files in the
                # folder at all. A delete-pending lock clears within
                # milliseconds, so a few short retries cover that case; past
                # them the refusal is the answer and is raised (the window
                # shows it), never retried for the whole wait.
                refused += 1
                if (refused >= LOCK_CREATE_REFUSED_TRIES
                        or time.monotonic() >= deadline):
                    raise
                time.sleep(LOCK_CREATE_REFUSED_SLEEP_S)
                continue
            if (_break_if_stale(lock, INDEX_LOCK_STALE_S, clock)
                    and time.monotonic() < deadline):
                # The lock is gone: try again at once -- but only inside the
                # wait, so this branch can never loop past the deadline.
                continue
            if time.monotonic() >= deadline:
                raise IndexLockBusy(
                    "%s is being written by another review window (gave up "
                    "after %.0f s). If no other review window is saving, the "
                    "lock was left by a window that closed mid-save: it is "
                    "removed automatically %d minutes after it was made, or "
                    "delete %s by hand while no review window is open"
                    % (lock.name[:-len(".lock")], wait_s,
                       int(INDEX_LOCK_STALE_S // 60), lock))
            time.sleep(INDEX_LOCK_POLL_S)
            continue
        try:
            os.write(fd, body)
        except BaseException:
            # The lock exists but says nothing (no token): left standing it
            # would block every writer for INDEX_LOCK_STALE_S. Nobody else
            # can hold it (we created it), so it is removed before the error
            # goes on.
            os.close(fd)
            _unlink_lock_file(lock)
            raise
        os.close(fd)
        return


def _release_lock(lock: Path, token: str) -> None:
    held = _lock_token(lock)
    if held == token:
        # Retried for a moment: something (a scanner, an indexer, a waiter
        # judging a stale lock) may have the lock open for reading at this
        # instant, and Windows refuses to delete an open file
        # (_unlink_lock_file).
        if not _unlink_lock_file(lock):
            logger.error("Could not remove the lock %s after %.0f s of "
                         "retrying; until it is %d minutes old every review "
                         "save skips the index and Flag Session fails",
                         lock, LOCK_UNLINK_RETRY_S, int(INDEX_LOCK_STALE_S // 60))
    elif held is None and lock.exists():
        # Our own lock could not be read back: left in place (it may not be
        # ours), and broken once stale. Said, so a slow index is explicable.
        logger.warning("Could not read back the lock %s to remove it; it is "
                       "broken once stale", lock)


def _locked_update(path: Path, change: Callable[[Dict[str, Any]], None],
                   wait_s: Optional[float] = None) -> Dict[str, Any]:
    """Read ``path`` (a JSON object, {} when absent), apply ``change`` to it,
    and write it back, all while holding ``<path>.lock``. Returns the document
    written. Raises IndexLockBusy when the lock cannot be had within
    ``wait_s`` (default INDEX_LOCK_WAIT_S), and whatever the read or the write
    raises (an unreadable file is never overwritten)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(path)
    token = uuid.uuid4().hex
    _take_lock(lock, token, wait_s)
    try:
        doc = _read_json(path)
        change(doc)
        _write_json(path, doc)
        return doc
    finally:
        _release_lock(lock, token)


def _flagged_sessions_path(review_root: Path) -> Path:
    return Path(review_root) / "flagged_sessions.json"


_FLAGGED_SESSIONS_CACHE: Dict[str, Dict[str, Any]] = {}


def load_flagged_sessions(review_root: Path, use_cache: bool = True) -> Dict[str, Any]:
    """Return {session_key: flag_record} for the review corpus.

    PERF: cached per review_root for the process. This is read once per bundle
    during the review-queue scan (via :func:`is_session_flagged`); without the
    cache that re-reads the same small NAS file hundreds of times per scan. The
    cache is invalidated by :func:`flag_session` (the only writer)."""
    key = str(review_root)
    if use_cache and key in _FLAGGED_SESSIONS_CACHE:
        return _FLAGGED_SESSIONS_CACHE[key]
    p = _flagged_sessions_path(review_root)
    data = _read_json(p).get("sessions", {}) if p.exists() else {}
    _FLAGGED_SESSIONS_CACHE[key] = data
    return data


def is_session_flagged(video_stem: str, review_root: Path) -> bool:
    return session_key(video_stem) in load_flagged_sessions(review_root)


def flag_session(video_stem: str, review_root: Path, reason: str = "",
                 flagged_by: Optional[str] = None,
                 lock_wait_s: Optional[float] = None) -> str:
    """Flag the video's whole session (mouse+day) as must-be-human-reviewed.

    Idempotent: re-flagging the same session updates the record. Returns the
    session key. Videos already GT'd/reviewed are still honored -- the flag only
    forces review of the ones that aren't yet resolved. ``lock_wait_s``: how
    long to wait for another writer's lock (the review window passes
    GUI_LOCK_WAIT_S); IndexLockBusy after that.
    """
    key = session_key(video_stem)
    p = _flagged_sessions_path(review_root)
    record = {
        "session_key": key,
        "reason": reason,
        "flagged_by": flagged_by or _get_username(),
        "flagged_at": _get_timestamp(),
        "requires_human_review": True,
    }

    def change(doc):
        doc.setdefault("type", "flagged_sessions")
        doc.setdefault("schema_version", "1.0")
        doc.setdefault("sessions", {})[key] = record
        doc["last_updated"] = _get_timestamp()

    # Under the shared-file lock (block comment above _lock_path): two windows
    # flagging at once must not drop each other's session.
    _locked_update(p, change, lock_wait_s)
    _FLAGGED_SESSIONS_CACHE.pop(str(review_root), None)  # invalidate read cache
    return key


# ---------------------------------------------------------------------------
# Corpus index
# ---------------------------------------------------------------------------

def _default_index_dir() -> Optional[Path]:
    """Resolve the corpus index directory.

    Prefers NAS_ROOT/review_records/ if NAS is configured,
    otherwise falls back to PROCESSING_ROOT/review_records/.
    """
    try:
        from mousereach.config import Paths
        if Paths.NAS_ROOT is not None:
            return Paths.NAS_ROOT / "review_records"
        if Paths.PROCESSING_ROOT is not None:
            return Paths.PROCESSING_ROOT / "review_records"
    except Exception:
        pass
    return None


def update_corpus_index(
    video_stem: str,
    review_file_path: Path,
    segments: List[Dict[str, Any]],
    reviewer: str,
    reviewed_at: str,
    index_dir: Optional[Path] = None,
    lock_wait_s: Optional[float] = None,
) -> Path:
    """Append or update corpus-level index entries for a reviewed video.

    The index is a JSON file mapping ``(video_stem, segment_num)`` pairs
    to their review file path plus key summary fields. Re-reviewing a
    segment overwrites the prior entry (idempotent).

    Parameters
    ----------
    video_stem : str
        Video identifier.
    review_file_path : Path
        Absolute path to the per-video review JSON.
    segments : list of dict
        Per-segment review records.
    reviewer : str
        Who reviewed.
    reviewed_at : str
        ISO timestamp.
    index_dir : Path, optional
        Override index directory. Defaults to NAS/review_records/.
    lock_wait_s : float, optional
        How long to wait for another writer's lock on the index (default
        INDEX_LOCK_WAIT_S; the review window passes GUI_LOCK_WAIT_S so it
        never freezes for long). IndexLockBusy after that.

    Returns
    -------
    Path to the index file.
    """
    if index_dir is None:
        index_dir = _default_index_dir()
    if index_dir is None:
        # No NAS or PROCESSING configured -- skip silently
        return Path("(no index directory configured)")

    index_path = Path(index_dir) / "causal_review_index.json"
    new_entries: Dict[str, Dict[str, Any]] = {}

    # 1.1 carries the full per-segment payload, not just a pointer plus two
    # summary fields. Entries written under 1.0 keep their shape until the
    # segment is reviewed again; read them defensively.
    for seg in segments:
        seg_num = seg.get("segment_num")
        key = f"{video_stem}__seg{seg_num}"
        human = seg.get("human", {}) or {}
        entry = {
            "video_stem": video_stem,
            "segment_num": seg_num,
            "pellet_num": seg.get("pellet_num"),
            "review_file": str(review_file_path),
            "durable_file": str(durable_review_path(video_stem) or ""),
            # Who answered THIS segment (the record's own values, saved by
            # save_causal_review), not who saved the file last.
            "reviewer": seg.get("reviewer") or reviewer,
            "reviewed_at": seg.get("reviewed_at") or reviewed_at,
            "human_outcome": human.get("outcome"),
            "agreed": human.get("agreed", False),
            # --- everything below is what makes a lost review recoverable ---
            # The index used to hold only the two fields above plus a path. When
            # the file at that path was destroyed, what the reviewer actually
            # decided -- which frames they looked at, which reach they picked --
            # was gone, and the index could not rebuild it. A pointer to a file
            # that no longer exists is not a record.
            "segment_span": seg.get("segment_span"),
            "human_causal_reach": human.get("causal_reach"),
            "human_is_phantom": human.get("is_phantom"),
            "algo_outcome": (seg.get("algo", {}) or {}).get("outcome"),
            "algo_causal_reach": (seg.get("algo", {}) or {}).get("causal_reach"),
            "answers": seg.get("answers"),
            "notes": seg.get("notes", ""),
        }
        if seg.get("true_segment_num") is not None:
            entry["true_segment_num"] = seg["true_segment_num"]
            entry["segmentation_wrong"] = True
        new_entries[key] = entry

    def change(index):
        if "type" not in index:
            index["type"] = "causal_review_index"
        # 1.1 carries the full per-segment payload (comment above).
        index["schema_version"] = "1.1"
        index.setdefault("entries", {}).update(new_entries)
        index["last_updated"] = _get_timestamp()

    # Read, change and write under the shared-file lock (block comment above
    # _lock_path). WHY: two reviewers saving at once used to read the same old
    # index, and the second write dropped the first one's entries.
    _locked_update(index_path, change, lock_wait_s)
    return index_path
