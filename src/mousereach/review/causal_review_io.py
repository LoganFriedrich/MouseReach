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
reviewer's work outlived a reprocess came down to timing. Measured on
2026-08-24: of 1,686 reviewed videos, 662 had a durable copy, 983 existed only
on one machine's local disk, and 41 existed only inside a Y: triage bundle -- one
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    place every node agrees to look, and the X: backup mirrors the NAS. Returns
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
        except Exception:
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
    half hour; per video it cost several missed lookups on the share -- about
    5 minutes of every scan for 4,597 videos on 2026-10-21, while the work
    loop waited. Stems with no review are absent. Never raises; a place that
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
    # <connectome root>/<cohort folder>; a cohort folder that does not exist
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


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
        except Exception:
            continue
    return None


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

    Returns new record dicts; the caller's are not changed."""
    prev_doc = previous or {}
    by_num: Dict[int, Dict[str, Any]] = {}
    by_span: Dict[str, Dict[str, Any]] = {}
    for p in prev_doc.get("segments") or []:
        if p.get("segment_num") is not None:
            by_num[int(p["segment_num"])] = p
        if p.get("segment_span"):
            by_span[json.dumps(p["segment_span"], sort_keys=True)] = p
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
            cands.append(by_num.get(int(rec["segment_num"])))
        if rec.get("segment_span"):
            cands.append(by_span.get(json.dumps(rec["segment_span"], sort_keys=True)))
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
) -> Path:
    """Write (or overwrite) the per-video causal review JSON.

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
    unmatched_answers : list of dict, optional
        Earlier answers whose frames match no current segment
        (``restore_review_for_segments``). Kept under ``UNMATCHED_ANSWERS_KEY``
        so a re-cut never deletes what a person decided; nothing applies them.

    Returns
    -------
    Path to the working copy (the durable copy is written first, and its
    location is available from ``durable_review_path``).
    """
    reviewer = reviewer or _get_username()
    timestamp = _get_timestamp()
    segments = stamp_segment_reviewers(
        segments, _previous_review(video_stem, output_dir), reviewer, timestamp)

    doc: Dict[str, Any] = {
        "type": "causal_review",
        "schema_version": "1.1",   # 1.1 adds segment_span to every record
        "video_stem": video_stem,
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "provenance": provenance,
        "segments": segments,
    }
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
# (2026-10-25: of 199 typed reaches, 14 lost the causal reach and 6 pointed to
# another reach). Both now call typed_reach_match.
TYPED_REACH_COVER = 0.5

# The longest a TYPED reach may be. A typed reach the detector missed is added
# to the reach list with exactly these frames and measured over them, so a typo
# in a frame number becomes a "reach" of hundreds of frames with nonsense
# kinematics. Real reaches: median 13 frames, 99.9% at most 94 (799,201
# detected reaches, 2026-10-26); typed ones: median 14, all but one at most 60.
# The one outlier, 35183-35914 (732 frames), reads like a slip of the keys.
# 300 frames is far past any real reach and still catches that.
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
# target, and the tie then drops the human's answer (seen live 2026-09-01 on a
# segment spanning frames 35108-35108).
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

    These are the SAME tests the matching applies (index_review_by_segment):
    an answer failing them is dropped there. WHY the review tool asks this at
    the moment of answering: until 2026-10-23 it recorded such an answer, the
    pipeline dropped it without a word, and the video came back asking the same
    question -- one reviewer had answered one segment three times. Part of
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
    typed = ((rec.get("human") or {}).get("causal_reach") or {}).get("typed")
    if typed and typed[1] - typed[0] + 1 > MAX_TYPED_REACH_FRAMES:
        return ("The reach you typed runs %d frames (frames %d-%d); no reach lasts that "
                "long. Check the two frame numbers and type them again."
                % (typed[1] - typed[0] + 1, typed[0], typed[1]))
    return None


# Each matching note (index_review_by_segment), in a reviewer's words, and
# which of its numbers are the frames the note is about -- so the triage banner
# can show a note only on the segment it concerns (Logan, 2026-10-23: "i only
# need the note for the one I'm looking at"). Words None: nothing to act on.
_PLAIN_NOTES = [
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


def segments_from_segmentation(seg_doc) -> List[Dict[str, Any]]:
    """The current segments ``[{segment_num, start_frame, end_frame}]`` from a
    ``*_segments.json`` document's boundaries -- the same rule the review tool
    uses to show them (segment N runs from boundary N-1 to the frame before
    boundary N). Empty when the document has no usable boundaries."""
    bounds = []
    for b in (seg_doc or {}).get("boundaries") or []:
        try:
            bounds.append(int(b.get("frame", b.get("index", 0))) if isinstance(b, dict) else int(b))
        except (TypeError, ValueError):
            return []
    if len(bounds) < 2:
        return []
    return [{"segment_num": i + 1, "start_frame": bounds[i], "end_frame": bounds[i + 1] - 1}
            for i in range(len(bounds) - 1)]


def segmentation_is_human(seg_doc) -> bool:
    """True when a person made (or corrected) this segmentation."""
    return (seg_doc or {}).get("boundary_source") == "human"


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
    watched it. A nightly check then sent the same videos back, every night
    (2026-10-21). Without ``current_segments`` this falls back to numbers, the
    only handle a video without a segmentation file has.
    """
    mapping, _ = index_review_by_segment(review_doc, current_segments,
                                         human_segmentation=human_segmentation)
    return {int(sn) for sn, rec in mapping.items() if is_answered(rec)}


def index_review_by_segment(review_doc, current_segments=None,
                            human_segmentation: bool = False):
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

    A segment shorter than MIN_REAL_SEGMENT_FRAMES is treated as a segmenter
    artifact -- unless ``human_segmentation`` is set: a person who cut the video
    decided where its segments are, and their cuts are logged as made, however
    short (Logan, 2026-10-21: "if a human segmented the video then that's what
    the algo needs to log even if the range was short").

    Args:
        review_doc: a loaded causal review document
        current_segments: the segments of the CURRENT segmentation, each with
            start_frame / end_frame. When absent, behaviour is the old
            number-matching, so callers that have no segmentation still work.
        human_segmentation: the current segmentation was made by a person
            (``boundary_source == "human"`` in its segments file).

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
        if rspan is None:
            # No frames recorded; the number is the only handle there is.
            if old_num is not None and int(old_num) in spans:
                mapping.setdefault(int(old_num), rec)
            continue

        # An answer whose causal reach lies wholly outside the frames it records
        # is about other footage: the review tool once put an old answer back on
        # the wrong segment and the next save stamped that segment's frames on
        # it (239 answers, mostly saved 2026-09, typically shifted by one
        # segment). Applied, it would make a neighbouring segment's reach this
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
        # frames beside it (59 files, 2026-10-22). File order is not time order,
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
    decided left triage and was FILED (125 such records in filed videos and 121
    in the queues that day, reaching the exports as "Undetermined"). A
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
    could finish (2026-10-22). Both now use index_review_by_segment.
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
                 flagged_by: Optional[str] = None) -> str:
    """Flag the video's whole session (mouse+day) as must-be-human-reviewed.

    Idempotent: re-flagging the same session updates the record. Returns the
    session key. Videos already GT'd/reviewed are still honored -- the flag only
    forces review of the ones that aren't yet resolved.
    """
    key = session_key(video_stem)
    p = _flagged_sessions_path(review_root)
    doc = _read_json(p) if p.exists() else {}
    doc.setdefault("type", "flagged_sessions")
    doc.setdefault("schema_version", "1.0")
    sessions = doc.setdefault("sessions", {})
    sessions[key] = {
        "session_key": key,
        "reason": reason,
        "flagged_by": flagged_by or _get_username(),
        "flagged_at": _get_timestamp(),
        "requires_human_review": True,
    }
    doc["last_updated"] = _get_timestamp()
    _write_json(p, doc)
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

    Returns
    -------
    Path to the index file.
    """
    if index_dir is None:
        index_dir = _default_index_dir()
    if index_dir is None:
        # No NAS or PROCESSING configured -- skip silently
        return Path("(no index directory configured)")

    index_path = index_dir / "causal_review_index.json"
    index = _read_json(index_path)

    # Ensure top-level structure
    if "type" not in index:
        index["type"] = "causal_review_index"
    # 1.1 carries the full per-segment payload, not just a pointer plus two
    # summary fields. Entries written under 1.0 keep their shape until the
    # segment is reviewed again; read them defensively.
    index["schema_version"] = "1.1"
    if "entries" not in index:
        index["entries"] = {}

    entries = index["entries"]

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
        entries[key] = entry

    index["last_updated"] = _get_timestamp()
    _write_json(index_path, index)
    return index_path
