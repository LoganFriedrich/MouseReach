"""
span_to_reaches.py - Translate flat reach spans into the nested VideoReaches
JSON structure that downstream production consumers require.

WHY THIS EXISTS
===============
The improved reach detector (`mousereach.reach.v8.detect_reaches_v8`) operates
at the whole-video level and returns a flat list of `(start_frame, end_frame)`
tuples. It computes nothing else -- no segments, no apex, no extent, no ruler.

Two production consumers, however, read the nested `*_reaches.json` file and
crash on bare spans:

  - `outcomes.core.pellet_outcome.PelletOutcomeDetector.detect` reads
    `reaches_data['segments']`, matches each by `seg['segment_num']`, and reads
    per-reach `start_frame`/`end_frame`/`apex_frame`/`reach_id`.
  - `kinematics.core.feature_extractor.FeatureExtractor.extract` hard-requires
    top-level `video_name`/`total_frames`/`n_segments`/`segments[]`; per segment
    `segment_num`/`ruler_pixels`/`n_reaches`/`reaches[]`; per reach
    `reach_id`/`reach_num`/`start_frame`/`end_frame`/`duration_frames`; and soft
    reads `apex_frame`, `max_extent_pixels`/`max_extent_ruler`.

This module is the single source of truth that re-shapes flat spans into the
EXACT same nested structure the old per-segment detector produced. It reuses the
existing `Reach`/`SegmentReaches`/`VideoReaches` dataclasses from
`reach_detector.py` so the JSON shape is identical; only WHO fills it in changes.

SEGMENT CONVENTION
==================
Critically, the feature extractor zips reach-segments with outcome-segments
POSITIONALLY (`zip(reaches_data['segments'], outcomes_data['segments'])`), and
the outcome detector matches reach-segments by `segment_num`. Misordering or
dropping any segment -- even an empty one -- corrupts everything after the first
gap. So this translator emits EXACTLY ONE segment entry per input segment, in
input order, including segments that contain zero spans.

The caller is responsible for building the `segments` list using the SAME
boundary->segment convention the old reach detector used
(`n_segments = len(boundaries)`, the last segment running to end-of-video), so
that the produced `segment_num` values and segment count match historical
output exactly.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .geometry import compute_segment_geometry, get_boxr_reference, load_segments
from .reach_detector import Reach, SegmentReaches, VideoReaches
from mousereach.lib.causal_attribution import compute_reach_apex


# Stamped onto VideoReaches.detector_version so the produced file records which
# reach detector actually filled it in. The improved detector's own version
# string lives in `mousereach.reach.v8.VERSION`.
try:  # pragma: no cover - import guard only
    from mousereach.reach.v8 import VERSION as _V8_VERSION
except Exception:  # pragma: no cover
    _V8_VERSION = "8.unknown"

DETECTOR_VERSION = _V8_VERSION


# A segment is described to this module as (segment_num, seg_start, seg_end).
# seg_start is inclusive; seg_end is the exclusive upper bound used the same way
# the old detector used it (boundaries[i+1] or len(df)).
Segment = Tuple[int, int, int]

# Span containment for assigning a reach to a segment: a span belongs to the
# segment whose [seg_start, seg_end) contains the span's START frame. This is
# the same reach-start containment rule the v6 outcome detector uses.


def build_video_reaches(
    dlc_df: pd.DataFrame,
    segments: Sequence[Segment],
    spans: Sequence[Tuple[int, int]],
    video_name: str,
    total_frames: int,
) -> VideoReaches:
    """Re-shape flat reach spans into a nested VideoReaches object.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        Flat DLC dataframe (columns like ``RightHand_x``), as loaded by
        ``mousereach.reach.v8.features.load_dlc_h5`` /
        ``mousereach.reach.core.geometry.load_dlc``. Used for apex and
        per-segment ruler geometry.
    segments : sequence of (segment_num, seg_start, seg_end)
        Segments in segments-file order, built with the SAME convention the
        old reach detector used (one segment per boundary, last to end-of-
        video). ``seg_start`` inclusive, ``seg_end`` exclusive.
    spans : sequence of (start_frame, end_frame)
        Flat reach spans for the whole video (``end_frame`` inclusive), as
        returned by ``detect_reaches_v8``.
    video_name : str
        Cleaned video name (``DLC_`` suffix already stripped).
    total_frames : int
        Frame count of the video (``len(dlc_df)``).

    Returns
    -------
    VideoReaches
        Nested structure identical in shape to the old detector's output.
        Pure -- performs no disk writes.
    """
    n_total = len(dlc_df)

    # ------------------------------------------------------------------
    # 1. Assign every span to a segment by reach-START containment.
    #    Orphan spans (start not inside any segment) attach to the nearest
    #    segment by frame distance and flag that segment for review. We never
    #    drop a span.
    # ------------------------------------------------------------------
    # Per-segment accumulator of (start, end) spans, keyed by list index so the
    # output order matches the input segment order exactly.
    seg_spans: List[List[Tuple[int, int]]] = [[] for _ in segments]
    seg_flagged: List[bool] = [False for _ in segments]
    seg_flag_reason: List[Optional[str]] = [None for _ in segments]

    for span_start, span_end in spans:
        seg_idx = _segment_index_for_span_start(span_start, segments)
        if seg_idx is None:
            # Orphan: attach to nearest segment by frame distance, flag it.
            seg_idx = _nearest_segment_index(span_start, segments)
            if seg_idx is None:
                # No segments at all -- nothing we can attach to. Per the
                # never-drop-a-span rule there is no valid home; skip only in
                # the degenerate no-segment case (cannot happen in production).
                continue
            seg_flagged[seg_idx] = True
            seg_flag_reason[seg_idx] = "reach_start_outside_all_segments"
        seg_spans[seg_idx].append((int(span_start), int(span_end)))

    # ------------------------------------------------------------------
    # 2. Build one SegmentReaches per input segment, in order, including
    #    segments with zero spans (n_reaches=0, reaches=[]).
    # ------------------------------------------------------------------
    boxr_x = float(get_boxr_reference(dlc_df))

    global_reach_id = 0  # OLD detector pre-increments -> first reach_id == 1
    segment_results: List[SegmentReaches] = []
    total_reaches = 0
    all_durations: List[int] = []
    all_extents: List[float] = []

    for list_idx, (segment_num, seg_start, seg_end) in enumerate(segments):
        geom = compute_segment_geometry(dlc_df, seg_start, seg_end, segment_num)
        ruler_pixels = geom.ruler_pixels

        # Keep spans in temporal order within the segment.
        spans_here = sorted(seg_spans[list_idx], key=lambda sp: sp[0])

        reaches: List[Reach] = []
        for reach_num, (start_frame, end_frame) in enumerate(spans_here, start=1):
            global_reach_id += 1
            duration_frames = end_frame - start_frame + 1
            apex_frame = int(compute_reach_apex(start_frame, end_frame, dlc_df))

            review_note = None
            if seg_flagged[list_idx] and seg_flag_reason[list_idx] == \
                    "reach_start_outside_all_segments":
                review_note = "reach_start_outside_all_segments"

            reaches.append(Reach(
                reach_id=global_reach_id,
                reach_num=reach_num,
                start_frame=int(start_frame),
                apex_frame=apex_frame,
                end_frame=int(end_frame),
                duration_frames=int(duration_frames),
                # Extent is not cheaply available from the flat detector; the
                # consumers `.get` these and tolerate None.
                max_extent_pixels=None,
                max_extent_ruler=None,
                source="algorithm",
                human_corrected=False,
                review_note=review_note,
            ))

        segment_results.append(SegmentReaches(
            segment_num=int(segment_num),
            start_frame=int(seg_start),
            end_frame=int(seg_end),
            ruler_pixels=round(float(ruler_pixels), 1),
            n_reaches=len(reaches),
            reaches=reaches,
            flagged_for_review=seg_flagged[list_idx],
            flag_reason=seg_flag_reason[list_idx],
        ))

        total_reaches += len(reaches)
        all_durations.extend(r.duration_frames for r in reaches)
        # max_extent_ruler is None here; only include real values in the mean.

    n_segments = len(segments)
    summary = {
        'total_reaches': total_reaches,
        'n_segments': n_segments,
        'reaches_per_segment_mean': round(total_reaches / n_segments, 1) if n_segments else 0,
        'reaches_per_segment_std': float(round(np.std([s.n_reaches for s in segment_results]), 1)) if segment_results else 0.0,
        'mean_duration_frames': float(round(np.mean(all_durations), 1)) if all_durations else 0.0,
        'mean_extent_ruler': float(round(np.mean(all_extents), 3)) if all_extents else 0.0,
    }

    return VideoReaches(
        detector_version=DETECTOR_VERSION,
        video_name=video_name,
        total_frames=int(total_frames if total_frames else n_total),
        boxr_x=round(boxr_x, 1),
        n_segments=n_segments,
        segments=segment_results,
        summary=summary,
        detected_at=datetime.now().isoformat(),
        validated=False,
        validated_by=None,
        validated_at=None,
        corrections_made=0,
        reaches_added=0,
        reaches_removed=0,
        segments_flagged=sum(1 for s in segment_results if s.flagged_for_review),
    )


def detect_video_reaches(dlc_path: Path, segments_path: Path) -> VideoReaches:
    """Shared production reach entry point used by BOTH batch.py and the
    napari pipeline.

    Loads DLC, loads segment boundaries, runs the improved whole-video reach
    detector (`detect_reaches_v8`), then re-shapes the flat spans into the
    nested ``VideoReaches`` structure via ``build_video_reaches``. Returns the
    ``VideoReaches`` object; performs NO disk writes (callers use the existing
    ``ReachDetector.save_results`` to persist).

    This is the single code path; both entry points call it so they produce
    identical ``*_reaches.json``.
    """
    from mousereach.reach.v8 import detect_reaches_v8
    from mousereach.reach.v8.features import load_dlc_h5

    dlc_path = Path(dlc_path)
    segments_path = Path(segments_path)

    dlc_df = load_dlc_h5(dlc_path)
    boundaries = load_segments(segments_path)
    total_frames = len(dlc_df)

    video_name = dlc_path.stem
    if 'DLC_' in video_name:
        video_name = video_name.split('DLC_')[0]

    segments = segments_from_boundaries(boundaries, total_frames)
    spans = detect_reaches_v8(dlc_df)

    return build_video_reaches(
        dlc_df=dlc_df,
        segments=segments,
        spans=spans,
        video_name=video_name,
        total_frames=total_frames,
    )


def segments_from_boundaries(
    boundaries: Sequence[int],
    total_frames: int,
) -> List[Segment]:
    """Build the (segment_num, seg_start, seg_end) list using the SAME
    convention the old reach detector used.

    Each boundary marks the START of a segment. ``n_segments == len(boundaries)``
    and the last segment runs from ``boundaries[-1]`` to ``total_frames``. This
    matches ``reach_detector.ReachDetector.detect`` and
    ``reach_detector_v8.ReachDetectorV8.detect`` exactly so produced
    ``segment_num`` values and the segment count are unchanged.

    ``seg_start`` is inclusive, ``seg_end`` exclusive.
    """
    segments: List[Segment] = []
    n = len(boundaries)
    for seg_idx in range(n):
        seg_start = int(boundaries[seg_idx])
        if seg_idx + 1 < n:
            seg_end = int(boundaries[seg_idx + 1])
        else:
            seg_end = int(total_frames)
        segments.append((seg_idx + 1, seg_start, seg_end))
    return segments


def _segment_index_for_span_start(
    span_start: int,
    segments: Sequence[Segment],
) -> Optional[int]:
    """Return the list index of the segment whose [seg_start, seg_end) contains
    ``span_start`` (reach-start containment). None if no segment contains it.
    """
    for idx, (_segment_num, seg_start, seg_end) in enumerate(segments):
        if seg_start <= span_start < seg_end:
            return idx
    # Last segment's upper bound is exclusive (== total_frames). A span that
    # starts exactly on the final frame is still inside the last segment.
    if segments:
        _num, last_start, last_end = segments[-1]
        if span_start == last_end and last_start <= span_start:
            return len(segments) - 1
    return None


def _nearest_segment_index(
    span_start: int,
    segments: Sequence[Segment],
) -> Optional[int]:
    """Return the list index of the segment nearest to ``span_start`` by frame
    distance. Distance to a segment is 0 if inside, else the gap to the nearer
    edge. Ties resolve to the earlier segment.
    """
    if not segments:
        return None
    best_idx = None
    best_dist = None
    for idx, (_segment_num, seg_start, seg_end) in enumerate(segments):
        if seg_start <= span_start < seg_end:
            return idx  # inside -> distance 0
        if span_start < seg_start:
            dist = seg_start - span_start
        else:  # span_start >= seg_end
            dist = span_start - (seg_end - 1)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
