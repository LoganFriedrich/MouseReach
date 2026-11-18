"""
Unified Ground Truth File Handler
=================================

Single unified GT file format that combines segmentation, reaches, and outcomes.

Features:
- Split determination for reaches (start_determined, end_determined independent)
- Per-item determination tracking with timestamps
- Component-level exhaustive flags
- Completion status computation
- Migration from old separate GT files
- Backward compatibility with v1 schema (auto-maps verified -> determined)

File pattern: *_unified_ground_truth.json

Schema versions:
- 1.0: Original format with verified/corrected/original_*/source fields
- 2.0: Simplified to determined/determined_by/determined_at + component exhaustive flags
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field, asdict


# Schema version for forward compatibility
SCHEMA_VERSION = "2.0"


@dataclass
class BoundaryGT:
    """Ground truth for a single segment boundary."""
    index: int = 0
    frame: int = 0
    determined: bool = False
    determined_by: Optional[str] = None
    determined_at: Optional[str] = None
    comment: Optional[str] = None  # Free-form notes about this boundary


@dataclass
class ReachGT:
    """Ground truth for a single reach with split start/end determination."""
    reach_id: int = 0
    segment_num: int = 0
    start_frame: int = 0
    # Start boundary determination
    start_determined: bool = False
    start_determined_by: Optional[str] = None
    start_determined_at: Optional[str] = None
    end_frame: int = 0
    # End boundary determination
    end_determined: bool = False
    end_determined_by: Optional[str] = None
    end_determined_at: Optional[str] = None
    # Apex (informational)
    apex_frame: Optional[int] = None
    # Metadata
    exclude_from_analysis: bool = False
    exclude_reason: Optional[str] = None
    comment: Optional[str] = None  # Free-form notes about this reach

    @property
    def fully_determined(self) -> bool:
        """A reach is fully determined when both start AND end are determined."""
        return self.start_determined and self.end_determined


@dataclass
class OutcomeGT:
    """Ground truth for a single segment outcome."""
    segment_num: int = 0
    outcome: str = "unknown"  # "retrieved", "displaced_sa", "displaced_outside", "untouched", etc.
    interaction_frame: Optional[int] = None
    outcome_known_frame: Optional[int] = None
    causal_reach_id: Optional[int] = None
    determined: bool = False
    determined_by: Optional[str] = None
    determined_at: Optional[str] = None
    comment: Optional[str] = None  # Free-form notes about this outcome


@dataclass
class Flag:
    """A flag indicating something needs attention."""
    type: str  # "boundary", "reach", "outcome"
    id: int  # boundary index, reach_id, or segment_num
    reason: str


@dataclass
class CompletionStatus:
    """Tracks completion of all determination tasks."""
    segments_complete: bool = False
    reaches_complete: bool = False
    outcomes_complete: bool = False
    all_complete: bool = False
    flags: List[Flag] = field(default_factory=list)

    # Counts for progress display
    boundaries_determined: int = 0
    boundaries_total: int = 0
    reaches_determined: int = 0  # Fully determined (both start AND end)
    reaches_total: int = 0
    outcomes_determined: int = 0
    outcomes_total: int = 0


@dataclass
class UnifiedGroundTruth:
    """Complete unified ground truth for a video."""
    video_name: str
    type: str = "unified_ground_truth"
    schema_version: str = SCHEMA_VERSION
    created_by: str = ""
    created_at: str = ""
    last_modified_at: str = ""
    last_modified_by: str = ""

    # Completion tracking
    completion_status: CompletionStatus = field(default_factory=CompletionStatus)

    # Segmentation data
    boundaries: List[BoundaryGT] = field(default_factory=list)
    segmentation_exhaustive: bool = False
    segmentation_exhaustive_determined_by: Optional[str] = None
    segmentation_exhaustive_determined_at: Optional[str] = None
    anomalies: List[str] = field(default_factory=list)
    anomaly_annotations: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Reach data
    reaches: List[ReachGT] = field(default_factory=list)
    reaches_exhaustive: bool = False
    reaches_exhaustive_determined_by: Optional[str] = None
    reaches_exhaustive_determined_at: Optional[str] = None

    # Outcome data
    outcomes: List[OutcomeGT] = field(default_factory=list)
    outcomes_exhaustive: bool = False
    outcomes_exhaustive_determined_by: Optional[str] = None
    outcomes_exhaustive_determined_at: Optional[str] = None
    # Determined outcomes of a saved ground truth that match no pellet of the
    # current cuts when the file is reopened (_overlay_gt_determinations).
    # Kept, written back on every save, never applied: a person's answer is
    # never dropped, and never put on footage it was not given for.
    unmatched_outcomes: List[OutcomeGT] = field(default_factory=list)

    def update_completion_status(self) -> CompletionStatus:
        """Recompute completion status from current data."""
        status = CompletionStatus()

        # Boundaries
        status.boundaries_total = len(self.boundaries)
        status.boundaries_determined = sum(1 for b in self.boundaries if b.determined)
        status.segments_complete = (
            self.segmentation_exhaustive
            and status.boundaries_determined == status.boundaries_total
            and status.boundaries_total > 0
        )

        # Reaches - must have BOTH start and end determined
        status.reaches_total = len(self.reaches)
        status.reaches_determined = sum(1 for r in self.reaches if r.fully_determined)
        status.reaches_complete = (
            self.reaches_exhaustive
            and status.reaches_determined == status.reaches_total
            and status.reaches_total > 0
        )

        # Outcomes
        status.outcomes_total = len(self.outcomes)
        status.outcomes_determined = sum(1 for o in self.outcomes if o.determined)
        status.outcomes_complete = (
            self.outcomes_exhaustive
            and status.outcomes_determined == status.outcomes_total
            and status.outcomes_total > 0
        )

        # All complete
        status.all_complete = (
            status.segments_complete
            and status.reaches_complete
            and status.outcomes_complete
        )

        # Build flags list
        status.flags = []

        # Undetermined boundaries
        for b in self.boundaries:
            if not b.determined:
                status.flags.append(Flag("boundary", b.index, "undetermined"))

        # Partially or undetermined reaches
        for r in self.reaches:
            if not r.start_determined:
                status.flags.append(Flag("reach", r.reach_id, "start undetermined"))
            if not r.end_determined:
                status.flags.append(Flag("reach", r.reach_id, "end undetermined"))

        # Undetermined outcomes
        for o in self.outcomes:
            if not o.determined:
                status.flags.append(Flag("outcome", o.segment_num, "undetermined"))

        self.completion_status = status
        return status


def get_username() -> str:
    """Get current username for determination tracking."""
    return os.environ.get("USERNAME", os.environ.get("USER", "unknown"))


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def get_unified_gt_path(video_path: Path, algo_dir: Optional[Path] = None) -> Path:
    """Get the unified GT file path for a video.

    ``algo_dir`` overrides the directory (used when the algo JSONs live apart
    from the video, e.g. copy-free review bundles). Defaults to the video's
    parent directory (the co-located / routine layout).
    """
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")
    base = Path(algo_dir) if algo_dir is not None else video_path.parent
    return base / f"{video_stem}_unified_ground_truth.json"


def load_unified_gt(video_path: Path, algo_dir: Optional[Path] = None) -> Optional[UnifiedGroundTruth]:
    """
    Load unified ground truth file for a video.

    Returns None if file doesn't exist.
    """
    gt_path = get_unified_gt_path(video_path, algo_dir)
    if not gt_path.exists():
        return None

    try:
        with open(gt_path, "r") as f:
            data = json.load(f)
        return _dict_to_unified_gt(data)
    except Exception as e:
        print(f"Error loading unified GT: {e}")
        return None


def save_unified_gt(gt: UnifiedGroundTruth, video_path: Path,
                    algo_dir: Optional[Path] = None) -> Path:
    """
    Save unified ground truth to file.

    Updates timestamps and completion status before saving. ``algo_dir``
    overrides the directory (bundle layout). If ``video_path`` is already a
    ``*_unified_ground_truth.json`` path, it is used verbatim (tolerates
    callers that pass the resolved GT path directly).
    """
    vp = Path(video_path)
    if vp.name.endswith("_unified_ground_truth.json"):
        gt_path = vp
    else:
        gt_path = get_unified_gt_path(vp, algo_dir)

    # Update metadata
    gt.last_modified_at = get_timestamp()
    gt.last_modified_by = get_username()
    if not gt.created_at:
        gt.created_at = gt.last_modified_at
        gt.created_by = gt.last_modified_by

    # Update completion status
    gt.update_completion_status()

    # Convert to dict and save. Through a temporary file and one rename
    # (fsutil.write_json_atomic), never in place: opening the real file for
    # writing empties it first, so a crash mid-save used to leave a truncated
    # ground-truth file that load_unified_gt rejects and every other reader
    # takes for "no ground truth" -- a person's answers lost silently.
    from mousereach.pipeline.fsutil import write_json_atomic
    data = _unified_gt_to_dict(gt)
    write_json_atomic(gt_path, data, indent=2)

    return gt_path


def finalized_ground_truth(gt: UnifiedGroundTruth) -> Tuple[Optional[UnifiedGroundTruth], List[str]]:
    """The copy "Save as Ground Truth" writes, or the reasons it must not be
    written. Returns ``(copy, [])`` or ``(None, problems)``.

    The copy keeps only what a person determined: boundaries and outcomes
    marked determined, reaches with a determined start or end (algorithm
    seeds nobody touched are not ground truth). It also keeps the three
    "this part is complete" (exhaustive) declarations with who made them and
    when. WHY: the copy used to be built without them, and it overwrites the
    file "Save Progress" writes, so pressing the finishing button erased the
    person's declarations; a finished ground truth then read as partial --
    the results kept reaches the person had ruled out and pellets came back
    to triage.

    Refused (nothing to write) when a part is marked complete but the filter
    would drop items from it: "complete" would then claim the shorter list is
    every item, and the dropped ones -- algorithm items in the video the
    person did not get to -- would be read as ruled out. The problem names the
    part and the count; the person determines (or removes) those items, or
    un-marks the part, and saves again.

    A pure function (no Qt), so the rule is tested without a window."""
    kept_b = [b for b in gt.boundaries if b.determined]
    kept_r = [r for r in gt.reaches if r.start_determined or r.end_determined]
    kept_o = [o for o in gt.outcomes if o.determined]
    problems = []
    for part, flag, before, after, noun in (
            ("segmentation", gt.segmentation_exhaustive, gt.boundaries, kept_b, "boundaries"),
            ("reaches", gt.reaches_exhaustive, gt.reaches, kept_r, "reaches"),
            ("outcomes", gt.outcomes_exhaustive, gt.outcomes, kept_o, "outcomes")):
        dropped = len(before) - len(after)
        if flag and dropped:
            problems.append(
                f"{part} is marked complete, but {dropped} of its {len(before)} {noun} "
                f"are not determined and would be left out")
    if problems:
        return None, problems
    out = UnifiedGroundTruth(
        video_name=gt.video_name,
        schema_version=gt.schema_version,
        created_by=gt.created_by,
        created_at=gt.created_at,
        anomalies=gt.anomalies,
        anomaly_annotations=gt.anomaly_annotations,
    )
    out.boundaries, out.reaches, out.outcomes = kept_b, kept_r, kept_o
    # Answers given on cuts the video no longer has: kept, never applied.
    out.unmatched_outcomes = list(gt.unmatched_outcomes)
    for part in ("segmentation", "reaches", "outcomes"):
        for suffix in ("exhaustive", "exhaustive_determined_by", "exhaustive_determined_at"):
            name = f"{part}_{suffix}"
            setattr(out, name, getattr(gt, name))
    return out, []


def _dict_to_unified_gt(data: Dict) -> UnifiedGroundTruth:
    """Convert dict from JSON to UnifiedGroundTruth dataclass.

    Supports both v1 (verified/corrected) and v2 (determined) schemas.
    If schema_version is "1.0" or fields use old names, auto-maps to v2.
    """
    seg_section = data.get("segmentation", {})
    reaches_section = data.get("reaches", {})
    outcomes_section = data.get("outcomes", {})

    gt = UnifiedGroundTruth(
        video_name=data.get("video_name", ""),
        type=data.get("type", "ground_truth"),
        schema_version=SCHEMA_VERSION,  # Always upgrade to current version
        created_by=data.get("created_by", ""),
        created_at=data.get("created_at", ""),
        last_modified_at=data.get("last_modified_at", ""),
        last_modified_by=data.get("last_modified_by", ""),
        anomalies=seg_section.get("anomalies", []),
        anomaly_annotations=seg_section.get("anomaly_annotations", {}),
        # Component-level exhaustive flags
        segmentation_exhaustive=seg_section.get("exhaustive", False),
        segmentation_exhaustive_determined_by=seg_section.get("exhaustive_determined_by"),
        segmentation_exhaustive_determined_at=seg_section.get("exhaustive_determined_at"),
        reaches_exhaustive=reaches_section.get("exhaustive", False),
        reaches_exhaustive_determined_by=reaches_section.get("exhaustive_determined_by"),
        reaches_exhaustive_determined_at=reaches_section.get("exhaustive_determined_at"),
        outcomes_exhaustive=outcomes_section.get("exhaustive", False),
        outcomes_exhaustive_determined_by=outcomes_section.get("exhaustive_determined_by"),
        outcomes_exhaustive_determined_at=outcomes_section.get("exhaustive_determined_at"),
    )

    # Parse boundaries
    for b_data in seg_section.get("boundaries", []):
        # v1 compat: map verified -> determined
        determined = b_data.get("determined", b_data.get("verified", False))
        determined_by = b_data.get("determined_by", b_data.get("verified_by"))
        determined_at = b_data.get("determined_at", b_data.get("verified_at"))
        gt.boundaries.append(BoundaryGT(
            index=b_data.get("index", 0),
            frame=b_data.get("frame", 0),
            determined=determined,
            determined_by=determined_by,
            determined_at=determined_at,
            comment=b_data.get("comment"),
        ))

    # Parse reaches
    for r_data in reaches_section.get("reaches", []):
        # v1 compat: map start_verified -> start_determined, etc.
        start_determined = r_data.get("start_determined", r_data.get("start_verified", False))
        start_determined_by = r_data.get("start_determined_by", r_data.get("start_verified_by"))
        start_determined_at = r_data.get("start_determined_at", r_data.get("start_verified_at"))
        end_determined = r_data.get("end_determined", r_data.get("end_verified", False))
        end_determined_by = r_data.get("end_determined_by", r_data.get("end_verified_by"))
        end_determined_at = r_data.get("end_determined_at", r_data.get("end_verified_at"))
        gt.reaches.append(ReachGT(
            reach_id=r_data.get("reach_id", 0),
            segment_num=r_data.get("segment_num", 0),
            start_frame=r_data.get("start_frame", 0),
            start_determined=start_determined,
            start_determined_by=start_determined_by,
            start_determined_at=start_determined_at,
            end_frame=r_data.get("end_frame", 0),
            end_determined=end_determined,
            end_determined_by=end_determined_by,
            end_determined_at=end_determined_at,
            apex_frame=r_data.get("apex_frame"),
            exclude_from_analysis=r_data.get("exclude_from_analysis", False),
            exclude_reason=r_data.get("exclude_reason"),
            comment=r_data.get("comment"),
        ))

    # Sort reaches by segment then start_frame for consistent display order
    gt.reaches.sort(key=lambda r: (r.segment_num, r.start_frame))

    # Parse outcomes
    for o_data in outcomes_section.get("segments", []):
        # v1 compat: map verified -> determined
        determined = o_data.get("determined", o_data.get("verified", False))
        determined_by = o_data.get("determined_by", o_data.get("verified_by"))
        determined_at = o_data.get("determined_at", o_data.get("verified_at"))
        gt.outcomes.append(OutcomeGT(
            segment_num=o_data.get("segment_num", 0),
            outcome=o_data.get("outcome", "unknown"),
            interaction_frame=o_data.get("interaction_frame"),
            outcome_known_frame=o_data.get("outcome_known_frame"),
            causal_reach_id=o_data.get("causal_reach_id"),
            determined=determined,
            determined_by=determined_by,
            determined_at=determined_at,
            comment=o_data.get("comment"),
        ))

    for o_data in outcomes_section.get("unmatched", []) or []:
        gt.unmatched_outcomes.append(OutcomeGT(
            segment_num=o_data.get("segment_num", 0),
            outcome=o_data.get("outcome", "unknown"),
            interaction_frame=o_data.get("interaction_frame"),
            outcome_known_frame=o_data.get("outcome_known_frame"),
            causal_reach_id=o_data.get("causal_reach_id"),
            determined=o_data.get("determined", False),
            determined_by=o_data.get("determined_by"),
            determined_at=o_data.get("determined_at"),
            comment=o_data.get("comment"),
        ))

    # Update completion status
    gt.update_completion_status()

    return gt


def _outcome_to_dict(o: OutcomeGT) -> Dict:
    return {
        "segment_num": o.segment_num,
        "outcome": o.outcome,
        "interaction_frame": o.interaction_frame,
        "outcome_known_frame": o.outcome_known_frame,
        "causal_reach_id": o.causal_reach_id,
        "determined": o.determined,
        "determined_by": o.determined_by,
        "determined_at": o.determined_at,
        "comment": o.comment,
    }


def _unified_gt_to_dict(gt: UnifiedGroundTruth) -> Dict:
    """Convert UnifiedGroundTruth to dict for JSON serialization (v2 schema)."""
    return {
        "video_name": gt.video_name,
        "type": gt.type,
        "schema_version": gt.schema_version,
        "created_at": gt.created_at,
        "created_by": gt.created_by,
        "last_modified_at": gt.last_modified_at,
        "last_modified_by": gt.last_modified_by,
        "segmentation": {
            "exhaustive": gt.segmentation_exhaustive,
            "exhaustive_determined_by": gt.segmentation_exhaustive_determined_by,
            "exhaustive_determined_at": gt.segmentation_exhaustive_determined_at,
            "n_boundaries": len(gt.boundaries),
            "boundaries": [
                {
                    "index": b.index,
                    "frame": b.frame,
                    "determined": b.determined,
                    "determined_by": b.determined_by,
                    "determined_at": b.determined_at,
                    "comment": b.comment,
                }
                for b in gt.boundaries
            ],
        },
        "reaches": {
            "exhaustive": gt.reaches_exhaustive,
            "exhaustive_determined_by": gt.reaches_exhaustive_determined_by,
            "exhaustive_determined_at": gt.reaches_exhaustive_determined_at,
            "total_reaches": len(gt.reaches),
            "reaches": [
                {
                    "reach_id": r.reach_id,
                    "segment_num": r.segment_num,
                    "start_frame": r.start_frame,
                    "start_determined": r.start_determined,
                    "start_determined_by": r.start_determined_by,
                    "start_determined_at": r.start_determined_at,
                    "end_frame": r.end_frame,
                    "end_determined": r.end_determined,
                    "end_determined_by": r.end_determined_by,
                    "end_determined_at": r.end_determined_at,
                    "apex_frame": r.apex_frame,
                    "exclude_from_analysis": r.exclude_from_analysis,
                    "exclude_reason": r.exclude_reason,
                    "comment": r.comment,
                }
                for r in gt.reaches
            ],
        },
        "outcomes": {
            "exhaustive": gt.outcomes_exhaustive,
            "exhaustive_determined_by": gt.outcomes_exhaustive_determined_by,
            "exhaustive_determined_at": gt.outcomes_exhaustive_determined_at,
            "n_segments": len(gt.outcomes),
            "segments": [_outcome_to_dict(o) for o in gt.outcomes],
            # Kept answers that match no pellet of the cuts they were last
            # opened on; no reader applies them (see UnifiedGroundTruth).
            "unmatched": [_outcome_to_dict(o) for o in gt.unmatched_outcomes],
        },
    }


# =============================================================================
# Migration from Old GT Formats
# =============================================================================


def migrate_from_old_formats(video_path: Path) -> Optional[UnifiedGroundTruth]:
    """
    Create a unified GT by migrating from old separate GT files.

    Looks for:
    - *_seg_ground_truth.json
    - *_reach_ground_truth.json
    - *_outcome_ground_truth.json (or *_outcomes_ground_truth.json)

    Returns None if no old GT files found.
    """
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")

    parent = video_path.parent

    # Look for old GT files
    seg_gt_path = parent / f"{video_stem}_seg_ground_truth.json"
    reach_gt_path = parent / f"{video_stem}_reach_ground_truth.json"
    outcome_gt_paths = [
        parent / f"{video_stem}_outcome_ground_truth.json",
        parent / f"{video_stem}_outcomes_ground_truth.json",
    ]

    seg_gt = None
    reach_gt = None
    outcome_gt = None

    if seg_gt_path.exists():
        with open(seg_gt_path) as f:
            seg_gt = json.load(f)

    if reach_gt_path.exists():
        with open(reach_gt_path) as f:
            reach_gt = json.load(f)

    for p in outcome_gt_paths:
        if p.exists():
            with open(p) as f:
                outcome_gt = json.load(f)
            break

    # If no old GT files, return None
    if not any([seg_gt, reach_gt, outcome_gt]):
        return None

    # Create unified GT
    gt = UnifiedGroundTruth(
        video_name=video_stem,
        created_by=get_username(),
        created_at=get_timestamp(),
    )

    # Migrate segmentation
    if seg_gt:
        _migrate_seg_gt(gt, seg_gt)
    else:
        # No seg GT - load from algorithm output
        _load_segments_from_algo(gt, parent, video_stem)

    # Migrate reaches
    if reach_gt:
        _migrate_reach_gt(gt, reach_gt)
    else:
        # No reach GT - load from algorithm output (undetermined)
        _load_reaches_from_algo(gt, parent, video_stem)

    # Migrate outcomes
    if outcome_gt:
        _migrate_outcome_gt(gt, outcome_gt)
    else:
        # No outcome GT - load from algorithm output
        _load_outcomes_from_algo(gt, parent, video_stem)

    gt.update_completion_status()
    return gt


def _migrate_seg_gt(gt: UnifiedGroundTruth, seg_data: Dict):
    """Migrate old segmentation GT format to v2 unified."""
    boundaries = seg_data.get("boundaries", [])
    boundaries_meta = seg_data.get("boundaries_with_meta", [])

    # If we have metadata, use it
    if boundaries_meta:
        for b_meta in boundaries_meta:
            gt.boundaries.append(BoundaryGT(
                index=b_meta.get("index", len(gt.boundaries)),
                frame=b_meta.get("frame", 0),
                determined=b_meta.get("human_verified", False),
                determined_by=b_meta.get("verified_by"),
                determined_at=b_meta.get("verified_at"),
            ))
    else:
        # Just frame numbers - assume all determined if GT file exists
        for i, frame in enumerate(boundaries):
            gt.boundaries.append(BoundaryGT(
                index=i,
                frame=frame,
                determined=True,  # Assume determined since it's in GT file
            ))

    gt.anomalies = seg_data.get("anomalies", [])
    gt.anomaly_annotations = seg_data.get("anomaly_annotations", {})


def _migrate_reach_gt(gt: UnifiedGroundTruth, reach_data: Dict):
    """Migrate old reach GT format to v2 unified with split determination."""
    for seg in reach_data.get("segments", []):
        for r in seg.get("reaches", []):
            # Old format has human_verified per-reach, not split
            # We'll mark both start and end as determined if the reach was verified
            old_determined = r.get("human_verified", False)

            gt.reaches.append(ReachGT(
                reach_id=r.get("reach_id", 0),
                segment_num=seg.get("segment_num", 0),
                start_frame=r.get("start_frame", 0),
                start_determined=old_determined,
                start_determined_by=r.get("verified_by"),
                start_determined_at=r.get("verified_at"),
                apex_frame=r.get("apex_frame"),
                end_frame=r.get("end_frame", 0),
                end_determined=old_determined,
                end_determined_by=r.get("verified_by"),
                end_determined_at=r.get("verified_at"),
                exclude_from_analysis=r.get("exclude_from_analysis", False),
                exclude_reason=r.get("exclude_reason"),
            ))


def _migrate_outcome_gt(gt: UnifiedGroundTruth, outcome_data: Dict):
    """Migrate old outcome GT format to v2 unified."""
    for seg in outcome_data.get("segments", []):
        gt.outcomes.append(OutcomeGT(
            segment_num=seg.get("segment_num", 0),
            outcome=seg.get("outcome", "unknown"),
            interaction_frame=seg.get("interaction_frame"),
            outcome_known_frame=seg.get("outcome_known_frame"),
            causal_reach_id=seg.get("causal_reach_id"),
            determined=seg.get("human_verified", False),
            determined_by=seg.get("verified_by"),
            determined_at=seg.get("verified_at"),
        ))


# =============================================================================
# Initialize from Algorithm Output
# =============================================================================


def create_from_algorithm_output(video_path: Path,
                                 algo_dir: Optional[Path] = None) -> Optional[UnifiedGroundTruth]:
    """
    Create a new unified GT from algorithm output files.

    Loads data from:
    - *_segments.json or *_seg_validation.json
    - *_reaches.json
    - *_pellet_outcomes.json

    ``algo_dir`` overrides where those JSONs are read from (bundle layout);
    defaults to the video's parent directory. All items start undetermined.
    """
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")

    parent = Path(algo_dir) if algo_dir is not None else video_path.parent

    gt = UnifiedGroundTruth(video_name=video_stem)

    # Load segments
    seg_loaded = _load_segments_from_algo(gt, parent, video_stem)

    # Load reaches
    reach_loaded = _load_reaches_from_algo(gt, parent, video_stem)

    # Load outcomes
    outcome_loaded = _load_outcomes_from_algo(gt, parent, video_stem)

    if not any([seg_loaded, reach_loaded, outcome_loaded]):
        return None

    # Ensure all expected segments have outcome entries
    # Number of segments = number of boundaries - 1 (if boundaries exist)
    if len(gt.boundaries) > 1:
        expected_segments = len(gt.boundaries) - 1
        existing_outcome_segs = {o.segment_num for o in gt.outcomes}

        for seg_num in range(1, expected_segments + 1):
            if seg_num not in existing_outcome_segs:
                # Add placeholder outcome for missing segment
                gt.outcomes.append(OutcomeGT(
                    segment_num=seg_num,
                    outcome="",  # Empty = no data yet
                    determined=False,
                ))

        # Sort outcomes by segment number
        gt.outcomes.sort(key=lambda o: o.segment_num)

    gt.update_completion_status()
    return gt


def _load_segments_from_algo(gt: UnifiedGroundTruth, parent: Path, video_stem: str) -> bool:
    """Load segment boundaries from algorithm output."""
    patterns = [
        f"{video_stem}_seg_validation.json",
        f"{video_stem}_segments_v2.json",
        f"{video_stem}_segments.json",
    ]

    for pattern in patterns:
        path = parent / pattern
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)

                boundaries = data.get("boundaries", data.get("validated_boundaries", []))
                for i, frame in enumerate(boundaries):
                    gt.boundaries.append(BoundaryGT(
                        index=i,
                        frame=frame,
                        determined=False,
                    ))

                gt.anomalies = data.get("anomalies", [])
                return True
            except Exception:
                pass

    return False


def _load_reaches_from_algo(gt: UnifiedGroundTruth, parent: Path, video_stem: str) -> bool:
    """Load reaches from algorithm output."""
    patterns = [
        f"{video_stem}_reaches_v*.json",
        f"{video_stem}_reaches.json",
    ]

    # Try glob patterns
    for pattern in patterns:
        for path in sorted(parent.glob(pattern), reverse=True):  # Newest version first
            try:
                with open(path) as f:
                    data = json.load(f)

                for seg in data.get("segments", []):
                    for r in seg.get("reaches", []):
                        gt.reaches.append(ReachGT(
                            reach_id=r.get("reach_id", 0),
                            segment_num=seg.get("segment_num", 0),
                            start_frame=r.get("start_frame", 0),
                            start_determined=False,
                            apex_frame=r.get("apex_frame"),
                            end_frame=r.get("end_frame", 0),
                            end_determined=False,
                        ))

                return True
            except Exception:
                pass

    return False


def _load_outcomes_from_algo(gt: UnifiedGroundTruth, parent: Path, video_stem: str) -> bool:
    """Load outcomes from algorithm output."""
    patterns = [
        f"{video_stem}_pellet_outcomes_v*.json",
        f"{video_stem}_pellet_outcomes.json",
    ]

    for pattern in patterns:
        for path in sorted(parent.glob(pattern), reverse=True):
            try:
                with open(path) as f:
                    data = json.load(f)

                for seg in data.get("segments", []):
                    gt.outcomes.append(OutcomeGT(
                        segment_num=seg.get("segment_num", 0),
                        outcome=seg.get("outcome", "unknown"),
                        interaction_frame=seg.get("interaction_frame"),
                        outcome_known_frame=seg.get("outcome_known_frame"),
                        determined=False,
                    ))

                return True
            except Exception:
                pass

    return False


# =============================================================================
# Convenience Functions
# =============================================================================


# =============================================================================
# Migration CLI
# =============================================================================


def migrate_gt_files(
    folder: Path,
    archive_folder: Optional[Path] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Migrate old separate GT files to unified format.

    Args:
        folder: Folder containing videos and GT files
        archive_folder: Where to archive old GT files (default: folder/archived_gt/)
        dry_run: If True, just report what would happen without making changes

    Returns:
        Dict with migration statistics
    """
    import shutil

    if archive_folder is None:
        archive_folder = folder / "archived_gt"

    stats = {
        "videos_found": 0,
        "videos_migrated": 0,
        "files_archived": 0,
        "unified_created": 0,
        "errors": [],
        "skipped": [],
    }

    # Find all videos with GT files
    gt_patterns = [
        "*_seg_ground_truth.json",
        "*_reach_ground_truth.json",
        "*_outcome_ground_truth.json",
        "*_outcomes_ground_truth.json",
    ]

    # Group GT files by video_id
    video_gt_files: Dict[str, List[Path]] = {}
    for pattern in gt_patterns:
        for gt_path in folder.glob(pattern):
            # Extract video_id from filename
            video_id = gt_path.stem
            for suffix in ["_seg_ground_truth", "_reach_ground_truth",
                          "_outcome_ground_truth", "_outcomes_ground_truth"]:
                if video_id.endswith(suffix):
                    video_id = video_id[:-len(suffix)]
                    break

            if video_id not in video_gt_files:
                video_gt_files[video_id] = []
            video_gt_files[video_id].append(gt_path)

    stats["videos_found"] = len(video_gt_files)

    if not video_gt_files:
        return stats

    # Process each video
    for video_id, gt_files in video_gt_files.items():
        # Check if unified GT already exists
        unified_path = folder / f"{video_id}_unified_ground_truth.json"
        if unified_path.exists():
            stats["skipped"].append(f"{video_id}: unified GT already exists")
            continue

        # Find video file to use as reference
        video_path = None
        for ext in [".mp4", ".avi"]:
            candidates = list(folder.glob(f"{video_id}*{ext}"))
            candidates = [c for c in candidates if "_preview" not in c.name]
            if candidates:
                video_path = candidates[0]
                break

        if not video_path:
            # Create a dummy path for migration
            video_path = folder / f"{video_id}.mp4"

        if dry_run:
            print(f"Would migrate: {video_id}")
            print(f"  Old GT files: {[f.name for f in gt_files]}")
            print(f"  Archive to: {archive_folder}")
            print(f"  Create: {unified_path.name}")
            continue

        try:
            # Create unified GT from old files
            unified_gt = migrate_from_old_formats(video_path)
            if unified_gt is None:
                stats["errors"].append(f"{video_id}: migration returned None")
                continue

            # Create archive folder if needed
            if not archive_folder.exists():
                archive_folder.mkdir(parents=True)

            # Archive old GT files
            for gt_file in gt_files:
                archive_dest = archive_folder / gt_file.name
                shutil.move(str(gt_file), str(archive_dest))
                stats["files_archived"] += 1

            # Save unified GT
            save_unified_gt(unified_gt, video_path)
            stats["unified_created"] += 1
            stats["videos_migrated"] += 1

        except Exception as e:
            stats["errors"].append(f"{video_id}: {e}")

    return stats


def cli_migrate():
    """CLI entry point for mousereach-migrate-gt."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate old separate GT files to unified format"
    )
    parser.add_argument(
        "folder", type=Path, nargs="?",
        help="Folder to scan (default: Processing folder)"
    )
    parser.add_argument(
        "--archive", "-a", type=Path,
        help="Archive folder for old GT files (default: <folder>/archived_gt/)"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    # Default to Processing folder
    if args.folder is None:
        from mousereach.config import Paths
        args.folder = Paths.PROCESSING_ROOT

    if not args.folder.exists():
        print(f"Error: Folder not found: {args.folder}")
        return

    print("MouseReach GT Migration")
    print("=" * 50)
    print(f"Scanning: {args.folder}")
    if args.dry_run:
        print("(DRY RUN - no changes will be made)")
    print()

    stats = migrate_gt_files(
        args.folder,
        archive_folder=args.archive,
        dry_run=args.dry_run
    )

    print()
    print("Results:")
    print(f"  Videos with old GT files: {stats['videos_found']}")
    print(f"  Videos migrated: {stats['videos_migrated']}")
    print(f"  Files archived: {stats['files_archived']}")
    print(f"  Unified GT files created: {stats['unified_created']}")

    if stats["skipped"]:
        print(f"\nSkipped ({len(stats['skipped'])}):")
        for msg in stats["skipped"][:5]:
            print(f"  - {msg}")
        if len(stats["skipped"]) > 5:
            print(f"  ... and {len(stats['skipped']) - 5} more")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"]:
            print(f"  - {err}")


def load_or_create_unified_gt(video_path: Path,
                              algo_dir: Optional[Path] = None) -> UnifiedGroundTruth:
    """
    Load algorithm outputs, then overlay any existing GT determinations.

    The review tool reviews ALGORITHM OUTPUTS, not GT files.
    GT is the OUTPUT of the review process.

    ``algo_dir`` overrides where the algo JSONs (and the saved unified GT) are
    read from -- used when they live apart from the video (copy-free review
    bundles). Defaults to the video's parent directory (co-located layout).

    Flow:
    1. ALWAYS load algorithm outputs first (segments, reaches, outcomes)
    2. If unified GT exists, overlay determination status and corrections
    3. If old GT files exist, overlay their corrections
    4. Return combined result
    """
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")

    # Step 1: Load algorithm outputs (the data we're reviewing)
    gt = create_from_algorithm_output(video_path, algo_dir)
    if gt is None:
        # No algorithm output - try unified GT or old GTs as fallback
        gt = load_unified_gt(video_path, algo_dir)
        if gt is not None:
            return gt
        gt = migrate_from_old_formats(video_path)
        if gt is not None:
            return gt
        # Return empty GT
        return UnifiedGroundTruth(video_name=video_stem)

    # Step 2: If unified GT exists, overlay determination status
    existing_gt = load_unified_gt(video_path, algo_dir)
    if existing_gt is not None:
        _overlay_gt_determinations(gt, existing_gt)
        return gt

    # Step 3: If old GT files exist, overlay their corrections
    _overlay_old_gt_corrections(gt, video_path)

    return gt


def _segments_of(gt: UnifiedGroundTruth):
    """[(segment_num, start, end)] from a ground truth's boundaries (by their
    index: segment N runs from cut N-1 to the frame before cut N --
    causal_review_io.pellet_span, the one definition)."""
    from mousereach.review.causal_review_io import pellet_span
    by_index = {b.index: b.frame for b in gt.boundaries if b.frame is not None}
    return [(i + 1,) + pellet_span(by_index, i) for i in sorted(by_index) if i + 1 in by_index]


def _overlay_gt_determinations(algo_gt: UnifiedGroundTruth, saved_gt: UnifiedGroundTruth):
    """Overlay determination status from saved GT onto algorithm GT."""
    # Copy exhaustive flags from saved GT
    algo_gt.segmentation_exhaustive = saved_gt.segmentation_exhaustive
    algo_gt.segmentation_exhaustive_determined_by = saved_gt.segmentation_exhaustive_determined_by
    algo_gt.segmentation_exhaustive_determined_at = saved_gt.segmentation_exhaustive_determined_at
    algo_gt.reaches_exhaustive = saved_gt.reaches_exhaustive
    algo_gt.reaches_exhaustive_determined_by = saved_gt.reaches_exhaustive_determined_by
    algo_gt.reaches_exhaustive_determined_at = saved_gt.reaches_exhaustive_determined_at
    algo_gt.outcomes_exhaustive = saved_gt.outcomes_exhaustive
    algo_gt.outcomes_exhaustive_determined_by = saved_gt.outcomes_exhaustive_determined_by
    algo_gt.outcomes_exhaustive_determined_at = saved_gt.outcomes_exhaustive_determined_at

    # Boundaries: match by index and apply determination status
    saved_bounds = {b.index: b for b in saved_gt.boundaries}
    for b in algo_gt.boundaries:
        if b.index in saved_bounds:
            sb = saved_bounds[b.index]
            b.determined = sb.determined
            b.determined_by = sb.determined_by
            b.determined_at = sb.determined_at
            # If saved GT has a different frame, use the determined value
            if sb.determined and sb.frame != b.frame:
                b.frame = sb.frame
            b.comment = sb.comment

    # Reaches: matched by FRAMES, never by reach_id. WHY: reach ids are
    # renumbered on every run, so after a re-pose or a new detector the
    # saved reach "id 7" is another reach of the new list; matched by id, a
    # determined reach took over (and erased) whichever current reach had its
    # number. A saved reach is matched to the current reach that covers most
    # of its frames (largest overlap, as truth_resolver._merge_reaches does);
    # the current reach keeps this run's id. A saved reach that
    # matches nothing is kept as its own row, with a new id if its old one is
    # taken -- never dropped.
    free = list(algo_gt.reaches)
    extra = []
    for sr in saved_gt.reaches:
        best, best_ov = None, 0
        if sr.start_frame is not None and sr.end_frame is not None:
            for r in free:
                if r.start_frame is None or r.end_frame is None:
                    continue
                ov = min(sr.end_frame, r.end_frame) - max(sr.start_frame, r.start_frame) + 1
                if ov > best_ov:
                    best, best_ov = r, ov
        if best is None:
            # No current reach shares a frame. Kept as its own row only when
            # a person touched it (determined a frame, excluded it or wrote a
            # comment); an untouched seed of an older run is the algorithm's
            # old guess and is not carried forward.
            if (sr.start_determined or sr.end_determined or sr.exclude_from_analysis
                    or sr.comment):
                extra.append(sr)
            continue
        r = best
        free.remove(r)
        r.start_determined = sr.start_determined
        r.start_determined_by = sr.start_determined_by
        r.start_determined_at = sr.start_determined_at
        r.end_determined = sr.end_determined
        r.end_determined_by = sr.end_determined_by
        r.end_determined_at = sr.end_determined_at
        # If determined, use the saved frame values
        if sr.start_determined:
            r.start_frame = sr.start_frame
        if sr.end_determined:
            r.end_frame = sr.end_frame
        r.exclude_from_analysis = sr.exclude_from_analysis
        r.exclude_reason = sr.exclude_reason
        r.comment = sr.comment
    taken = {r.reach_id for r in algo_gt.reaches}
    next_id = 1 + max([r.reach_id for r in algo_gt.reaches + extra if r.reach_id is not None] + [0])
    for sr in extra:
        if sr.reach_id in taken:
            sr.reach_id = next_id
            next_id += 1
        taken.add(sr.reach_id)
        algo_gt.reaches.append(sr)

    # Sort reaches by segment then start_frame for consistent display order
    algo_gt.reaches.sort(key=lambda r: (r.segment_num, r.start_frame))

    # Outcomes: a saved outcome is put back on its pellet only when that
    # pellet's frames in the SAVED file's own cuts are exactly the frames of
    # the same-numbered pellet now. WHY: after a re-cut, the saved "pellet 7"
    # is other footage than today's pellet 7, and an outcome a person
    # determined landed on a pellet nobody judged -- and was saved there as
    # theirs. A re-cut voids the answers given on the old cuts; they are not
    # moved onto new pellets either. Such an outcome is kept in
    # unmatched_outcomes (written back on save, never applied, listed when the
    # file is opened); an earlier save's unmatched answers are carried along
    # the same way. A file that records no cuts for a pellet (an older file,
    # or one saved with only some cuts) is matched by number, the only handle
    # it has.
    from mousereach.review.causal_review_io import gt_segments
    saved_spans = {s_["segment_num"]: (s_["start_frame"], s_["end_frame"])
                   for s_ in gt_segments(_unified_gt_to_dict(saved_gt))}
    current_spans = {n: (a, b) for n, a, b in _segments_of(algo_gt)}
    by_seg = {}
    for so in saved_gt.outcomes:
        own = saved_spans.get(so.segment_num)
        if own is None or current_spans.get(so.segment_num) == own:
            by_seg[so.segment_num] = so
    placed = set()
    cur_outcomes = {o.segment_num: o for o in algo_gt.outcomes}
    for sn, so in by_seg.items():
        o = cur_outcomes.get(int(sn))
        if o is None:
            continue
        placed.add(id(so))
        o.determined = so.determined
        o.determined_by = so.determined_by
        o.determined_at = so.determined_at
        # If determined, use the saved outcome value
        if so.determined:
            o.outcome = so.outcome
        if so.interaction_frame is not None:
            o.interaction_frame = so.interaction_frame
        if so.outcome_known_frame is not None:
            o.outcome_known_frame = so.outcome_known_frame
        if so.causal_reach_id is not None:
            o.causal_reach_id = so.causal_reach_id
        o.comment = so.comment
    algo_gt.unmatched_outcomes = list(saved_gt.unmatched_outcomes) + [
        so for so in saved_gt.outcomes if id(so) not in placed and so.determined]
    for so in algo_gt.unmatched_outcomes:
        logger.warning("ground truth %s: saved outcome for pellet %s (%s) matches no pellet of "
                       "the current cuts; kept unapplied", algo_gt.video_name,
                       so.segment_num, so.outcome)

    algo_gt.update_completion_status()


def _overlay_old_gt_corrections(gt: UnifiedGroundTruth, video_path: Path):
    """Overlay corrections from old separate GT files, mapping to v2 fields."""
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")
    parent = video_path.parent

    # Load old seg GT and apply boundary corrections
    seg_gt_path = parent / f"{video_stem}_seg_ground_truth.json"
    if seg_gt_path.exists():
        try:
            with open(seg_gt_path) as f:
                seg_data = json.load(f)
            gt_boundaries = seg_data.get("boundaries", [])
            # Mark boundaries as determined if they exist in old GT
            for i, b in enumerate(gt.boundaries):
                if i < len(gt_boundaries):
                    if b.frame != gt_boundaries[i]:
                        b.frame = gt_boundaries[i]
                    b.determined = True
                    b.determined_by = seg_data.get("created_by", "unknown")
                    b.determined_at = seg_data.get("created_at")
        except Exception as err:
            logger.warning('Could not read %s, so its boundary corrections were not applied: %s', seg_gt_path, err)
            pass

    # Load old reach GT and apply reach corrections
    reach_gt_path = parent / f"{video_stem}_reach_ground_truth.json"
    if reach_gt_path.exists():
        try:
            with open(reach_gt_path) as f:
                reach_data = json.load(f)
            # Apply reach corrections from GT (structure varies)
            # Old format had segments[].reaches[]
            for seg in reach_data.get("segments", []):
                for r_gt in seg.get("reaches", []):
                    reach_id = r_gt.get("reach_id")
                    for r in gt.reaches:
                        if r.reach_id == reach_id:
                            if r_gt.get("human_corrected"):
                                if r_gt.get("start_frame") != r.start_frame:
                                    r.start_frame = r_gt.get("start_frame")
                                if r_gt.get("end_frame") != r.end_frame:
                                    r.end_frame = r_gt.get("end_frame")
                            r.start_determined = True
                            r.end_determined = True
                            r.exclude_from_analysis = r_gt.get("exclude_from_analysis", False)
                            r.exclude_reason = r_gt.get("exclude_reason")
                            break
        except Exception as err:
            logger.warning('Could not read %s, so its reach corrections were not applied: %s', reach_gt_path, err)
            pass

    # Load old outcome GT and apply corrections
    outcome_gt_paths = [
        parent / f"{video_stem}_outcome_ground_truth.json",
        parent / f"{video_stem}_outcomes_ground_truth.json",
    ]
    for outcome_gt_path in outcome_gt_paths:
        if outcome_gt_path.exists():
            try:
                with open(outcome_gt_path) as f:
                    outcome_data = json.load(f)
                for seg_gt in outcome_data.get("segments", []):
                    seg_num = seg_gt.get("segment_num")
                    for o in gt.outcomes:
                        if o.segment_num == seg_num:
                            gt_outcome = seg_gt.get("outcome")
                            if gt_outcome and gt_outcome != o.outcome:
                                o.outcome = gt_outcome
                            o.determined = True
                            o.determined_by = outcome_data.get("created_by", "unknown")
                            o.determined_at = outcome_data.get("created_at")
                            if seg_gt.get("interaction_frame") is not None:
                                o.interaction_frame = seg_gt.get("interaction_frame")
                            if seg_gt.get("outcome_known_frame") is not None:
                                o.outcome_known_frame = seg_gt.get("outcome_known_frame")
                            break
            except Exception as err:
                logger.warning('Could not read %s, so its outcome corrections were not applied: %s', outcome_gt_path, err)
                pass
            break

    gt.update_completion_status()
