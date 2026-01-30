"""
Unified Ground Truth File Handler
=================================

Single unified GT file format that combines segmentation, reaches, and outcomes.

Features:
- Split verification for reaches (start_verified, end_verified independent)
- Per-item verification tracking with timestamps
- Completion status computation
- Migration from old separate GT files

File pattern: *_unified_ground_truth.json
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict


# Schema version for forward compatibility
SCHEMA_VERSION = "1.0"


@dataclass
class BoundaryGT:
    """Ground truth for a single segment boundary."""
    index: int
    frame: int
    verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    corrected: bool = False
    original_frame: Optional[int] = None
    source: str = "algorithm"  # "algorithm" = from algo output, "human_added" = manually added
    comment: Optional[str] = None  # Free-form notes about this boundary


@dataclass
class ReachGT:
    """Ground truth for a single reach with split start/end verification."""
    # Required fields first
    reach_id: int
    segment_num: int
    start_frame: int
    end_frame: int
    # Optional fields with defaults
    # Start boundary verification
    start_verified: bool = False
    start_verified_by: Optional[str] = None
    start_verified_at: Optional[str] = None
    start_corrected: bool = False
    original_start_frame: Optional[int] = None
    # Apex (informational)
    apex_frame: Optional[int] = None
    # End boundary verification
    end_verified: bool = False
    end_verified_by: Optional[str] = None
    end_verified_at: Optional[str] = None
    end_corrected: bool = False
    original_end_frame: Optional[int] = None
    # Metadata
    source: str = "algorithm"  # "algorithm" or "human_added"
    exclude_from_analysis: bool = False
    exclude_reason: Optional[str] = None
    comment: Optional[str] = None  # Free-form notes about this reach

    @property
    def fully_verified(self) -> bool:
        """A reach is fully verified when both start AND end are verified."""
        return self.start_verified and self.end_verified


@dataclass
class OutcomeGT:
    """Ground truth for a single segment outcome."""
    segment_num: int
    outcome: str  # "retrieved", "displaced_sa", "displaced_outside", "untouched", etc.
    verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    corrected: bool = False
    original_outcome: Optional[str] = None
    # Key frames
    interaction_frame: Optional[int] = None
    outcome_known_frame: Optional[int] = None
    causal_reach_id: Optional[int] = None
    # Algorithm metadata
    confidence: float = 0.0
    flagged: bool = False
    flag_reason: Optional[str] = None
    source: str = "algorithm"  # "algorithm" = from algo output, "human_added" = manually added
    comment: Optional[str] = None  # Free-form notes about this outcome


@dataclass
class Flag:
    """A flag indicating something needs attention."""
    type: str  # "boundary", "reach", "outcome"
    id: int  # boundary index, reach_id, or segment_num
    reason: str


@dataclass
class CompletionStatus:
    """Tracks completion of all verification tasks."""
    segments_complete: bool = False
    reaches_complete: bool = False
    outcomes_complete: bool = False
    all_complete: bool = False
    flags: List[Flag] = field(default_factory=list)

    # Counts for progress display
    boundaries_verified: int = 0
    boundaries_total: int = 0
    reaches_verified: int = 0  # Fully verified (both start AND end)
    reaches_total: int = 0
    outcomes_verified: int = 0
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
    anomalies: List[str] = field(default_factory=list)
    anomaly_annotations: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Reach data
    reaches: List[ReachGT] = field(default_factory=list)

    # Outcome data
    outcomes: List[OutcomeGT] = field(default_factory=list)

    def update_completion_status(self) -> CompletionStatus:
        """Recompute completion status from current data."""
        status = CompletionStatus()

        # Boundaries
        status.boundaries_total = len(self.boundaries)
        status.boundaries_verified = sum(1 for b in self.boundaries if b.verified)
        status.segments_complete = (
            status.boundaries_verified == status.boundaries_total
            and status.boundaries_total > 0
        )

        # Reaches - must have BOTH start and end verified
        status.reaches_total = len(self.reaches)
        status.reaches_verified = sum(1 for r in self.reaches if r.fully_verified)
        status.reaches_complete = (
            status.reaches_verified == status.reaches_total
            and status.reaches_total > 0
        )

        # Outcomes
        status.outcomes_total = len(self.outcomes)
        status.outcomes_verified = sum(1 for o in self.outcomes if o.verified)
        status.outcomes_complete = (
            status.outcomes_verified == status.outcomes_total
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

        # Unverified boundaries
        for b in self.boundaries:
            if not b.verified:
                status.flags.append(Flag("boundary", b.index, "unverified"))

        # Partially or unverified reaches
        for r in self.reaches:
            if not r.start_verified:
                status.flags.append(Flag("reach", r.reach_id, "start unverified"))
            if not r.end_verified:
                status.flags.append(Flag("reach", r.reach_id, "end unverified"))

        # Unverified outcomes
        for o in self.outcomes:
            if not o.verified:
                reason = "unverified"
                if o.flagged:
                    reason = f"flagged: {o.flag_reason or 'needs review'}"
                status.flags.append(Flag("outcome", o.segment_num, reason))

        self.completion_status = status
        return status


def get_username() -> str:
    """Get current username for verification tracking."""
    return os.environ.get("USERNAME", os.environ.get("USER", "unknown"))


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def get_unified_gt_path(video_path: Path) -> Path:
    """Get the unified GT file path for a video."""
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")
    return video_path.parent / f"{video_stem}_unified_ground_truth.json"


def load_unified_gt(video_path: Path) -> Optional[UnifiedGroundTruth]:
    """
    Load unified ground truth file for a video.

    Returns None if file doesn't exist.
    """
    gt_path = get_unified_gt_path(video_path)
    if not gt_path.exists():
        return None

    try:
        with open(gt_path, "r") as f:
            data = json.load(f)
        return _dict_to_unified_gt(data)
    except Exception as e:
        print(f"Error loading unified GT: {e}")
        return None


def save_unified_gt(gt: UnifiedGroundTruth, video_path: Path) -> Path:
    """
    Save unified ground truth to file.

    Updates timestamps and completion status before saving.
    """
    gt_path = get_unified_gt_path(video_path)

    # Update metadata
    gt.last_modified_at = get_timestamp()
    gt.last_modified_by = get_username()
    if not gt.created_at:
        gt.created_at = gt.last_modified_at
        gt.created_by = gt.last_modified_by

    # Update completion status
    gt.update_completion_status()

    # Convert to dict and save
    data = _unified_gt_to_dict(gt)
    with open(gt_path, "w") as f:
        json.dump(data, f, indent=2)

    return gt_path


def _dict_to_unified_gt(data: Dict) -> UnifiedGroundTruth:
    """Convert dict from JSON to UnifiedGroundTruth dataclass."""
    gt = UnifiedGroundTruth(
        video_name=data.get("video_name", ""),
        type=data.get("type", "unified_ground_truth"),
        schema_version=data.get("schema_version", "1.0"),
        created_by=data.get("created_by", ""),
        created_at=data.get("created_at", ""),
        last_modified_at=data.get("last_modified_at", ""),
        last_modified_by=data.get("last_modified_by", ""),
        anomalies=data.get("segmentation", {}).get("anomalies", []),
        anomaly_annotations=data.get("segmentation", {}).get("anomaly_annotations", {}),
    )

    # Parse boundaries
    for b_data in data.get("segmentation", {}).get("boundaries", []):
        gt.boundaries.append(BoundaryGT(
            index=b_data.get("index", 0),
            frame=b_data.get("frame", 0),
            verified=b_data.get("verified", False),
            verified_by=b_data.get("verified_by"),
            verified_at=b_data.get("verified_at"),
            corrected=b_data.get("corrected", False),
            original_frame=b_data.get("original_frame"),
            source=b_data.get("source", "algorithm"),
        ))

    # Parse reaches
    for r_data in data.get("reaches", {}).get("reaches", []):
        gt.reaches.append(ReachGT(
            reach_id=r_data.get("reach_id", 0),
            segment_num=r_data.get("segment_num", 0),
            start_frame=r_data.get("start_frame", 0),
            start_verified=r_data.get("start_verified", False),
            start_verified_by=r_data.get("start_verified_by"),
            start_verified_at=r_data.get("start_verified_at"),
            start_corrected=r_data.get("start_corrected", False),
            original_start_frame=r_data.get("original_start_frame"),
            apex_frame=r_data.get("apex_frame"),
            end_frame=r_data.get("end_frame", 0),
            end_verified=r_data.get("end_verified", False),
            end_verified_by=r_data.get("end_verified_by"),
            end_verified_at=r_data.get("end_verified_at"),
            end_corrected=r_data.get("end_corrected", False),
            original_end_frame=r_data.get("original_end_frame"),
            source=r_data.get("source", "algorithm"),
            exclude_from_analysis=r_data.get("exclude_from_analysis", False),
            exclude_reason=r_data.get("exclude_reason"),
        ))

    # Sort reaches by segment then start_frame for consistent display order
    gt.reaches.sort(key=lambda r: (r.segment_num, r.start_frame))

    # Parse outcomes
    for o_data in data.get("outcomes", {}).get("segments", []):
        gt.outcomes.append(OutcomeGT(
            segment_num=o_data.get("segment_num", 0),
            outcome=o_data.get("outcome", ""),
            verified=o_data.get("verified", False),
            verified_by=o_data.get("verified_by"),
            verified_at=o_data.get("verified_at"),
            corrected=o_data.get("corrected", False),
            original_outcome=o_data.get("original_outcome"),
            interaction_frame=o_data.get("interaction_frame"),
            outcome_known_frame=o_data.get("outcome_known_frame"),
            causal_reach_id=o_data.get("causal_reach_id"),
            confidence=o_data.get("confidence", 0.0),
            flagged=o_data.get("flagged", False),
            flag_reason=o_data.get("flag_reason"),
            source=o_data.get("source", "algorithm"),
        ))

    # Update completion status
    gt.update_completion_status()

    return gt


def _unified_gt_to_dict(gt: UnifiedGroundTruth) -> Dict:
    """Convert UnifiedGroundTruth to dict for JSON serialization."""
    return {
        "video_name": gt.video_name,
        "type": gt.type,
        "schema_version": gt.schema_version,
        "created_by": gt.created_by,
        "created_at": gt.created_at,
        "last_modified_at": gt.last_modified_at,
        "last_modified_by": gt.last_modified_by,
        "completion_status": {
            "segments_complete": gt.completion_status.segments_complete,
            "reaches_complete": gt.completion_status.reaches_complete,
            "outcomes_complete": gt.completion_status.outcomes_complete,
            "all_complete": gt.completion_status.all_complete,
            "boundaries_verified": gt.completion_status.boundaries_verified,
            "boundaries_total": gt.completion_status.boundaries_total,
            "reaches_verified": gt.completion_status.reaches_verified,
            "reaches_total": gt.completion_status.reaches_total,
            "outcomes_verified": gt.completion_status.outcomes_verified,
            "outcomes_total": gt.completion_status.outcomes_total,
            "flags": [
                {"type": f.type, "id": f.id, "reason": f.reason}
                for f in gt.completion_status.flags
            ],
        },
        "segmentation": {
            "n_boundaries": len(gt.boundaries),
            "boundaries": [
                {
                    "index": b.index,
                    "frame": b.frame,
                    "verified": b.verified,
                    "verified_by": b.verified_by,
                    "verified_at": b.verified_at,
                    "corrected": b.corrected,
                    "original_frame": b.original_frame,
                    "source": b.source,
                }
                for b in gt.boundaries
            ],
            "anomalies": gt.anomalies,
            "anomaly_annotations": gt.anomaly_annotations,
        },
        "reaches": {
            "total_reaches": len(gt.reaches),
            "reaches": [
                {
                    "reach_id": r.reach_id,
                    "segment_num": r.segment_num,
                    "start_frame": r.start_frame,
                    "start_verified": r.start_verified,
                    "start_verified_by": r.start_verified_by,
                    "start_verified_at": r.start_verified_at,
                    "start_corrected": r.start_corrected,
                    "original_start_frame": r.original_start_frame,
                    "apex_frame": r.apex_frame,
                    "end_frame": r.end_frame,
                    "end_verified": r.end_verified,
                    "end_verified_by": r.end_verified_by,
                    "end_verified_at": r.end_verified_at,
                    "end_corrected": r.end_corrected,
                    "original_end_frame": r.original_end_frame,
                    "source": r.source,
                    "exclude_from_analysis": r.exclude_from_analysis,
                    "exclude_reason": r.exclude_reason,
                }
                for r in gt.reaches
            ],
        },
        "outcomes": {
            "n_segments": len(gt.outcomes),
            "segments": [
                {
                    "segment_num": o.segment_num,
                    "outcome": o.outcome,
                    "verified": o.verified,
                    "verified_by": o.verified_by,
                    "verified_at": o.verified_at,
                    "corrected": o.corrected,
                    "original_outcome": o.original_outcome,
                    "interaction_frame": o.interaction_frame,
                    "outcome_known_frame": o.outcome_known_frame,
                    "causal_reach_id": o.causal_reach_id,
                    "confidence": o.confidence,
                    "flagged": o.flagged,
                    "flag_reason": o.flag_reason,
                    "source": o.source,
                }
                for o in gt.outcomes
            ],
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
        # No reach GT - load from algorithm output (unverified)
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
    """Migrate old segmentation GT format to unified."""
    boundaries = seg_data.get("boundaries", [])
    boundaries_meta = seg_data.get("boundaries_with_meta", [])

    # If we have metadata, use it
    if boundaries_meta:
        for b_meta in boundaries_meta:
            gt.boundaries.append(BoundaryGT(
                index=b_meta.get("index", len(gt.boundaries)),
                frame=b_meta.get("frame", 0),
                verified=b_meta.get("human_verified", False),
                verified_by=b_meta.get("verified_by"),
                verified_at=b_meta.get("verified_at"),
                corrected=b_meta.get("human_corrected", False),
                original_frame=b_meta.get("original_frame"),
            ))
    else:
        # Just frame numbers - assume all verified if GT file exists
        for i, frame in enumerate(boundaries):
            gt.boundaries.append(BoundaryGT(
                index=i,
                frame=frame,
                verified=True,  # Assume verified since it's in GT file
            ))

    gt.anomalies = seg_data.get("anomalies", [])
    gt.anomaly_annotations = seg_data.get("anomaly_annotations", {})


def _migrate_reach_gt(gt: UnifiedGroundTruth, reach_data: Dict):
    """Migrate old reach GT format to unified with split verification."""
    for seg in reach_data.get("segments", []):
        for r in seg.get("reaches", []):
            # Old format has human_verified per-reach, not split
            # We'll mark both start and end as verified if the reach was verified
            old_verified = r.get("human_verified", False)

            gt.reaches.append(ReachGT(
                reach_id=r.get("reach_id", 0),
                segment_num=seg.get("segment_num", 0),
                start_frame=r.get("start_frame", 0),
                start_verified=old_verified,
                start_verified_by=r.get("verified_by"),
                start_verified_at=r.get("verified_at"),
                start_corrected=r.get("human_corrected", False),
                original_start_frame=r.get("original_start"),
                apex_frame=r.get("apex_frame"),
                end_frame=r.get("end_frame", 0),
                end_verified=old_verified,
                end_verified_by=r.get("verified_by"),
                end_verified_at=r.get("verified_at"),
                end_corrected=r.get("human_corrected", False),
                original_end_frame=r.get("original_end"),
                source=r.get("source", "algorithm"),
                exclude_from_analysis=r.get("exclude_from_analysis", False),
                exclude_reason=r.get("exclude_reason"),
            ))


def _migrate_outcome_gt(gt: UnifiedGroundTruth, outcome_data: Dict):
    """Migrate old outcome GT format to unified."""
    for seg in outcome_data.get("segments", []):
        gt.outcomes.append(OutcomeGT(
            segment_num=seg.get("segment_num", 0),
            outcome=seg.get("outcome", ""),
            verified=seg.get("human_verified", False),
            verified_by=seg.get("verified_by"),
            verified_at=seg.get("verified_at"),
            corrected=seg.get("human_corrected", False) or (
                seg.get("original_outcome") is not None
                and seg.get("original_outcome") != seg.get("outcome")
            ),
            original_outcome=seg.get("original_outcome"),
            interaction_frame=seg.get("interaction_frame"),
            outcome_known_frame=seg.get("outcome_known_frame"),
            causal_reach_id=seg.get("causal_reach_id"),
            confidence=seg.get("confidence", 0.0),
            flagged=seg.get("flagged_for_review", False),
            flag_reason=seg.get("flag_reason"),
        ))


# =============================================================================
# Initialize from Algorithm Output
# =============================================================================


def create_from_algorithm_output(video_path: Path) -> Optional[UnifiedGroundTruth]:
    """
    Create a new unified GT from algorithm output files.

    Loads data from:
    - *_segments.json or *_seg_validation.json
    - *_reaches.json
    - *_pellet_outcomes.json

    All items start unverified.
    """
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")

    parent = video_path.parent

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
                    verified=False,
                    original_outcome=None,
                    confidence=0.0,
                    source="placeholder",  # Indicates no algorithm data
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
                        verified=False,
                        original_frame=frame,
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
                            start_verified=False,
                            original_start_frame=r.get("start_frame"),
                            apex_frame=r.get("apex_frame"),
                            end_frame=r.get("end_frame", 0),
                            end_verified=False,
                            original_end_frame=r.get("end_frame"),
                            source="algorithm",
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
                        outcome=seg.get("outcome", ""),
                        verified=False,
                        original_outcome=seg.get("outcome"),
                        interaction_frame=seg.get("interaction_frame"),
                        outcome_known_frame=seg.get("outcome_known_frame"),
                        confidence=seg.get("confidence", 0.0),
                        flagged=seg.get("flagged_for_review", False),
                        flag_reason=seg.get("flag_reason"),
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


def load_or_create_unified_gt(video_path: Path) -> UnifiedGroundTruth:
    """
    Load algorithm outputs, then overlay any existing GT corrections/verifications.

    The review tool reviews ALGORITHM OUTPUTS, not GT files.
    GT is the OUTPUT of the review process.

    Flow:
    1. ALWAYS load algorithm outputs first (segments, reaches, outcomes)
    2. If unified GT exists, overlay verification status and corrections
    3. If old GT files exist, overlay their corrections
    4. Return combined result
    """
    video_stem = video_path.stem.replace("_preview", "")
    if "DLC" in video_stem:
        video_stem = video_stem.split("DLC")[0].rstrip("_")

    # Step 1: Load algorithm outputs (the data we're reviewing)
    gt = create_from_algorithm_output(video_path)
    if gt is None:
        # No algorithm output - try unified GT or old GTs as fallback
        gt = load_unified_gt(video_path)
        if gt is not None:
            return gt
        gt = migrate_from_old_formats(video_path)
        if gt is not None:
            return gt
        # Return empty GT
        return UnifiedGroundTruth(video_name=video_stem)

    # Step 2: If unified GT exists, overlay verification status
    existing_gt = load_unified_gt(video_path)
    if existing_gt is not None:
        _overlay_gt_verifications(gt, existing_gt)
        return gt

    # Step 3: If old GT files exist, overlay their corrections
    _overlay_old_gt_corrections(gt, video_path)

    return gt


def _overlay_gt_verifications(algo_gt: UnifiedGroundTruth, saved_gt: UnifiedGroundTruth):
    """Overlay verification status from saved GT onto algorithm GT."""
    # Boundaries: match by index and apply verification
    saved_bounds = {b.index: b for b in saved_gt.boundaries}
    for b in algo_gt.boundaries:
        if b.index in saved_bounds:
            sb = saved_bounds[b.index]
            b.verified = sb.verified
            b.verified_by = sb.verified_by
            b.verified_at = sb.verified_at
            if sb.corrected:
                b.frame = sb.frame
                b.corrected = True
                b.original_frame = sb.original_frame

    # Reaches: match by reach_id and apply verification
    saved_reaches = {r.reach_id: r for r in saved_gt.reaches}
    for r in algo_gt.reaches:
        if r.reach_id in saved_reaches:
            sr = saved_reaches[r.reach_id]
            r.start_verified = sr.start_verified
            r.start_verified_by = sr.start_verified_by
            r.start_verified_at = sr.start_verified_at
            r.end_verified = sr.end_verified
            r.end_verified_by = sr.end_verified_by
            r.end_verified_at = sr.end_verified_at
            if sr.start_corrected:
                r.start_frame = sr.start_frame
                r.start_corrected = True
                r.original_start_frame = sr.original_start_frame
            if sr.end_corrected:
                r.end_frame = sr.end_frame
                r.end_corrected = True
                r.original_end_frame = sr.original_end_frame
            r.exclude_from_analysis = sr.exclude_from_analysis
            r.exclude_reason = sr.exclude_reason

    # Also add any human-added reaches from saved GT
    algo_reach_ids = {r.reach_id for r in algo_gt.reaches}
    for sr in saved_gt.reaches:
        if sr.reach_id not in algo_reach_ids and sr.source == "human_added":
            algo_gt.reaches.append(sr)

    # Sort reaches by segment then start_frame for consistent display order
    algo_gt.reaches.sort(key=lambda r: (r.segment_num, r.start_frame))

    # Outcomes: match by segment_num and apply verification
    saved_outcomes = {o.segment_num: o for o in saved_gt.outcomes}
    for o in algo_gt.outcomes:
        if o.segment_num in saved_outcomes:
            so = saved_outcomes[o.segment_num]
            o.verified = so.verified
            o.verified_by = so.verified_by
            o.verified_at = so.verified_at
            if so.corrected:
                o.outcome = so.outcome
                o.corrected = True
                o.original_outcome = so.original_outcome
            if so.interaction_frame is not None:
                o.interaction_frame = so.interaction_frame
            if so.outcome_known_frame is not None:
                o.outcome_known_frame = so.outcome_known_frame
            if so.causal_reach_id is not None:
                o.causal_reach_id = so.causal_reach_id

    algo_gt.update_completion_status()


def _overlay_old_gt_corrections(gt: UnifiedGroundTruth, video_path: Path):
    """Overlay corrections from old separate GT files."""
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
            # Mark boundaries as verified if they match GT
            for i, b in enumerate(gt.boundaries):
                if i < len(gt_boundaries):
                    if b.frame != gt_boundaries[i]:
                        b.original_frame = b.frame
                        b.frame = gt_boundaries[i]
                        b.corrected = True
                    b.verified = True
                    b.verified_by = seg_data.get("created_by", "unknown")
                    b.verified_at = seg_data.get("created_at")
        except Exception:
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
                                    r.original_start_frame = r.start_frame
                                    r.start_frame = r_gt.get("start_frame")
                                    r.start_corrected = True
                                if r_gt.get("end_frame") != r.end_frame:
                                    r.original_end_frame = r.end_frame
                                    r.end_frame = r_gt.get("end_frame")
                                    r.end_corrected = True
                            r.start_verified = True
                            r.end_verified = True
                            r.exclude_from_analysis = r_gt.get("exclude_from_analysis", False)
                            r.exclude_reason = r_gt.get("exclude_reason")
                            break
        except Exception:
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
                                o.original_outcome = o.outcome
                                o.outcome = gt_outcome
                                o.corrected = True
                            o.verified = True
                            o.verified_by = outcome_data.get("created_by", "unknown")
                            o.verified_at = outcome_data.get("created_at")
                            if seg_gt.get("interaction_frame") is not None:
                                o.interaction_frame = seg_gt.get("interaction_frame")
                            if seg_gt.get("outcome_known_frame") is not None:
                                o.outcome_known_frame = seg_gt.get("outcome_known_frame")
                            break
            except Exception:
                pass
            break

    gt.update_completion_status()
