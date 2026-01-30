"""
Pipeline Core - Orchestration logic for the unified analysis pipeline.

Handles the flow of videos through all three processing stages:
1. Segmentation → auto-approved continue, needs-review pause
2. Outcome Detection → all continue
3. Reach Detection → validated complete, anomalies pause
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime
import json
import shutil


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_associated_files(parent_dir: Path, video_id: str) -> List[Path]:
    """Get ALL files associated with a video (everything with video_id prefix)."""
    files = []
    for f in parent_dir.iterdir():
        if f.is_file() and f.name.startswith(video_id):
            files.append(f)
    return files


def has_complete_gt(folder: Path, video_id: str, gt_type: str) -> bool:
    """
    Check if a complete ground truth file exists for this video.

    Complete GT means a human has reviewed the entire video - no further review needed.

    Args:
        folder: Directory containing the files
        video_id: Video identifier
        gt_type: One of 'seg', 'reach', 'outcome'

    Returns:
        True if complete GT exists
    """
    gt_suffixes = {
        'seg': '_seg_ground_truth.json',
        'reach': '_reach_ground_truth.json',
        'outcome': '_outcome_ground_truth.json',
    }
    suffix = gt_suffixes.get(gt_type)
    if not suffix:
        return False

    gt_path = folder / f"{video_id}{suffix}"
    if not gt_path.exists():
        return False

    try:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        # Check explicit gt_complete field, or infer from type="ground_truth"
        if gt_data.get("gt_complete"):
            return True
        if gt_data.get("type") == "ground_truth":
            return True
        return False
    except (json.JSONDecodeError, OSError):
        return False


def count_gt_covered_outcome_segments(outcome_path: Path, gt_path: Path) -> int:
    """
    Count outcome segments covered by GT (don't need review).

    Does NOT modify any files - raw algorithm output preserved for comparison.

    Args:
        outcome_path: Path to algorithm outcome results (_pellet_outcomes.json)
        gt_path: Path to ground truth file (_outcome_ground_truth.json)

    Returns:
        Number of segments that have GT coverage
    """
    if not outcome_path.exists() or not gt_path.exists():
        return 0

    try:
        with open(outcome_path, 'r') as f:
            outcomes = json.load(f)
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0

    # Build lookup of GT segments by segment_num
    gt_segments = {seg['segment_num']: seg for seg in gt_data.get('segments', [])}

    covered = 0
    for seg in outcomes.get('segments', []):
        seg_num = seg.get('segment_num')
        gt_seg = gt_segments.get(seg_num)

        # Segment is covered if GT has human-verified answer
        if gt_seg and gt_seg.get('human_verified', False):
            covered += 1

    return covered


def count_gt_covered_reach_segments(reach_path: Path, gt_path: Path) -> int:
    """
    Count reach segments covered by GT (don't need review).

    Does NOT modify any files - raw algorithm output preserved for comparison.

    Args:
        reach_path: Path to algorithm reach results (_reaches.json)
        gt_path: Path to ground truth file (_reach_ground_truth.json)

    Returns:
        Number of segments that have GT coverage
    """
    if not reach_path.exists() or not gt_path.exists():
        return 0

    try:
        with open(reach_path, 'r') as f:
            reaches = json.load(f)
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0

    # Build lookup of GT segments by segment_num
    gt_segments = {seg['segment_num']: seg for seg in gt_data.get('segments', [])}

    covered = 0
    for seg in reaches.get('segments', []):
        seg_num = seg.get('segment_num')
        gt_seg = gt_segments.get(seg_num)

        if gt_seg:
            # Segment is covered if GT has human-corrected reaches
            gt_reaches = gt_seg.get('reaches', [])
            if any(r.get('human_corrected', False) for r in gt_reaches):
                covered += 1

    return covered


def move_files_to_folder(files: List[Path], dest_folder: Path) -> None:
    """Move files to destination folder."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = dest_folder / f.name
        if f.exists() and f != dest:
            shutil.move(str(f), str(dest))


def find_dlc_files(input_dir: Path) -> List[Path]:
    """Find all DLC .h5 files in directory."""
    return sorted(input_dir.glob("*DLC*.h5"))


def extract_video_id(dlc_path: Path) -> str:
    """Extract video ID from DLC filename."""
    return dlc_path.stem.split("DLC")[0].rstrip('_')


@dataclass
class PipelineStatus:
    """Current state of the pipeline - what's ready to process.

    Unified architecture: All files in Processing/, status tracked by JSON metadata.
    """
    # Files needing segmentation (have DLC but no _segments.json)
    dlc_ready: int = 0
    dlc_files: List[Path] = field(default_factory=list)

    # Files needing outcomes/reaches (have segments but no outputs)
    seg_validated_ready: int = 0
    seg_validated_files: List[Path] = field(default_factory=list)

    # Legacy fields kept for compatibility but not used in unified architecture
    reach_validated_ready: int = 0
    reach_validated_files: List[Path] = field(default_factory=list)
    outcome_validated_ready: int = 0
    outcome_validated_files: List[Path] = field(default_factory=list)

    @property
    def total_ready(self) -> int:
        return self.dlc_ready + self.seg_validated_ready

    def summary_lines(self) -> List[str]:
        """Get human-readable summary lines."""
        lines = []
        if self.dlc_ready > 0:
            lines.append(f"{self.dlc_ready} need segmentation")
        if self.seg_validated_ready > 0:
            lines.append(f"{self.seg_validated_ready} need outcomes/reaches")
        if not lines:
            lines.append("Nothing ready to process")
        return lines


def count_all_videos_in_pipeline(base_dir: Path) -> Dict:
    """
    Count all videos in the pipeline.

    Unified architecture: All videos are in Processing/ (or Failed/).
    Returns dict with folder counts and total.
    """
    counts = {}
    video_ids = set()

    # Processing folder (main location for all videos)
    processing = base_dir / "Processing"
    if processing.exists():
        processing_videos = set()
        for dlc_path in processing.glob("*DLC*.h5"):
            video_id = extract_video_id(dlc_path)
            processing_videos.add(video_id)
            video_ids.add(video_id)
        counts['Processing'] = len(processing_videos)
    else:
        counts['Processing'] = 0

    # Failed folder (videos with processing errors)
    failed = base_dir / "Failed"
    if failed.exists():
        failed_videos = set()
        for dlc_path in failed.glob("*DLC*.h5"):
            video_id = extract_video_id(dlc_path)
            failed_videos.add(video_id)
            video_ids.add(video_id)
        counts['Failed'] = len(failed_videos)
    else:
        counts['Failed'] = 0

    # DLC_Queue (videos waiting for DLC processing)
    dlc_queue = base_dir / "DLC_Queue"
    if dlc_queue.exists():
        queue_videos = set()
        for video_path in dlc_queue.glob("*.mp4"):
            video_id = video_path.stem
            queue_videos.add(video_id)
            video_ids.add(video_id)
        counts['DLC_Queue'] = len(queue_videos)
    else:
        counts['DLC_Queue'] = 0

    counts['total'] = len(video_ids)
    return counts


@dataclass
class UnifiedResults:
    """Summary of unified pipeline run."""
    # What we started with
    status_before: Optional[PipelineStatus] = None

    # Stage 1: Segmentation results
    seg_processed: int = 0
    seg_auto_approved: int = 0
    seg_needs_review: int = 0
    seg_failed: int = 0

    # Stage 2: Outcome detection results
    outcome_processed: int = 0
    outcome_completed: int = 0
    outcome_needs_review: int = 0
    outcome_failed: int = 0

    # Stage 3: Reach detection results
    reach_processed: int = 0
    reach_validated: int = 0
    reach_needs_review: int = 0
    reach_failed: int = 0

    # Advancing validated files
    reach_advanced: int = 0
    outcome_advanced: int = 0

    # Final counts
    fully_completed: int = 0
    paused_for_review: int = 0
    failed: int = 0

    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def consolidate_all_to_dlc_complete(base_dir: Path, delete_outputs: bool = True) -> Dict:
    """
    Prepare all videos for reprocessing by deleting derived outputs.

    Unified architecture: Files stay in Processing/, only derived outputs are deleted.

    Args:
        base_dir: Pipeline root (PROCESSING_ROOT)
        delete_outputs: If True, delete _segments.json, _reaches.json, _pellet_outcomes.json

    Returns:
        Dict with counts of files deleted
    """
    stats = {
        'derived_deleted': 0,
        'videos_reset': 0,
    }

    # Derived file patterns to delete
    derived_patterns = [
        "_segments.json",
        "_seg_validation.json",
        "_pellet_outcomes.json",
        "_reaches.json",
        "_grasp_features.json",
    ]

    processing = base_dir / "Processing"
    if not processing.exists():
        return stats

    for dlc_path in processing.glob("*DLC*.h5"):
        video_id = extract_video_id(dlc_path)

        # Delete derived files if requested
        if delete_outputs:
            for suffix in derived_patterns:
                derived = processing / f"{video_id}{suffix}"
                if derived.exists():
                    derived.unlink()
                    stats['derived_deleted'] += 1

        stats['videos_reset'] += 1

    return stats


def scan_pipeline_status(base_dir: Path) -> PipelineStatus:
    """
    Scan the pipeline to determine what's ready to process.

    Unified architecture: All files are in Processing/.
    Status is determined by which output files exist:
    - No _segments.json → needs segmentation
    - Has segments but no outcomes/reaches → needs processing
    - Has all outputs → complete (status in JSON determines review state)
    """
    status = PipelineStatus()

    processing = base_dir / "Processing"
    if not processing.exists():
        return status

    for dlc_path in processing.glob("*DLC*.h5"):
        video_id = extract_video_id(dlc_path)
        seg_path = processing / f"{video_id}_segments.json"
        outcomes_path = processing / f"{video_id}_pellet_outcomes.json"
        reaches_path = processing / f"{video_id}_reaches.json"

        # No segments? Needs segmentation
        if not seg_path.exists():
            status.dlc_files.append(dlc_path)
        # Has segments but no outcomes/reaches? Needs processing
        elif not outcomes_path.exists() or not reaches_path.exists():
            status.seg_validated_files.append(dlc_path)

    status.dlc_ready = len(status.dlc_files)
    status.seg_validated_ready = len(status.seg_validated_files)

    return status


class UnifiedPipelineProcessor:
    """
    Unified processor that handles everything in a single Run action.

    When you click Run:
    1. New DLC files → Segmentation (status tracked in JSON)
    2. Segmented files → Outcomes → Reaches (status tracked in JSON)

    All files stay in Processing/. Status is tracked via validation_status
    field in JSON files, not by folder location.
    """

    def __init__(
        self,
        base_dir: Path,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        specific_files: Optional[List[Path]] = None
    ):
        """
        Args:
            base_dir: Pipeline root (PROCESSING_ROOT)
            progress_callback: Called with (stage_name, current, total, message)
            specific_files: If provided, only process these files (for targeted reprocessing)
        """
        self.base_dir = Path(base_dir)
        self.progress_callback = progress_callback or (lambda *args: None)
        self.specific_files = specific_files

        # Unified architecture: everything stays in Processing/
        # Status is tracked via validation_status field in JSON files, not folder location
        self.processing = self.base_dir / "Processing"
        self.processing.mkdir(parents=True, exist_ok=True)

    def run(self) -> UnifiedResults:
        """Run the unified pipeline - process everything that's ready."""
        results = UnifiedResults()
        results.started_at = datetime.now().isoformat()

        # Scan what's ready
        results.status_before = scan_pipeline_status(self.base_dir)

        # If specific files provided, filter to those
        if self.specific_files:
            self._run_specific_files(results)
        else:
            # Full pipeline run
            self._run_full_pipeline(results)

        # Calculate final counts
        results.fully_completed = results.reach_validated + results.reach_advanced + results.outcome_advanced
        results.paused_for_review = results.seg_needs_review + results.reach_needs_review + results.outcome_needs_review
        results.failed = results.seg_failed + results.outcome_failed + results.reach_failed

        results.completed_at = datetime.now().isoformat()
        return results

    def _run_full_pipeline(self, results: UnifiedResults) -> None:
        """Run the full pipeline on everything that's ready.

        Unified architecture: All files in Processing/, status tracked in JSON.
        Step 1: New DLC files → Segmentation
        Step 2: Segmented files → Outcomes + Reaches
        """
        status = results.status_before

        # Step 1: Process new DLC files through segmentation
        if status.dlc_ready > 0:
            self._process_dlc_complete(status.dlc_files, results)

        # Step 2: Process files that have segments but need outcomes/reaches
        if status.seg_validated_ready > 0:
            self._process_seg_validated(status.seg_validated_files, results)

    def _run_specific_files(self, results: UnifiedResults) -> None:
        """Process only specific files."""
        # Categorize files by what they need
        dlc_files = []
        seg_validated_files = []

        for f in self.specific_files:
            if f.suffix == ".h5" and "DLC" in f.name:
                # DLC file - needs segmentation
                video_id = extract_video_id(f)
                seg_path = f.parent / f"{video_id}_segments.json"
                if not seg_path.exists():
                    dlc_files.append(f)
                else:
                    # Has segments - check if needs outcomes
                    outcome_path = f.parent / f"{video_id}_pellet_outcomes.json"
                    if not outcome_path.exists():
                        seg_validated_files.append(f)

        if dlc_files:
            self._process_dlc_complete(dlc_files, results)

        if seg_validated_files:
            self._process_seg_validated(seg_validated_files, results)

    def _process_dlc_complete(self, dlc_files: List[Path], results: UnifiedResults) -> None:
        """Process DLC files through segmentation.

        Unified architecture: Files stay in place, status tracked in JSON.
        """
        from mousereach.segmentation.core.batch import process_single, add_validation_status

        total = len(dlc_files)
        self.progress_callback('segmentation', 0, total, f"Segmenting {total} files...")

        for i, dlc_path in enumerate(dlc_files):
            video_id = extract_video_id(dlc_path)
            self.progress_callback('segmentation', i + 1, total, f"Segmenting {video_id}...")

            try:
                result = process_single(dlc_path)
                results.seg_processed += 1

                if result.get('success', False):
                    status = result['status']
                    json_path = Path(result['output_file'])

                    # Determine segmentation status
                    if status == 'good':
                        add_validation_status(json_path, 'auto_approved')
                        results.seg_auto_approved += 1
                    elif has_complete_gt(dlc_path.parent, video_id, 'seg'):
                        add_validation_status(json_path, 'validated')
                        results.seg_auto_approved += 1
                    else:
                        add_validation_status(json_path, 'needs_review')
                        results.seg_needs_review += 1

                    # ALWAYS continue to reach/outcome detection
                    # All three stages are independent - don't gate on seg status
                    self._process_single_through_outcomes_reaches(
                        video_id, results, source_dir=dlc_path.parent
                    )
                else:
                    results.seg_failed += 1

            except Exception as e:
                results.seg_failed += 1

    def _process_seg_validated(self, dlc_files: List[Path], results: UnifiedResults) -> None:
        """Process segmented files through outcomes and reaches."""
        total = len(dlc_files)
        self.progress_callback('outcomes', 0, total, f"Processing {total} validated files...")

        for i, dlc_path in enumerate(dlc_files):
            video_id = extract_video_id(dlc_path)
            self.progress_callback('outcomes', i + 1, total, f"Processing {video_id}...")

            self._process_single_through_outcomes_reaches(video_id, results, source_dir=dlc_path.parent)

    def _process_single_through_outcomes_reaches(
        self,
        video_id: str,
        results: UnifiedResults,
        source_dir: Optional[Path] = None
    ) -> None:
        """Process a single video through outcomes and reaches.

        Unified architecture: Files stay in place, status tracked in JSON.
        """
        from mousereach.outcomes.core.pellet_outcome import PelletOutcomeDetector
        from mousereach.reach.core.reach_detector import ReachDetector
        from mousereach.reach.core.triage import check_anomalies

        if source_dir is None:
            source_dir = self.processing

        try:
            # Find files
            dlc_path = None
            for f in source_dir.glob(f"{video_id}DLC*.h5"):
                dlc_path = f
                break

            seg_path = source_dir / f"{video_id}_segments.json"

            if not dlc_path or not seg_path.exists():
                results.outcome_failed += 1
                return

            # Run outcome detection
            self.progress_callback('outcomes', 1, 2, f"Detecting outcomes for {video_id}...")
            outcome_detector = PelletOutcomeDetector()
            outcomes = outcome_detector.detect(dlc_path, seg_path)
            outcome_path = source_dir / f"{video_id}_pellet_outcomes.json"
            PelletOutcomeDetector.save_results(outcomes, outcome_path)
            results.outcome_processed += 1

            # Count segments needing review (considering GT coverage)
            # Raw algo output is preserved - GT checked separately
            gt_outcome_path = source_dir / f"{video_id}_outcome_ground_truth.json"
            gt_covered = count_gt_covered_outcome_segments(outcome_path, gt_outcome_path)
            outcome_needs_review = self._count_unresolved_outcome_segments(
                outcome_path, gt_outcome_path
            )
            if outcome_needs_review > 0:
                results.outcome_needs_review += 1
            else:
                results.outcome_completed += 1

            # Run reach detection in same folder
            self.progress_callback('reaches', 2, 2, f"Detecting reaches for {video_id}...")
            reach_detector = ReachDetector()
            reaches = reach_detector.detect(dlc_path, seg_path)
            reach_path = source_dir / f"{video_id}_reaches.json"
            ReachDetector.save_results(reaches, reach_path)
            results.reach_processed += 1

            # Triage reaches - check anomalies but exclude GT-covered segments
            gt_reach_path = source_dir / f"{video_id}_reach_ground_truth.json"
            gt_reach_covered = count_gt_covered_reach_segments(reach_path, gt_reach_path)

            reach_data = {
                'n_segments': reaches.n_segments,
                'segments': [{'n_reaches': s.n_reaches} for s in reaches.segments],
                'summary': reaches.summary
            }
            anomalies = check_anomalies(reach_data)

            # If all segments have GT or no anomalies, validated
            if gt_reach_covered == reaches.n_segments or not anomalies:
                results.reach_validated += 1
            else:
                results.reach_needs_review += 1

        except Exception as e:
            results.outcome_failed += 1

    def _count_unresolved_outcome_segments(
        self, outcome_path: Path, gt_path: Optional[Path] = None
    ) -> int:
        """
        Count segments that still need human review.

        A segment needs review if:
        - confidence < 0.85 AND
        - not human_verified in algo output AND
        - not covered by GT file

        Args:
            outcome_path: Path to algorithm outcome results
            gt_path: Path to GT file (if exists)

        Returns:
            Number of segments needing review
        """
        if not outcome_path.exists():
            return 0

        try:
            with open(outcome_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 0

        # Load GT segments lookup if GT exists
        gt_segments = {}
        if gt_path and gt_path.exists():
            try:
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                gt_segments = {
                    seg['segment_num']: seg
                    for seg in gt_data.get('segments', [])
                }
            except (json.JSONDecodeError, OSError):
                pass

        count = 0
        for seg in data.get('segments', []):
            seg_num = seg.get('segment_num')

            # Skip if already verified by human in algo output
            if seg.get('human_verified', False):
                continue

            # Skip if GT has human-verified answer for this segment
            gt_seg = gt_segments.get(seg_num)
            if gt_seg and gt_seg.get('human_verified', False):
                continue

            # Check confidence threshold
            if seg.get('confidence', 0) < 0.85:
                count += 1

        return count
