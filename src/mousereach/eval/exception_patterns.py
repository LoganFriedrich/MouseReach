"""
Exception Pattern Detection - Identifies and categorizes algorithm error patterns.

Provides:
- Automatic error pattern grouping
- Human-readable descriptions for each pattern
- Clickable examples for each pattern
- Fix suggestions based on error analysis
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


@dataclass
class ErrorExample:
    """A specific example of an error pattern."""
    video_id: str
    segment_num: Optional[int] = None
    reach_id: Optional[int] = None
    frame: Optional[int] = None

    # Details
    gt_value: Any = None
    algo_value: Any = None
    error_magnitude: float = 0.0

    # Navigation info for widget
    processing_dir: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            'video_id': self.video_id,
            'segment_num': self.segment_num,
            'reach_id': self.reach_id,
            'frame': self.frame,
            'gt_value': self.gt_value,
            'algo_value': self.algo_value,
            'error_magnitude': self.error_magnitude,
            'processing_dir': self.processing_dir,
        }


@dataclass
class ExceptionPattern:
    """A detected error pattern with description and examples."""
    pattern_id: str
    name: str
    description: str
    count: int = 0
    severity: str = "medium"  # low, medium, high, critical

    # Human-readable explanation
    explanation: str = ""
    potential_fix: str = ""

    # Examples (max 10 for UI)
    examples: List[ErrorExample] = field(default_factory=list)

    # Statistics
    affected_videos: List[str] = field(default_factory=list)
    mean_error: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)

    def add_example(self, example: ErrorExample):
        """Add an example, keeping max 10."""
        self.count += 1
        if example.video_id not in self.affected_videos:
            self.affected_videos.append(example.video_id)

        if len(self.examples) < 10:
            self.examples.append(example)


# ============ Reach Error Patterns ============

REACH_PATTERNS = {
    "reach_starts_early": ExceptionPattern(
        pattern_id="reach_starts_early",
        name="Reach starts early",
        description="Algorithm detects reach start before the hand actually crosses the slit",
        explanation=(
            "The algorithm detects hand visibility before the hand actually "
            "crosses the slit boundary. This happens when DLC tracks the hand "
            "on the cage side, causing premature start detection."
        ),
        potential_fix=(
            "Require hand X > BOXR_x (or a threshold) before triggering start. "
            "Add hysteresis to prevent flickering near boundary."
        ),
        severity="medium"
    ),

    "reach_starts_late": ExceptionPattern(
        pattern_id="reach_starts_late",
        name="Reach starts late",
        description="Algorithm misses the initial reach motion",
        explanation=(
            "The algorithm detects reach start after the hand has already "
            "begun reaching. This may be due to the hand being occluded at "
            "reach initiation or DLC confidence dropping briefly."
        ),
        potential_fix=(
            "Consider lookback when reach is confirmed to find actual start. "
            "Lower confidence threshold for start detection."
        ),
        severity="medium"
    ),

    "reach_ends_early": ExceptionPattern(
        pattern_id="reach_ends_early",
        name="Reach ends early",
        description="Algorithm terminates reach before full retraction",
        explanation=(
            "The algorithm ends the reach while the hand is still extended "
            "or retracting. Often happens when the hand pauses mid-retraction "
            "or DLC tracking momentarily fails."
        ),
        potential_fix=(
            "Require sustained retraction before ending reach. "
            "Add minimum reach duration constraint."
        ),
        severity="medium"
    ),

    "reach_ends_late": ExceptionPattern(
        pattern_id="reach_ends_late",
        name="Reach ends late",
        description="Algorithm extends reach beyond actual retraction",
        explanation=(
            "The algorithm includes frames after the hand has fully retracted. "
            "May include rest period or start of next reach."
        ),
        potential_fix=(
            "Tighten velocity threshold for end detection. "
            "Check for hand returning to rest position."
        ),
        severity="low"
    ),

    "reach_missed": ExceptionPattern(
        pattern_id="reach_missed",
        name="Reach completely missed",
        description="Algorithm failed to detect a reach that humans marked",
        explanation=(
            "A reach that humans identified was not detected at all by the "
            "algorithm. This is a false negative - the most critical error type "
            "as it affects recall."
        ),
        potential_fix=(
            "Check if extent filter is too strict. "
            "Verify detection thresholds match hand positions in these cases. "
            "May need lower minimum extent requirement."
        ),
        severity="high"
    ),

    "phantom_reach": ExceptionPattern(
        pattern_id="phantom_reach",
        name="Phantom reach detected",
        description="Algorithm detected reach where none exists",
        explanation=(
            "The algorithm detected a reach that humans did not mark. "
            "This could be hand grooming, repositioning, or noise in tracking."
        ),
        potential_fix=(
            "Add filtering for non-reach hand movements. "
            "Require approach toward slit before triggering. "
            "Check pellet presence timing."
        ),
        severity="medium"
    ),

    "reach_merged": ExceptionPattern(
        pattern_id="reach_merged",
        name="Reaches merged together",
        description="Multiple GT reaches detected as single reach",
        explanation=(
            "The algorithm merged two or more separate reaches into one. "
            "Usually happens when there's minimal retraction between reaches "
            "(quick successive attempts)."
        ),
        potential_fix=(
            "Lower the inter-reach gap threshold. "
            "Detect brief pauses or direction reversals."
        ),
        severity="medium"
    ),

    "reach_split": ExceptionPattern(
        pattern_id="reach_split",
        name="Reach split into multiple",
        description="Single GT reach detected as multiple reaches",
        explanation=(
            "The algorithm split one reach into multiple detections. "
            "Often happens when DLC tracking briefly fails mid-reach or "
            "the hand pauses during extension."
        ),
        potential_fix=(
            "Add reach merging for gaps < N frames. "
            "Increase minimum reach duration."
        ),
        severity="low"
    ),
}


# ============ Outcome Error Patterns ============

OUTCOME_PATTERNS = {
    "displaced_as_retrieved": ExceptionPattern(
        pattern_id="displaced_as_retrieved",
        name="Displaced classified as retrieved",
        description="Pellet was displaced but algorithm said retrieved",
        explanation=(
            "The algorithm thinks the pellet was retrieved (eaten) but the "
            "human marked it as displaced. The pellet may have moved but "
            "remained visible, or the algorithm's eating signature detection "
            "triggered incorrectly."
        ),
        potential_fix=(
            "Check pellet visibility after interaction. "
            "Verify eating signature requires sustained occlusion. "
            "May need stricter nose proximity requirement."
        ),
        severity="high"
    ),

    "retrieved_as_displaced": ExceptionPattern(
        pattern_id="retrieved_as_displaced",
        name="Retrieved classified as displaced",
        description="Pellet was retrieved but algorithm said displaced",
        explanation=(
            "The algorithm classified as displaced when the pellet was "
            "actually retrieved. The eating event may have been missed or "
            "the pellet tracking lost the pellet before eating was confirmed."
        ),
        potential_fix=(
            "Check for pellet disappearance + eating posture combo. "
            "May need longer lookback for eating confirmation."
        ),
        severity="high"
    ),

    "untouched_as_displaced": ExceptionPattern(
        pattern_id="untouched_as_displaced",
        name="Untouched classified as displaced",
        description="No interaction but algorithm detected displacement",
        explanation=(
            "The algorithm detected pellet displacement when the pellet "
            "wasn't actually touched. Common causes: camera shake, tray wobble, "
            "or tracking noise creating phantom movement."
        ),
        potential_fix=(
            "Increase displacement threshold. "
            "Require paw proximity during displacement. "
            "Filter out small movements from vibration."
        ),
        severity="medium"
    ),

    "displaced_as_untouched": ExceptionPattern(
        pattern_id="displaced_as_untouched",
        name="Displaced classified as untouched",
        description="Pellet was displaced but algorithm said no interaction",
        explanation=(
            "The algorithm missed a displacement. The pellet moved but "
            "wasn't detected - possibly small displacement below threshold "
            "or pellet tracking lost the pellet."
        ),
        potential_fix=(
            "Lower displacement threshold. "
            "Check pellet tracking stability in these cases. "
            "May need better handling of partial occlusion."
        ),
        severity="medium"
    ),

    "interaction_frame_error": ExceptionPattern(
        pattern_id="interaction_frame_error",
        name="Interaction frame timing error",
        description="Outcome correct but interaction frame is off",
        explanation=(
            "The outcome classification is correct but the interaction "
            "frame (when the pellet was touched/eaten) is significantly "
            "different from ground truth."
        ),
        potential_fix=(
            "Review interaction detection logic. "
            "May need better pellet visibility tracking. "
            "Check for occlusion during interaction."
        ),
        severity="low"
    ),

    "causal_reach_wrong": ExceptionPattern(
        pattern_id="causal_reach_wrong",
        name="Wrong causal reach identified",
        description="Outcome correct but attributed to wrong reach",
        explanation=(
            "The outcome is correct but the algorithm attributed it to a "
            "different reach than the human did. This affects reach-outcome "
            "linkage for downstream analysis."
        ),
        potential_fix=(
            "Review reach-outcome linkage logic. "
            "May need better temporal overlap detection. "
            "Check for multi-reach segments."
        ),
        severity="low"
    ),
}


class ExceptionPatternDetector:
    """
    Detects and categorizes error patterns from evaluation results.

    Analyzes evaluation data to find systematic errors and group them
    into actionable patterns with examples.
    """

    def __init__(self, processing_root: Path = None):
        """
        Initialize detector.

        Args:
            processing_root: Root for finding video files
        """
        if processing_root is None:
            from mousereach.config import PROCESSING_ROOT
            processing_root = PROCESSING_ROOT

        self.processing_root = Path(processing_root)

        # Pattern storage
        self.reach_patterns: Dict[str, ExceptionPattern] = {}
        self.outcome_patterns: Dict[str, ExceptionPattern] = {}

        self._init_patterns()

    def _init_patterns(self):
        """Initialize pattern templates."""
        # Deep copy templates
        import copy
        self.reach_patterns = {
            k: copy.deepcopy(v) for k, v in REACH_PATTERNS.items()
        }
        self.outcome_patterns = {
            k: copy.deepcopy(v) for k, v in OUTCOME_PATTERNS.items()
        }

    def analyze_reach_errors(self, eval_results: List[Dict]) -> Dict[str, ExceptionPattern]:
        """
        Analyze reach evaluation results for error patterns.

        Args:
            eval_results: List of ReachEvalResult dicts

        Returns:
            Dict of pattern_id -> ExceptionPattern
        """
        self._init_patterns()  # Reset counts

        for result in eval_results:
            if not result.get('success', False):
                continue

            video_id = result.get('video_id', 'unknown')
            processing_dir = self._find_processing_dir(video_id)

            # Analyze timing errors in matched reaches
            for match in result.get('matches', []):
                if not match.get('matched', False):
                    continue

                start_error = match.get('start_error', 0)
                end_error = match.get('end_error', 0)
                segment_num = match.get('segment_num')
                reach_id = match.get('gt_reach_id')

                # Start timing patterns
                if start_error < -2:  # Starts too early
                    self.reach_patterns['reach_starts_early'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=segment_num,
                            reach_id=reach_id,
                            gt_value=f"GT start",
                            algo_value=f"Algo start ({start_error} frames early)",
                            error_magnitude=abs(start_error),
                            processing_dir=processing_dir
                        )
                    )
                elif start_error > 2:  # Starts too late
                    self.reach_patterns['reach_starts_late'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=segment_num,
                            reach_id=reach_id,
                            gt_value=f"GT start",
                            algo_value=f"Algo start ({start_error} frames late)",
                            error_magnitude=abs(start_error),
                            processing_dir=processing_dir
                        )
                    )

                # End timing patterns
                if end_error < -2:  # Ends too early
                    self.reach_patterns['reach_ends_early'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=segment_num,
                            reach_id=reach_id,
                            gt_value=f"GT end",
                            algo_value=f"Algo end ({end_error} frames early)",
                            error_magnitude=abs(end_error),
                            processing_dir=processing_dir
                        )
                    )
                elif end_error > 2:  # Ends too late
                    self.reach_patterns['reach_ends_late'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=segment_num,
                            reach_id=reach_id,
                            gt_value=f"GT end",
                            algo_value=f"Algo end ({end_error} frames late)",
                            error_magnitude=abs(end_error),
                            processing_dir=processing_dir
                        )
                    )

            # False negatives (missed reaches)
            for fn in result.get('false_negatives', []):
                self.reach_patterns['reach_missed'].add_example(
                    ErrorExample(
                        video_id=video_id,
                        segment_num=fn.get('segment'),
                        reach_id=fn.get('reach_id'),
                        gt_value="Reach present",
                        algo_value="Not detected",
                        processing_dir=processing_dir
                    )
                )

            # False positives (phantom reaches)
            for fp in result.get('false_positives', []):
                self.reach_patterns['phantom_reach'].add_example(
                    ErrorExample(
                        video_id=video_id,
                        segment_num=fp.get('segment'),
                        reach_id=fp.get('reach_id'),
                        frame=fp.get('start'),
                        gt_value="No reach",
                        algo_value="Reach detected",
                        processing_dir=processing_dir
                    )
                )

            # Merged reaches (check segment count differences)
            for seg_error in result.get('segment_count_errors', []):
                if seg_error.get('diff', 0) < -1:  # Algo found fewer
                    self.reach_patterns['reach_merged'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=seg_error.get('segment'),
                            gt_value=f"{seg_error.get('gt_count')} reaches",
                            algo_value=f"{seg_error.get('algo_count')} reaches",
                            processing_dir=processing_dir
                        )
                    )
                elif seg_error.get('diff', 0) > 1:  # Algo found more
                    self.reach_patterns['reach_split'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=seg_error.get('segment'),
                            gt_value=f"{seg_error.get('gt_count')} reaches",
                            algo_value=f"{seg_error.get('algo_count')} reaches",
                            processing_dir=processing_dir
                        )
                    )

        # Calculate statistics
        for pattern in self.reach_patterns.values():
            if pattern.examples:
                pattern.mean_error = np.mean([
                    e.error_magnitude for e in pattern.examples
                    if e.error_magnitude > 0
                ] or [0])

        return self.reach_patterns

    def analyze_outcome_errors(self, eval_results: List[Dict]) -> Dict[str, ExceptionPattern]:
        """
        Analyze outcome evaluation results for error patterns.

        Args:
            eval_results: List of OutcomeEvalResult dicts

        Returns:
            Dict of pattern_id -> ExceptionPattern
        """
        self._init_patterns()

        for result in eval_results:
            if not result.get('success', False):
                continue

            video_id = result.get('video_id', 'unknown')
            processing_dir = self._find_processing_dir(video_id)

            # Analyze misclassifications
            for misclass in result.get('misclassifications', []):
                gt = misclass.get('gt', '')
                algo = misclass.get('algo', '')
                segment = misclass.get('segment')
                confidence = misclass.get('confidence', 0)

                example = ErrorExample(
                    video_id=video_id,
                    segment_num=segment,
                    gt_value=gt,
                    algo_value=algo,
                    processing_dir=processing_dir
                )

                # Categorize by error type
                if gt == 'retrieved' and algo in ['displaced_sa', 'displaced_outside']:
                    self.outcome_patterns['retrieved_as_displaced'].add_example(example)
                elif gt in ['displaced_sa', 'displaced_outside'] and algo == 'retrieved':
                    self.outcome_patterns['displaced_as_retrieved'].add_example(example)
                elif gt == 'untouched' and algo in ['displaced_sa', 'displaced_outside']:
                    self.outcome_patterns['untouched_as_displaced'].add_example(example)
                elif gt in ['displaced_sa', 'displaced_outside'] and algo == 'untouched':
                    self.outcome_patterns['displaced_as_untouched'].add_example(example)

            # Analyze interaction frame errors
            for match in result.get('matches', []):
                if not match.get('matched', False):
                    continue

                frame_error = match.get('interaction_frame_error')
                if frame_error is not None and abs(frame_error) > 10:
                    self.outcome_patterns['interaction_frame_error'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=match.get('segment_num'),
                            frame=match.get('interaction_frame_gt'),
                            gt_value=f"Frame {match.get('interaction_frame_gt')}",
                            algo_value=f"Frame {match.get('interaction_frame_algo')}",
                            error_magnitude=abs(frame_error),
                            processing_dir=processing_dir
                        )
                    )

                # Causal reach mismatch
                if match.get('matched') and not match.get('causal_reach_matched', True):
                    self.outcome_patterns['causal_reach_wrong'].add_example(
                        ErrorExample(
                            video_id=video_id,
                            segment_num=match.get('segment_num'),
                            gt_value=f"Reach {match.get('causal_reach_gt')}",
                            algo_value=f"Reach {match.get('causal_reach_algo')}",
                            processing_dir=processing_dir
                        )
                    )

        return self.outcome_patterns

    def _find_processing_dir(self, video_id: str) -> Optional[str]:
        """Find the Processing directory for a video."""
        for processing_dir in self.processing_root.glob("*/Processing"):
            if any(processing_dir.glob(f"{video_id}*")):
                return str(processing_dir)
        return None

    def get_all_patterns(self) -> Dict[str, List[ExceptionPattern]]:
        """Get all detected patterns organized by category."""
        return {
            'reach': [p for p in self.reach_patterns.values() if p.count > 0],
            'outcome': [p for p in self.outcome_patterns.values() if p.count > 0],
        }

    def get_priority_patterns(self, min_count: int = 3) -> List[ExceptionPattern]:
        """
        Get patterns that should be prioritized for fixing.

        Args:
            min_count: Minimum occurrences to be considered

        Returns:
            List of high-priority patterns sorted by severity and count
        """
        all_patterns = []
        for patterns in [self.reach_patterns.values(), self.outcome_patterns.values()]:
            for p in patterns:
                if p.count >= min_count:
                    all_patterns.append(p)

        # Sort by severity (critical > high > medium > low) then by count
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_patterns.sort(key=lambda p: (severity_order.get(p.severity, 9), -p.count))

        return all_patterns

    def generate_report(self) -> str:
        """Generate a human-readable report of all patterns."""
        lines = []
        lines.append("=" * 60)
        lines.append("EXCEPTION PATTERN REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Reach patterns
        reach_with_errors = [p for p in self.reach_patterns.values() if p.count > 0]
        if reach_with_errors:
            lines.append("=== REACH DETECTION PATTERNS ===")
            lines.append("")

            for pattern in sorted(reach_with_errors, key=lambda p: -p.count):
                lines.append(f"[{pattern.severity.upper()}] {pattern.name} ({pattern.count} cases)")
                lines.append(f"  {pattern.description}")
                lines.append(f"  Affected videos: {len(pattern.affected_videos)}")

                if pattern.explanation:
                    lines.append(f"")
                    lines.append(f"  WHY: {pattern.explanation}")

                if pattern.potential_fix:
                    lines.append(f"")
                    lines.append(f"  FIX: {pattern.potential_fix}")

                if pattern.examples:
                    lines.append(f"")
                    lines.append(f"  Examples:")
                    for ex in pattern.examples[:3]:
                        lines.append(f"    - {ex.video_id}, seg {ex.segment_num}, reach {ex.reach_id}")

                lines.append("")

        # Outcome patterns
        outcome_with_errors = [p for p in self.outcome_patterns.values() if p.count > 0]
        if outcome_with_errors:
            lines.append("=== OUTCOME DETECTION PATTERNS ===")
            lines.append("")

            for pattern in sorted(outcome_with_errors, key=lambda p: -p.count):
                lines.append(f"[{pattern.severity.upper()}] {pattern.name} ({pattern.count} cases)")
                lines.append(f"  {pattern.description}")
                lines.append(f"  Affected videos: {len(pattern.affected_videos)}")

                if pattern.explanation:
                    lines.append(f"")
                    lines.append(f"  WHY: {pattern.explanation}")

                if pattern.potential_fix:
                    lines.append(f"")
                    lines.append(f"  FIX: {pattern.potential_fix}")

                if pattern.examples:
                    lines.append(f"")
                    lines.append(f"  Examples:")
                    for ex in pattern.examples[:3]:
                        lines.append(f"    - {ex.video_id}, seg {ex.segment_num}: GT={ex.gt_value}, Algo={ex.algo_value}")

                lines.append("")

        if not reach_with_errors and not outcome_with_errors:
            lines.append("No error patterns detected. Algorithm is performing well!")

        return "\n".join(lines)


# Convenience function
def detect_all_patterns(processing_root: Path = None) -> ExceptionPatternDetector:
    """
    Run pattern detection on all GT data.

    Returns:
        Configured ExceptionPatternDetector with analyzed patterns
    """
    from .reach_evaluator import ReachEvaluator
    from .outcome_evaluator import OutcomeEvaluator

    if processing_root is None:
        from mousereach.config import PROCESSING_ROOT
        processing_root = PROCESSING_ROOT

    detector = ExceptionPatternDetector(processing_root)

    # Find and analyze all GT files
    reach_results = []
    outcome_results = []

    for proc_dir in processing_root.glob("*/Processing"):
        # Reach GT
        for gt_file in proc_dir.glob("*_reach_ground_truth.json"):
            video_id = gt_file.stem.replace("_reach_ground_truth", "")
            evaluator = ReachEvaluator(gt_dir=proc_dir, algo_dir=proc_dir)
            result = evaluator.compare(video_id)
            if result.success:
                reach_results.append(result.__dict__)

        # Outcome GT
        for gt_file in proc_dir.glob("*_outcome*_ground_truth.json"):
            video_id = gt_file.stem.replace("_outcome_ground_truth", "").replace("_outcomes_ground_truth", "")
            evaluator = OutcomeEvaluator(gt_dir=proc_dir, algo_dir=proc_dir)
            result = evaluator.compare(video_id)
            if result.success:
                outcome_results.append(result.__dict__)

    # Analyze patterns
    detector.analyze_reach_errors(reach_results)
    detector.analyze_outcome_errors(outcome_results)

    return detector
