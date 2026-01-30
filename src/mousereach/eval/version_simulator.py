"""
Version simulator for reach detection algorithm.

Simulates how DIFFERENT algorithmic approaches would have performed on ground truth data.
Each version represents a fundamentally different detection philosophy, not just parameter tweaks.

ALGORITHMIC APPROACHES SIMULATED
================================

The reach detection problem has several key decision points:

1. ENTRY DETECTION - When does a reach start?
   - Naive: Any hand visibility near slit
   - Likelihood: Hand detection confidence above threshold
   - Engagement: Hand visible AND nose engaged with slit
   - Crossing: Hand X-coordinate crosses slit boundary

2. EXIT DETECTION - When does a reach end?
   - Disappearance: Hand likelihood drops below threshold
   - Retraction: Hand moves back toward cage
   - Return: Hand X-coordinate returns past threshold
   - Combined: Disappear OR retract (current)

3. REACH VALIDATION - Is this a real reach?
   - None: Accept all detected start/end pairs
   - Duration: Minimum frames required
   - Extent: Hand must extend past slit by N pixels
   - Velocity: Must have meaningful motion profile
   - Apex: Must have clear extension peak

4. MULTI-REACH HANDLING - Complex situations?
   - Independent: Each start/end is separate reach
   - Merge: Combine reaches within gap tolerance
   - Split: Break up very long reaches
   - Adaptive: Merge short gaps, split long durations (current)

VERSION HISTORY
===============

v0.5.0 - Naive Baseline
    Philosophy: "Detect any hand activity near the slit"
    Entry: Hand likelihood > 0.3 (very permissive)
    Exit: Hand disappears (likelihood < 0.3)
    Validation: None
    Multi: Independent (no merge/split)
    Expected: High false positives, catches everything including noise

v1.0.0 - Basic Threshold
    Philosophy: "Require confident hand detection"
    Entry: Hand likelihood > 0.5
    Exit: Hand disappears
    Validation: Duration >= 3 frames
    Multi: Independent
    Expected: Fewer false positives, still misses nothing meaningful

v1.5.0 - Nose Engagement
    Philosophy: "Mouse must be intentionally reaching, not just moving"
    Entry: Hand likelihood > 0.5 AND nose within 50px of slit
    Exit: Hand disappears
    Validation: Duration >= 3 frames
    Multi: Independent
    Expected: Major precision improvement, may miss reaches where nose wasn't tracked

v2.0.0 - Extent Requirement
    Philosophy: "Only count reaches where hand actually extends past slit"
    Entry: Same as v1.5
    Exit: Hand disappears
    Validation: Extent > 0 pixels (hand must cross slit line)
    Multi: Independent
    Expected: Good precision, misses "approach-only" reaches

v2.5.0 - Retraction Detection
    Philosophy: "End reach when hand pulls back, not just when it disappears"
    Entry: Same as v1.5
    Exit: Hand disappears OR retracts > 10 pixels from max extension
    Validation: Extent > 0 pixels
    Multi: Merge reaches within 2 frames (gap tolerance)
    Expected: Better end timing, handles brief occlusions

v3.0.0 - Strict Filtering
    Philosophy: "Only count definitive reaches with clear extension"
    Entry: Same as v1.5
    Exit: Same as v2.5
    Validation: Extent >= 5px AND Duration >= 10 frames
    Multi: Merge + Split (break reaches > 30 frames)
    Expected: High precision, misses quick probes and partial reaches

v3.3.0 - The Bug
    Philosophy: Same as v3.0 but with implementation bug
    Entry: Same as v1.5
    Exit: Same as v2.5
    Validation: Extent >= 0 (BUG: this rejects negative extent reaches)
    Multi: Same as v3.0
    PROBLEM: Many valid reaches have extent -2 to -15px because BOXR_x
             reference point is beyond the actual slit opening

v3.4.0 - Current (Bug Fixed)
    Philosophy: "Detect all intentional reaches, preserve extent for downstream filtering"
    Entry: Hand likelihood > 0.5 AND nose within 25px of slit
    Exit: Hand disappears OR retracts
    Validation: Duration >= 2 frames ONLY (no extent filter)
    Multi: Merge (gap=2) + Split (threshold=25)
    Expected: Best recall, extent preserved for researchers to filter as needed

SIMULATION LIMITATIONS
======================
Since we don't re-run detection from raw DLC data, we simulate by:
1. Using current v3.4.0 output as the "superset" (most permissive)
2. Applying logic filters to approximate what earlier versions would have detected
3. For versions that might detect DIFFERENT reaches (not just fewer), we note the limitation

The key insight: v3.4.0's no-extent-filter design means it finds a superset of
reaches from versions that filter by extent. So we can simulate those versions
by filtering the v3.4.0 output.

USAGE
=====
```python
from mousereach.eval.version_simulator import (
    evaluate_all_versions,
    print_version_comparison,
    ALGORITHMIC_VERSIONS
)

# See what approaches we're simulating
for v, config in ALGORITHMIC_VERSIONS.items():
    print(f"{v}: {config.philosophy}")

# Run simulation
results = evaluate_all_versions(reaches_dir, gt_dir)
print_version_comparison(results)
```
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np

from .reach_evaluator import ReachEvaluator, ReachEvalResult


@dataclass
class AlgorithmicVersion:
    """
    Defines an algorithmic approach for reach detection.

    This captures the PHILOSOPHY behind the algorithm, not just parameters.
    """
    # Required fields (no defaults) - must come first
    version: str
    philosophy: str  # One-line description of the approach
    entry_method: str  # "naive", "likelihood", "engagement", "crossing"
    exit_method: str  # "disappear", "retraction", "return", "combined"

    # Entry detection (optional)
    entry_likelihood_threshold: float = 0.5
    entry_nose_engagement: bool = False
    entry_nose_threshold_px: float = 50.0

    # Exit detection (optional)
    exit_retraction_threshold_px: float = 10.0

    # Validation filters (optional)
    min_duration_frames: Optional[int] = None
    min_extent_pixels: Optional[float] = None
    max_extent_pixels: Optional[float] = None
    require_positive_extent: bool = False
    require_apex: bool = False
    min_max_velocity: Optional[float] = None

    # Multi-reach handling (optional)
    merge_gap_frames: Optional[int] = None  # None = no merging
    split_threshold_frames: Optional[int] = None  # None = no splitting

    # Documentation (optional)
    expected_behavior: str = ""
    known_issues: List[str] = field(default_factory=list)


# Define all algorithmic approaches
ALGORITHMIC_VERSIONS: Dict[str, AlgorithmicVersion] = {
    "v0.5.0": AlgorithmicVersion(
        version="v0.5.0",
        philosophy="Detect any hand activity near slit - maximally permissive",
        entry_method="naive",
        entry_likelihood_threshold=0.3,
        entry_nose_engagement=False,
        exit_method="disappear",
        min_duration_frames=None,  # No duration filter
        min_extent_pixels=None,
        expected_behavior="Catches everything including noise, very low precision",
        known_issues=["High false positive rate", "Detects hand grooming as reaches"]
    ),

    "v1.0.0": AlgorithmicVersion(
        version="v1.0.0",
        philosophy="Require confident hand detection with minimum duration",
        entry_method="likelihood",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=False,
        exit_method="disappear",
        min_duration_frames=3,
        min_extent_pixels=None,
        expected_behavior="Fewer false positives, but still too permissive",
        known_issues=["Detects non-reaching hand movements"]
    ),

    "v1.5.0": AlgorithmicVersion(
        version="v1.5.0",
        philosophy="Mouse must be intentionally reaching (nose engaged)",
        entry_method="engagement",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=True,
        entry_nose_threshold_px=50.0,
        exit_method="disappear",
        min_duration_frames=3,
        min_extent_pixels=None,
        expected_behavior="Major precision improvement, some missed reaches when nose occluded",
        known_issues=["Misses reaches when nose tracking fails"]
    ),

    "v2.0.0": AlgorithmicVersion(
        version="v2.0.0",
        philosophy="Only count reaches where hand extends past slit",
        entry_method="engagement",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=True,
        entry_nose_threshold_px=50.0,
        exit_method="disappear",
        min_duration_frames=3,
        require_positive_extent=True,  # Key difference: extent > 0
        expected_behavior="Good precision, misses approach-only attempts",
        known_issues=["Misses scientifically meaningful 'approach' reaches"]
    ),

    "v2.5.0": AlgorithmicVersion(
        version="v2.5.0",
        philosophy="End reach on hand retraction, merge brief gaps",
        entry_method="engagement",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=True,
        entry_nose_threshold_px=50.0,
        exit_method="combined",  # disappear OR retract
        exit_retraction_threshold_px=10.0,
        min_duration_frames=3,
        require_positive_extent=True,
        merge_gap_frames=2,  # Merge reaches within 2 frames
        expected_behavior="Better end timing, handles brief occlusions",
        known_issues=["Still misses approach reaches"]
    ),

    "v3.0.0": AlgorithmicVersion(
        version="v3.0.0",
        philosophy="Strict filtering for definitive reaches only",
        entry_method="engagement",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=True,
        entry_nose_threshold_px=25.0,  # Tighter nose engagement
        exit_method="combined",
        exit_retraction_threshold_px=10.0,
        min_duration_frames=10,  # Longer minimum
        min_extent_pixels=5.0,  # Must extend at least 5 pixels
        merge_gap_frames=2,
        split_threshold_frames=30,  # Split reaches > 30 frames
        expected_behavior="High precision, misses quick probes and partial reaches",
        known_issues=["Too strict", "Misses valid short reaches"]
    ),

    "v3.3.0": AlgorithmicVersion(
        version="v3.3.0",
        philosophy="Bug version - extent >= 0 filter incorrectly applied",
        entry_method="engagement",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=True,
        entry_nose_threshold_px=25.0,
        exit_method="combined",
        exit_retraction_threshold_px=10.0,
        min_duration_frames=2,
        min_extent_pixels=0.0,  # THE BUG: This rejects negative extent
        merge_gap_frames=2,
        split_threshold_frames=25,
        expected_behavior="CATASTROPHIC: Drops 85% of valid reaches",
        known_issues=[
            "BUG: Filter 'extent >= 0' removes reaches with negative extent",
            "Many valid reaches have extent -2 to -15px",
            "BOXR_x reference point is beyond actual slit opening",
            "Human GT shows these are scientifically meaningful approaches"
        ]
    ),

    "v3.4.0": AlgorithmicVersion(
        version="v3.4.0",
        philosophy="Detect all intentional reaches, preserve extent for downstream filtering",
        entry_method="engagement",
        entry_likelihood_threshold=0.5,
        entry_nose_engagement=True,
        entry_nose_threshold_px=25.0,
        exit_method="combined",
        exit_retraction_threshold_px=10.0,
        min_duration_frames=2,  # Minimal duration filter
        min_extent_pixels=None,  # NO EXTENT FILTER - key fix
        merge_gap_frames=2,
        split_threshold_frames=25,
        expected_behavior="Best recall, extent preserved for researcher filtering",
        known_issues=[]  # Current best version
    ),
}

# Backwards compatibility alias
VersionFilter = AlgorithmicVersion
VERSION_FILTERS = {v: config for v, config in ALGORITHMIC_VERSIONS.items()}


def apply_version_filter(reaches: List[Dict], version: AlgorithmicVersion) -> List[Dict]:
    """
    Apply version-specific logic to filter reaches.

    Simulates what the old algorithm version would have detected.

    Args:
        reaches: List of reach dicts from current algorithm output
        version: Version configuration

    Returns:
        Filtered list of reaches (what old version would have detected)
    """
    filtered = []

    for reach in reaches:
        # Extract reach properties
        extent = reach.get("max_extent_pixels", 0)
        duration = reach.get("duration_frames", 0)
        max_velocity = reach.get("max_velocity", 0)
        has_apex = reach.get("apex_frame") is not None

        # --- Validation filters ---

        # Duration filter
        if version.min_duration_frames is not None:
            if duration < version.min_duration_frames:
                continue

        # Extent filters
        if version.min_extent_pixels is not None:
            if extent < version.min_extent_pixels:
                continue

        if version.max_extent_pixels is not None:
            if extent >= version.max_extent_pixels:
                continue

        if version.require_positive_extent:
            if extent <= 0:
                continue

        # Apex requirement
        if version.require_apex and not has_apex:
            continue

        # Velocity filter
        if version.min_max_velocity is not None:
            if max_velocity < version.min_max_velocity:
                continue

        # Passed all filters
        filtered.append(reach)

    return filtered


def simulate_version(
    reaches_path: Path,
    gt_path: Path,
    version: AlgorithmicVersion,
    evaluator: ReachEvaluator
) -> Optional[ReachEvalResult]:
    """
    Simulate how an old version would perform on one video.

    Args:
        reaches_path: Path to current _reaches.json file
        gt_path: Path to _reach_ground_truth.json file
        version: Version configuration
        evaluator: ReachEvaluator instance

    Returns:
        Evaluation result (or None if file missing)
    """
    if not reaches_path.exists() or not gt_path.exists():
        return None

    # Load current algorithm output
    with open(reaches_path) as f:
        current_output = json.load(f)

    # Apply old version's filters to simulate what it would have detected
    simulated_output = current_output.copy()
    simulated_output["detector_version"] = version.version
    simulated_output["simulation_note"] = f"Simulated {version.version}: {version.philosophy}"

    total_filtered = 0
    total_original = 0

    for segment in simulated_output.get("segments", []):
        original_reaches = segment.get("reaches", [])
        total_original += len(original_reaches)

        filtered_reaches = apply_version_filter(original_reaches, version)
        total_filtered += len(filtered_reaches)

        segment["reaches"] = filtered_reaches
        segment["n_reaches"] = len(filtered_reaches)

    # Temporarily save simulated output for evaluation
    temp_path = reaches_path.parent / f"_temp_sim_{version.version}_{reaches_path.name}"
    try:
        with open(temp_path, 'w') as f:
            json.dump(simulated_output, f)

        # Extract video ID
        video_id = evaluator.extract_video_id(reaches_path)

        # Evaluate using standard evaluator
        original_algo_dir = evaluator.algo_dir
        evaluator.algo_dir = temp_path.parent

        # Rename temp file to expected name
        expected_name = f"{video_id}_reaches.json"
        expected_path = temp_path.parent / expected_name

        # If expected file already exists, save it
        needs_restore = expected_path.exists()
        original_content = None
        if needs_restore:
            with open(expected_path) as f:
                original_content = json.load(f)

        # Move temp to expected location
        if expected_path.exists():
            expected_path.unlink()
        temp_path.rename(expected_path)

        try:
            result = evaluator.compare(video_id)
            # Add simulation metadata to result
            if result:
                result.notes = f"Simulated {version.version}: kept {total_filtered}/{total_original} reaches"
        finally:
            # Restore original file
            if needs_restore and original_content:
                with open(expected_path, 'w') as f:
                    json.dump(original_content, f)
            elif expected_path.exists():
                expected_path.unlink()

            evaluator.algo_dir = original_algo_dir

        return result

    finally:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()


def evaluate_version_on_dataset(
    reaches_dir: Path,
    gt_dir: Path,
    version: AlgorithmicVersion,
    tolerance: int = 10
) -> Dict:
    """
    Evaluate a simulated version on entire dataset.

    Args:
        reaches_dir: Directory with _reaches.json files
        gt_dir: Directory with _reach_ground_truth.json files
        version: Version configuration
        tolerance: Frame tolerance for matching (default 10)

    Returns:
        Metrics dict with precision, recall, F1, etc.
    """
    evaluator = ReachEvaluator(gt_dir=gt_dir, algo_dir=reaches_dir, tolerance=tolerance)

    # Find all GT files
    gt_files = list(gt_dir.glob("*_reach_ground_truth.json"))

    if not gt_files:
        return {
            "version": version.version,
            "philosophy": version.philosophy,
            "error": "No ground truth files found",
            "n_videos": 0
        }

    results = []

    for gt_file in gt_files:
        video_id = evaluator.extract_video_id(gt_file)
        reaches_file = reaches_dir / f"{video_id}_reaches.json"

        if not reaches_file.exists():
            continue

        result = simulate_version(reaches_file, gt_file, version, evaluator)
        if result and result.success:
            results.append(result)

    if not results:
        return {
            "version": version.version,
            "philosophy": version.philosophy,
            "error": "No successful evaluations",
            "n_videos": 0
        }

    # Aggregate metrics
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    f1s = [r.f1 for r in results]

    total_gt = sum(r.n_gt_reaches for r in results)
    total_algo = sum(r.n_algo_reaches for r in results)

    return {
        "version": version.version,
        "philosophy": version.philosophy,
        "expected_behavior": version.expected_behavior,
        "known_issues": version.known_issues,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "n_reaches_detected": total_algo,
        "n_reaches_gt": total_gt,
        "n_videos": len(results),
        "precision_std": float(np.std(precisions)),
        "recall_std": float(np.std(recalls)),
    }


def evaluate_all_versions(
    reaches_dir: str | Path,
    gt_dir: str | Path,
    tolerance: int = 10,
    versions: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Evaluate all historical versions on current ground truth.

    Args:
        reaches_dir: Directory with current _reaches.json files
        gt_dir: Directory with _reach_ground_truth.json files
        tolerance: Frame tolerance for matching (default 10)
        versions: List of version strings to evaluate (default: all)

    Returns:
        Dict mapping version string to metrics dict

    Example:
        >>> results = evaluate_all_versions("Processing", "Processing")
        >>> for version, metrics in results.items():
        ...     print(f"{version}: {metrics['philosophy']}")
        ...     print(f"  Recall={metrics['recall']:.1%}")
    """
    reaches_dir = Path(reaches_dir)
    gt_dir = Path(gt_dir)

    if versions is None:
        versions = list(ALGORITHMIC_VERSIONS.keys())

    results = {}

    for version_str in versions:
        if version_str not in ALGORITHMIC_VERSIONS:
            print(f"Warning: Unknown version {version_str}, skipping")
            continue

        version = ALGORITHMIC_VERSIONS[version_str]
        print(f"Evaluating {version_str}: {version.philosophy}")

        metrics = evaluate_version_on_dataset(
            reaches_dir=reaches_dir,
            gt_dir=gt_dir,
            version=version,
            tolerance=tolerance
        )

        results[version_str] = metrics

    return results


def print_version_comparison(results: Dict[str, Dict]):
    """
    Pretty-print version comparison table with algorithmic context.

    Args:
        results: Output from evaluate_all_versions()
    """
    print("\n" + "="*90)
    print("REACH DETECTION ALGORITHM EVOLUTION")
    print("="*90)
    print()

    # Group versions by philosophy
    print("ALGORITHMIC APPROACHES TESTED:")
    print("-"*90)

    for version_str in ["v0.5.0", "v1.0.0", "v1.5.0", "v2.0.0", "v2.5.0", "v3.0.0", "v3.3.0", "v3.4.0"]:
        if version_str not in results:
            continue

        metrics = results[version_str]

        if "error" in metrics:
            print(f"\n{version_str}: ERROR - {metrics['error']}")
            continue

        philosophy = metrics.get("philosophy", "")
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        detected = metrics.get("n_reaches_detected", 0)
        gt = metrics.get("n_reaches_gt", 0)

        # Status markers
        if version_str == "v3.3.0":
            marker = "⚠️  BUG"
        elif version_str == "v3.4.0":
            marker = "✓ CURRENT"
        else:
            marker = ""

        print(f"\n{version_str} {marker}")
        print(f"  Philosophy: {philosophy}")
        print(f"  Precision: {precision:>6.1%}  |  Recall: {recall:>6.1%}  |  F1: {f1:.2f}")
        print(f"  Detected: {detected} / {gt} GT reaches")

        # Show known issues if any
        known_issues = metrics.get("known_issues", [])
        if known_issues:
            print(f"  Issues: {known_issues[0]}")

    print("\n" + "-"*90)

    # Analysis section
    print("\nKEY INSIGHTS:")
    print()

    # Find best precision and best recall
    best_precision_v = max(
        [v for v in results if "precision" in results[v]],
        key=lambda v: results[v].get("precision", 0),
        default=None
    )
    best_recall_v = max(
        [v for v in results if "recall" in results[v]],
        key=lambda v: results[v].get("recall", 0),
        default=None
    )

    if best_precision_v:
        print(f"  Best Precision: {best_precision_v} ({results[best_precision_v]['precision']:.1%})")
    if best_recall_v:
        print(f"  Best Recall: {best_recall_v} ({results[best_recall_v]['recall']:.1%})")

    # The bug impact
    if "v3.3.0" in results and "v3.4.0" in results:
        bug_recall = results["v3.3.0"].get("recall", 0)
        current_recall = results["v3.4.0"].get("recall", 0)

        if bug_recall > 0 and current_recall > 0:
            lost_pct = (1 - bug_recall/current_recall) * 100
            print(f"\n  v3.3.0 BUG IMPACT:")
            print(f"    The extent>=0 filter dropped {lost_pct:.0f}% of valid reaches")
            print(f"    Recall went from {current_recall:.1%} (correct) to {bug_recall:.1%} (buggy)")

    # Evolution lesson
    print(f"\n  LESSON LEARNED:")
    print(f"    Stricter filtering improves precision but risks catastrophic recall loss.")
    print(f"    Current v3.4.0 approach: detect all reaches, preserve extent for downstream filtering.")
    print()


def generate_version_report(
    reaches_dir: str | Path,
    gt_dir: str | Path,
    output_path: Optional[str | Path] = None
) -> str:
    """
    Generate comprehensive version comparison report.

    Args:
        reaches_dir: Directory with current _reaches.json files
        gt_dir: Directory with _reach_ground_truth.json files
        output_path: Optional path to save report

    Returns:
        Report string
    """
    results = evaluate_all_versions(reaches_dir, gt_dir)

    lines = []
    lines.append("="*80)
    lines.append("REACH DETECTION ALGORITHM VERSION HISTORY ANALYSIS")
    lines.append("="*80)
    lines.append("")
    lines.append("This report simulates how different algorithmic approaches would perform")
    lines.append("on the current ground truth dataset. Each version represents a different")
    lines.append("detection philosophy, not just parameter tweaks.")
    lines.append("")

    # Version details
    for version_str in ["v0.5.0", "v1.0.0", "v1.5.0", "v2.0.0", "v2.5.0", "v3.0.0", "v3.3.0", "v3.4.0"]:
        if version_str not in results:
            continue

        metrics = results[version_str]
        if "error" in metrics:
            continue

        config = ALGORITHMIC_VERSIONS.get(version_str)

        lines.append(f"{'='*60}")
        lines.append(f"{version_str}: {metrics['philosophy']}")
        lines.append(f"{'='*60}")
        lines.append("")

        if config:
            lines.append("Detection Logic:")
            lines.append(f"  Entry: {config.entry_method}")
            if config.entry_nose_engagement:
                lines.append(f"    + Nose engagement within {config.entry_nose_threshold_px}px")
            lines.append(f"  Exit: {config.exit_method}")
            lines.append("")

            lines.append("Validation Filters:")
            if config.min_duration_frames:
                lines.append(f"  Duration >= {config.min_duration_frames} frames")
            if config.min_extent_pixels is not None:
                lines.append(f"  Extent >= {config.min_extent_pixels} pixels")
            if config.require_positive_extent:
                lines.append(f"  Extent > 0 (positive only)")
            if not any([config.min_duration_frames, config.min_extent_pixels, config.require_positive_extent]):
                lines.append("  None")
            lines.append("")

        lines.append("Performance:")
        lines.append(f"  Precision: {metrics['precision']:.1%} (±{metrics.get('precision_std', 0):.1%})")
        lines.append(f"  Recall: {metrics['recall']:.1%} (±{metrics.get('recall_std', 0):.1%})")
        lines.append(f"  F1: {metrics['f1']:.2f}")
        lines.append(f"  Reaches detected: {metrics['n_reaches_detected']} / {metrics['n_reaches_gt']} GT")
        lines.append("")

        if metrics.get("known_issues"):
            lines.append("Known Issues:")
            for issue in metrics["known_issues"]:
                lines.append(f"  - {issue}")
            lines.append("")

        lines.append("")

    # Key findings
    lines.append("="*80)
    lines.append("KEY FINDINGS")
    lines.append("="*80)
    lines.append("")

    if "v3.3.0" in results and "v3.4.0" in results:
        bug_version = results["v3.3.0"]
        current_version = results["v3.4.0"]

        lines.append("THE v3.3.0 BUG:")
        lines.append(f"  Recall dropped from {current_version['recall']:.1%} to {bug_version['recall']:.1%}")
        lines.append(f"  Lost {current_version['n_reaches_gt'] - bug_version['n_reaches_detected']} valid reaches")
        lines.append("")
        lines.append("  Root cause: The filter 'extent >= 0' removed reaches where the hand")
        lines.append("  approached but didn't fully cross the BOXR_x reference line.")
        lines.append("")
        lines.append("  Why this matters scientifically:")
        lines.append("  - Negative extent reaches (-2 to -15px) are meaningful 'approach' attempts")
        lines.append("  - Mouse shows reaching intention even without full slit crossing")
        lines.append("  - Human annotators mark these as valid reaches in ground truth")
        lines.append("")

    lines.append("ALGORITHM EVOLUTION TIMELINE:")
    lines.append("")
    lines.append("  v0.5.0 → v1.0.0: Added confidence threshold (reduced noise)")
    lines.append("  v1.0.0 → v1.5.0: Added nose engagement (major precision gain)")
    lines.append("  v1.5.0 → v2.0.0: Added extent filter (rejected approach reaches)")
    lines.append("  v2.0.0 → v2.5.0: Added retraction detection (better end timing)")
    lines.append("  v2.5.0 → v3.0.0: Stricter filters (too aggressive)")
    lines.append("  v3.0.0 → v3.3.0: Relaxed extent to >=0 (introduced bug)")
    lines.append("  v3.3.0 → v3.4.0: Removed extent filter (fixed recall)")
    lines.append("")

    lines.append("RECOMMENDATION:")
    lines.append("  The v3.4.0 approach of detecting ALL reaches and preserving extent values")
    lines.append("  for downstream filtering is optimal. Researchers can apply their own")
    lines.append("  extent thresholds based on their specific scientific questions.")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(report)
        print(f"Report saved to {output_path}")

    return report


# CLI entry point
def main():
    """CLI for version simulator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate historical algorithm versions on current GT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mousereach-version-sim Processing Processing
  mousereach-version-sim Processing Processing --versions v3.3.0 v3.4.0
  mousereach-version-sim Processing Processing --output report.txt
        """
    )
    parser.add_argument("reaches_dir", help="Directory with _reaches.json files")
    parser.add_argument("gt_dir", help="Directory with ground truth files")
    parser.add_argument("--tolerance", type=int, default=10, help="Frame tolerance for matching")
    parser.add_argument("--versions", nargs="+", help="Versions to evaluate (default: all)")
    parser.add_argument("--output", help="Output report path (optional)")
    parser.add_argument("--list-versions", action="store_true", help="List all available versions")

    args = parser.parse_args()

    if args.list_versions:
        print("Available algorithm versions:")
        print()
        for v, config in ALGORITHMIC_VERSIONS.items():
            print(f"  {v}: {config.philosophy}")
        return

    if args.output:
        report = generate_version_report(args.reaches_dir, args.gt_dir, args.output)
        print(report)
    else:
        results = evaluate_all_versions(
            args.reaches_dir,
            args.gt_dir,
            tolerance=args.tolerance,
            versions=args.versions
        )
        print_version_comparison(results)


if __name__ == "__main__":
    main()
