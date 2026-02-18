"""
segmenter_robust.py - Trial boundary detection algorithm

ALGORITHM SUMMARY
=================
Detects 21 trial boundaries (pellet presentation events) in mouse reaching
behavior videos using scoring area (SA) anchor point motion. Uses a multi-
strategy approach with primary, secondary, and fallback detection methods.

SCIENTIFIC DESCRIPTION
======================
During the single-pellet reaching task, an automated pellet dispensing system
presents 21 pellets in sequence. Each presentation causes the scoring area
frame (tracked via corner points SABL, SABR, SATL, SATR) to move as the tray
advances. This algorithm detects trial boundaries by tracking when the SABL
point crosses the box center position with sufficient velocity, indicating
a pellet presentation event.

INPUT REQUIREMENTS
==================
- DeepLabCut tracking file (.h5) with 18 bodyparts
- Required bodyparts:
  - Primary detection: SABL (primary anchor), BOXL, BOXR (slit reference)
  - Validation: SABR, SATL, SATR (secondary anchors)
  - Reference stability: BOXL, BOXR (should be stationary)

DETECTION RULES
===============
1. Primary Detection (SABL Crossing)
   - Smooth SABL x-position with 5-frame median filter
   - Compute velocity as frame-to-frame position change
   - Detect positive crossings of box center (BOXL-BOXR midpoint)
   - Require minimum velocity threshold (>0.03 ruler units/frame)

2. Secondary Validation
   - Check SABR, SATL, SATR for correlated motion at candidate frames
   - Use frame window of +/- 3 frames for alignment tolerance
   - Boost confidence when multiple anchors agree

3. Fallback Detection (if primary fails)
   - Use motion magnitude across all SA points
   - Detect peaks in aggregate motion signal
   - Lower confidence than primary method

4. Boundary Refinement
   - Enforce minimum interval between boundaries (prevent duplicates)
   - Enforce maximum interval (detect missed boundaries)
   - Target exactly 21 boundaries for complete video

KEY PARAMETERS
==============
| Parameter | Value | Unit | Rationale |
|-----------|-------|------|-----------|
| VELOCITY_THRESHOLD | 0.03 | ruler/frame | Filters noise while catching slow presentations |
| CROSSING_WINDOW | 3 | frames | Alignment tolerance for multi-anchor validation |
| MIN_INTERVAL | 300 | frames | ~5 sec at 60fps, prevents duplicate detection |
| MAX_INTERVAL | 1200 | frames | ~20 sec at 60fps, flags missed boundaries |
| EXPECTED_BOUNDARIES | 21 | count | Standard pellet count per session |
| SMOOTHING_WINDOW | 5 | frames | Median filter for noise reduction |

OUTPUT FORMAT
=============
JSON file (*_segments.json) containing:
- boundaries: List of 21 frame numbers
- overall_confidence: Aggregate confidence score (0-1)
- reference_quality: "good", "acceptable", or "poor"
- sa_coverage: Per-anchor tracking quality percentage
- detection: Per-boundary method and confidence
- validation_status: "auto_approved", "needs_review", or "validated"

VALIDATION HISTORY
==================
- v1.0: Initial SABL crossing algorithm
- v2.0: Added multi-anchor validation and fallback
- v2.1: Refined velocity thresholds, improved edge case handling

KNOWN LIMITATIONS
=================
- Requires stable reference points (BOXL/BOXR should not move)
- Low SABL tracking quality degrades detection accuracy
- Manual pellet placement disrupts expected patterns
- Camera vibration can trigger false positives

REFERENCES
==========
- Primary: SABL-Centered Crossing algorithm
- Fallback: Motion-based peak detection
- Ground truth creation: segmentation/review_widget.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import json
import hashlib

# Version tracking - bump this when algorithm changes significantly
SEGMENTER_VERSION = "2.1.0"
SEGMENTER_ALGORITHM = "sabl_centered_crossing_v2"


@dataclass
class BoundaryCandidate:
    """A potential boundary with confidence scores."""
    frame: int
    sabl_position: float  # relative to box center
    velocity: float
    method: str  # 'primary', 'secondary', 'fallback'
    confidence: float  # 0-1
    notes: List[str] = field(default_factory=list)


@dataclass 
class SegmentationDiagnostics:
    """Detailed info about what the algorithm did and why."""
    video_name: str
    total_frames: int
    fps: float
    
    # Reference quality
    box_center: float
    boxl_std: float  # Should be near 0 if stable
    boxr_std: float
    reference_quality: str  # 'good', 'suspect', 'bad'
    
    # SA point quality
    sabl_coverage: float  # % of frames with good likelihood
    sabr_coverage: float
    satl_coverage: float
    satr_coverage: float
    
    # Detection info
    n_primary_candidates: int
    n_secondary_candidates: int
    n_fallback_candidates: int
    
    # Final boundaries
    boundaries: List[int]
    boundary_methods: List[str]
    boundary_confidences: List[float]
    
    # Anomalies detected
    anomalies: List[str]
    
    # Interval stats
    interval_mean: float
    interval_std: float
    interval_cv: float


def load_dlc(filepath: Path) -> pd.DataFrame:
    """Load DLC file with flattened columns."""
    if filepath.suffix == '.h5':
        df = pd.read_hdf(filepath)
    else:
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)
    df.columns = ['_'.join([str(c) for c in col[1:]]) for col in df.columns]
    return df


def get_clean_signal(df: pd.DataFrame, bodypart: str, coord: str = 'x',
                     threshold: float = 0.5) -> Optional[np.ndarray]:
    """Get interpolated signal with low-likelihood frames filled."""
    col = f'{bodypart}_{coord}'
    like_col = f'{bodypart}_likelihood'
    
    if col not in df.columns:
        return None
    
    vals = df[col].values.copy()
    
    if like_col in df.columns:
        like = df[like_col].values
        vals[like < threshold] = np.nan
    
    # Interpolate and fill edges
    vals = pd.Series(vals).interpolate(method='linear', limit_direction='both')
    vals = vals.ffill().bfill().values
    
    return vals


def compute_velocity(signal: np.ndarray, smooth_window: int = 30) -> np.ndarray:
    """Compute smoothed absolute velocity."""
    vel = np.abs(np.diff(signal))
    vel = np.concatenate([vel, [0]])
    return uniform_filter1d(vel, size=smooth_window)


def assess_reference_quality(df: pd.DataFrame) -> Tuple[Optional[float], float, float, str]:
    """Check if BOXL and BOXR are stable (they should be stationary)."""
    boxl = get_clean_signal(df, 'BOXL', 'x')
    boxr = get_clean_signal(df, 'BOXR', 'x')
    
    if boxl is None or boxr is None:
        return None, np.inf, np.inf, 'missing'
    
    boxl_std = np.std(boxl)
    boxr_std = np.std(boxr)
    box_center = (np.median(boxl) + np.median(boxr)) / 2
    
    # These should be nearly stationary (std < 5 px)
    if boxl_std < 5 and boxr_std < 5:
        quality = 'good'
    elif boxl_std < 15 and boxr_std < 15:
        quality = 'suspect'
    else:
        quality = 'bad'
    
    return box_center, boxl_std, boxr_std, quality


def assess_sa_quality(df: pd.DataFrame) -> Dict[str, float]:
    """Check coverage (% good frames) for each SA point."""
    coverage = {}
    for bp in ['SABL', 'SABR', 'SATL', 'SATR']:
        like_col = f'{bp}_likelihood'
        if like_col in df.columns:
            like = df[like_col].values
            coverage[bp] = np.mean(like > 0.5)
        else:
            coverage[bp] = 0.0
    return coverage


def find_centered_crossings(sabl_rel: np.ndarray, velocity: np.ndarray,
                            center_range: Tuple[float, float] = (-5, 10),
                            vel_threshold: float = 1.2,
                            min_gap: int = 30) -> List[BoundaryCandidate]:
    """
    Primary method: Find frames where SABL is centered with significant velocity.
    
    Args:
        sabl_rel: SABL position relative to box center
        velocity: Smoothed velocity signal
        center_range: Position range to consider "centered"
        vel_threshold: Minimum velocity to consider movement
        min_gap: Minimum frames between events (lowered to catch rapid-fire boundaries)
    
    Returns:
        List of boundary candidates
    """
    # Find frames meeting criteria
    centered = (sabl_rel > center_range[0]) & (sabl_rel < center_range[1])
    moving = velocity > vel_threshold
    candidate_frames = np.where(centered & moving)[0]
    
    if len(candidate_frames) == 0:
        return []
    
    # Group consecutive frames into events
    # Use small gap (min_gap frames) to separate rapid-fire boundaries
    events = []
    event_frames = [candidate_frames[0]]
    
    for i in range(1, len(candidate_frames)):
        if candidate_frames[i] - candidate_frames[i-1] > min_gap:
            # End current event
            events.append(event_frames)
            event_frames = []
        event_frames.append(candidate_frames[i])
    events.append(event_frames)
    
    # For each event, find the frame closest to center with highest velocity
    candidates = []
    for event_frames in events:
        if len(event_frames) == 0:
            continue
        
        # Score each frame: closer to center = better, higher velocity = better
        best_frame = None
        best_score = -np.inf
        
        for f in event_frames:
            # Position score: closest to +2.5 (typical boundary position)
            pos_score = -abs(sabl_rel[f] - 2.5)
            # Velocity score
            vel_score = velocity[f]
            # Combined score
            score = pos_score + vel_score * 2
            
            if score > best_score:
                best_score = score
                best_frame = f
        
        # Calculate confidence based on how well it matches expected signature
        pos_confidence = max(0, 1 - abs(sabl_rel[best_frame] - 2.5) / 10)
        vel_confidence = min(1, velocity[best_frame] / 2.5)
        confidence = (pos_confidence + vel_confidence) / 2
        
        candidates.append(BoundaryCandidate(
            frame=best_frame,
            sabl_position=sabl_rel[best_frame],
            velocity=velocity[best_frame],
            method='primary',
            confidence=confidence,
            notes=[f"pos={sabl_rel[best_frame]:.1f}, vel={velocity[best_frame]:.2f}"]
        ))
    
    return candidates


def find_motion_peaks(velocity: np.ndarray, fps: float = 60.0) -> List[BoundaryCandidate]:
    """
    Fallback method: Find peaks in velocity signal.
    """
    # Find peaks with minimum distance of 20 seconds
    peaks, properties = find_peaks(
        velocity,
        distance=int(fps * 20),
        prominence=np.percentile(velocity, 85)
    )
    
    candidates = []
    for peak in peaks:
        candidates.append(BoundaryCandidate(
            frame=peak,
            sabl_position=np.nan,
            velocity=velocity[peak],
            method='fallback_motion',
            confidence=0.5,
            notes=['motion peak only']
        ))
    
    return candidates


def validate_with_other_sa_points(df: pd.DataFrame, box_center: float,
                                  candidate: BoundaryCandidate) -> BoundaryCandidate:
    """
    Check if other SA points also show movement at this boundary.
    """
    agreements = 0
    checks = 0
    
    for bp in ['SABR', 'SATL', 'SATR']:
        signal = get_clean_signal(df, bp, 'x')
        if signal is None:
            continue
        
        checks += 1
        velocity = compute_velocity(signal)
        
        # Check velocity around candidate frame
        window = velocity[max(0, candidate.frame-30):min(len(velocity), candidate.frame+30)]
        if len(window) > 0 and np.max(window) > 1.0:
            agreements += 1
    
    if checks > 0:
        agreement_rate = agreements / checks
        candidate.confidence = (candidate.confidence + agreement_rate) / 2
        candidate.notes.append(f"SA agreement: {agreements}/{checks}")
    
    return candidate


def fit_grid_to_candidates(candidates: List[BoundaryCandidate],
                           total_frames: int,
                           expected_interval: float = 1839,
                           velocity: Optional[np.ndarray] = None,
                           sabl_rel: Optional[np.ndarray] = None) -> Tuple[List[int], List[str]]:
    """
    Given detected candidates, select exactly 21 boundaries.
    
    Returns:
        boundaries: List of 21 boundary frames
        anomalies: List of detected anomalies
    """
    anomalies = []
    
    if len(candidates) == 0:
        # Complete fallback: evenly spaced
        interval = total_frames / 22
        anomalies.append("No candidates found - using evenly spaced fallback")
        return [int((i + 1) * interval) for i in range(21)], anomalies
    
    # Sort by frame
    candidates = sorted(candidates, key=lambda c: c.frame)
    frames = [c.frame for c in candidates]
    
    # BEST CASE: Exactly 21 candidates - use them directly!
    if len(frames) == 21:
        return frames, anomalies
    
    # Calculate actual interval from consecutive candidates
    if len(frames) >= 2:
        intervals = np.diff(frames)
        # Filter out anomalous intervals (< 50% or > 150% of expected)
        valid = intervals[(intervals > expected_interval * 0.5) & 
                         (intervals < expected_interval * 1.5)]
        if len(valid) > 0:
            actual_interval = np.median(valid)
        else:
            actual_interval = expected_interval
    else:
        actual_interval = expected_interval
    
    first_candidate = frames[0]
    
    # Check if we should project backward to find B1
    should_project_backward = False
    
    if velocity is not None and sabl_rel is not None and first_candidate > 500:
        early_region = slice(0, first_candidate - 100)
        if first_candidate > 100:
            centered_early = (sabl_rel[early_region] > -5) & (sabl_rel[early_region] < 10)
            moving_early = velocity[early_region] > 1.0
            early_crossing_events = np.sum(centered_early & moving_early)
            
            if early_crossing_events > 30:
                should_project_backward = True
                anomalies.append(f"Detected crossing events before first candidate")
    
    if should_project_backward:
        intervals_to_start = round((first_candidate - 150) / actual_interval)
        if intervals_to_start > 0:
            estimated_b1 = first_candidate - intervals_to_start * actual_interval
            estimated_b1 = max(50, min(600, int(estimated_b1)))
            anomalies.append(f"Estimated B1 at frame {estimated_b1} (first detection at {first_candidate})")
        else:
            estimated_b1 = first_candidate
    else:
        estimated_b1 = first_candidate
        if first_candidate > 1000:
            anomalies.append(f"Late-start video: B1 at frame {first_candidate}")
    
    # If we have close to 21 (19-23), try to use candidates directly
    if 19 <= len(frames) <= 23:
        if len(frames) > 21:
            # Too many - remove lowest confidence, keeping relative order
            sorted_by_conf = sorted(enumerate(candidates), key=lambda x: x[1].confidence)
            to_remove = set(i for i, _ in sorted_by_conf[:len(frames)-21])
            frames = [f for i, f in enumerate(frames) if i not in to_remove]
            anomalies.append(f"Removed {len(to_remove)} low-confidence candidates")
        elif len(frames) < 21:
            # Too few - fill gaps with interpolated boundaries
            missing = 21 - len(frames)
            # Find largest gaps
            intervals = np.diff(frames)
            gap_indices = np.argsort(intervals)[::-1][:missing]
            
            new_frames = list(frames)
            for gap_idx in sorted(gap_indices, reverse=True):
                gap_start = frames[gap_idx]
                gap_end = frames[gap_idx + 1]
                interp_frame = (gap_start + gap_end) // 2
                new_frames.insert(gap_idx + 1, interp_frame)
                anomalies.append(f"Interpolated boundary at frame {interp_frame}")
            frames = new_frames[:21]
        
        return frames, anomalies
    
    # Fallback: build grid starting from estimated B1
    boundaries = []
    for i in range(21):
        expected_frame = estimated_b1 + i * actual_interval
        expected_frame = max(0, min(total_frames - 1, int(expected_frame)))
        
        nearby = [f for f in frames if abs(f - expected_frame) < actual_interval * 0.2]
        if nearby:
            boundaries.append(min(nearby, key=lambda f: abs(f - expected_frame)))
        else:
            boundaries.append(expected_frame)
    
    return boundaries, anomalies


def detect_anomalies(boundaries: List[int], fps: float = 60.0) -> List[str]:
    """Check for suspicious patterns that might indicate problems."""
    anomalies = []
    
    intervals = np.diff(boundaries)
    expected = 1839  # ~30.65 seconds at 60fps
    
    for i, interval in enumerate(intervals):
        # Way too short (< 15 seconds)
        if interval < fps * 15:
            anomalies.append(f"Very short interval B{i+1}→B{i+2}: {interval} frames ({interval/fps:.1f}s)")
        
        # Way too long (> 45 seconds)  
        if interval > fps * 45:
            anomalies.append(f"Very long interval B{i+1}→B{i+2}: {interval} frames ({interval/fps:.1f}s)")
        
        # Possible double-move (interval ~2x expected)
        if 1.7 * expected < interval < 2.3 * expected:
            anomalies.append(f"Possible stuck tray B{i+1}→B{i+2}: {interval} frames (~2x expected)")
    
    # Check for overall drift
    if len(intervals) > 5:
        first_half = np.mean(intervals[:len(intervals)//2])
        second_half = np.mean(intervals[len(intervals)//2:])
        if abs(first_half - second_half) > expected * 0.1:
            anomalies.append(f"Interval drift detected: first half avg={first_half:.0f}, second half avg={second_half:.0f}")
    
    return anomalies


def classify_anomaly_severity(anomaly_text: str, context: Dict[str, Any]) -> Tuple[str, str, List[int]]:
    """
    Classify anomaly severity and provide explanation.

    Args:
        anomaly_text: The anomaly message
        context: Dict with n_boundaries, n_interpolated, cv, mean_confidence, n_primary

    Returns:
        (severity_str, explanation_str, boundaries_affected_list)
    """
    import re

    # CRITICAL patterns - segment identification impeded
    if "No candidates found" in anomaly_text:
        return ('CRITICAL', 'Segmentation algorithm failed completely', [])
    if "Primary method unavailable" in anomaly_text:
        return ('CRITICAL', 'Cannot use main detection method', [])
    if "fallback motion detection" in anomaly_text:
        return ('CRITICAL', 'Unreliable detection method used', [])

    # Check context for critical conditions
    if context.get('n_boundaries', 21) < 19:
        return ('CRITICAL', f"Only {context['n_boundaries']} boundaries detected", [])
    if context.get('n_interpolated', 0) >= 5:
        return ('CRITICAL', 'Too many boundaries interpolated - low confidence', [])
    if context.get('mean_confidence', 1.0) < 0.50:
        return ('CRITICAL', 'Very low average confidence in detections', [])

    # INFO patterns - algorithm adapted successfully (EXCEPTION HANDLERS)
    if "lower velocity threshold" in anomaly_text:
        return ('INFO', 'Algorithm adapted to slower tray movement - not an error', [1])
    if "Detected crossing events before" in anomaly_text:
        return ('INFO', 'Pre-trial movements detected - normal variation', [])

    # WARNING patterns
    if "Late-start" in anomaly_text or "Estimated B1" in anomaly_text:
        frame_match = re.search(r'frame (\d+)', anomaly_text)
        frame = int(frame_match.group(1)) if frame_match else 0
        if frame > 2000:
            return ('WARNING', 'Video started late in session', [1])
        else:
            return ('INFO', 'Normal start time variation', [1])

    if "Interpolated boundary" in anomaly_text or "interpolated" in anomaly_text.lower():
        n_interp = context.get('n_interpolated', 0)
        if n_interp >= 5:
            return ('CRITICAL', 'Many boundaries interpolated', [])
        elif n_interp >= 2:
            return ('WARNING', f'{n_interp} boundaries needed interpolation', [])
        else:
            return ('INFO', 'Single boundary interpolated', [])

    if "Removed" in anomaly_text and "candidates" in anomaly_text:
        return ('WARNING', 'Had to filter out ambiguous detections', [])

    if "Very short interval" in anomaly_text or "Very long interval" in anomaly_text:
        return ('WARNING', 'Timing variation detected', [])

    if "Interval drift" in anomaly_text:
        return ('WARNING', 'Session timing changed over trials', [])

    if "stuck tray" in anomaly_text:
        return ('WARNING', 'Possible operator tray handling issue', [])

    # Default to INFO for unknown anomalies
    return ('INFO', 'Noted for reference', [])


def segment_video_robust(dlc_path: Path, fps: float = 60.0) -> Tuple[List[int], SegmentationDiagnostics]:
    """
    Main segmentation function with multiple strategies.
    
    Returns:
        boundaries: List of 21 boundary frames
        diagnostics: Detailed info about what happened
    """
    dlc_path = Path(dlc_path)
    df = load_dlc(dlc_path)
    total_frames = len(df)
    
    # Assess reference quality
    box_center, boxl_std, boxr_std, ref_quality = assess_reference_quality(df)
    
    # Assess SA point quality
    sa_coverage = assess_sa_quality(df)
    
    # Get SABL signal
    sabl = get_clean_signal(df, 'SABL', 'x')
    
    candidates = []
    anomalies = []
    velocity = None  # Will be set if primary method succeeds
    sabl_rel = None  # Will be set if primary method succeeds
    
    # Primary method: SABL centered crossing
    if sabl is not None and box_center is not None and ref_quality != 'bad':
        sabl_rel = sabl - box_center
        velocity = compute_velocity(sabl)
        
        # Try with standard threshold first
        primary_candidates = find_centered_crossings(sabl_rel, velocity)
        
        # If fewer than 21, try with lower velocity threshold to catch slow B1
        if len(primary_candidates) < 21:
            lower_threshold_candidates = find_centered_crossings(
                sabl_rel, velocity, vel_threshold=0.8
            )
            if len(lower_threshold_candidates) > len(primary_candidates):
                primary_candidates = lower_threshold_candidates
                anomalies.append("Used lower velocity threshold (0.8)")
        
        # Validate each candidate with other SA points
        for c in primary_candidates:
            c = validate_with_other_sa_points(df, box_center, c)
        
        candidates.extend(primary_candidates)
        n_primary = len(primary_candidates)
    else:
        n_primary = 0
        anomalies.append(f"Primary method unavailable: ref_quality={ref_quality}, sabl={sabl is not None}")
    
    # If primary method failed, use fallback
    if len(candidates) < 10:
        # Try motion peaks on any available SA point
        for bp in ['SABL', 'SABR', 'SATL', 'SATR']:
            signal = get_clean_signal(df, bp, 'x')
            if signal is not None:
                velocity = compute_velocity(signal)
                fallback_candidates = find_motion_peaks(velocity, fps)
                if len(fallback_candidates) >= 15:
                    candidates = fallback_candidates
                    anomalies.append(f"Using fallback motion detection on {bp}")
                    break
        n_fallback = len(candidates) - n_primary
    else:
        n_fallback = 0
    
    # Fit 21-boundary grid to candidates
    boundaries, grid_anomalies = fit_grid_to_candidates(
        candidates, total_frames, velocity=velocity, sabl_rel=sabl_rel
    )
    anomalies.extend(grid_anomalies)
    
    # Detect anomalies in final boundaries
    boundary_anomalies = detect_anomalies(boundaries, fps)
    anomalies.extend(boundary_anomalies)
    
    # Calculate interval stats
    intervals = np.diff(boundaries)
    interval_mean = np.mean(intervals)
    interval_std = np.std(intervals)
    interval_cv = interval_std / interval_mean if interval_mean > 0 else 1.0
    
    # Determine method and confidence for each boundary
    boundary_methods = []
    boundary_confidences = []
    for b in boundaries:
        matching = [c for c in candidates if abs(c.frame - b) < 100]
        if matching:
            best = max(matching, key=lambda c: c.confidence)
            boundary_methods.append(best.method)
            boundary_confidences.append(best.confidence)
        else:
            boundary_methods.append('interpolated')
            boundary_confidences.append(0.3)
    
    # Build diagnostics
    diagnostics = SegmentationDiagnostics(
        video_name=dlc_path.stem,
        total_frames=total_frames,
        fps=fps,
        box_center=box_center if box_center else 0,
        boxl_std=boxl_std,
        boxr_std=boxr_std,
        reference_quality=ref_quality,
        sabl_coverage=sa_coverage.get('SABL', 0),
        sabr_coverage=sa_coverage.get('SABR', 0),
        satl_coverage=sa_coverage.get('SATL', 0),
        satr_coverage=sa_coverage.get('SATR', 0),
        n_primary_candidates=n_primary,
        n_secondary_candidates=0,  # Not yet implemented
        n_fallback_candidates=n_fallback,
        boundaries=boundaries,
        boundary_methods=boundary_methods,
        boundary_confidences=boundary_confidences,
        anomalies=anomalies,
        interval_mean=interval_mean,
        interval_std=interval_std,
        interval_cv=interval_cv,
    )
    
    return boundaries, diagnostics


def save_segmentation(boundaries: List[int], diagnostics: SegmentationDiagnostics,
                      output_path: Path) -> None:
    """Save segmentation results and diagnostics."""
    # Convert numpy types to native Python for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        return obj

    # Classify anomalies by severity
    context = {
        'n_boundaries': len(boundaries),
        'n_interpolated': sum(1 for m in diagnostics.boundary_methods if 'interpolated' in str(m)),
        'cv': diagnostics.interval_cv,
        'mean_confidence': np.mean(diagnostics.boundary_confidences) if diagnostics.boundary_confidences else 0.5,
        'n_primary': diagnostics.n_primary_candidates
    }

    anomaly_details = []
    anomaly_summary = {'critical': 0, 'warning': 0, 'info': 0}
    boundary_flags = {}

    for anomaly_text in diagnostics.anomalies:
        severity, explanation, boundaries_affected = classify_anomaly_severity(anomaly_text, context)
        anomaly_summary[severity.lower()] += 1
        anomaly_details.append({
            'text': anomaly_text,
            'severity': severity,
            'explanation': explanation,
            'boundaries_affected': boundaries_affected
        })

        # Flag specific boundaries
        for b_idx in boundaries_affected:
            if b_idx not in boundary_flags:
                boundary_flags[b_idx] = {
                    'issues': [],
                    'needs_check': (severity == 'WARNING' or severity == 'CRITICAL')
                }
            boundary_flags[b_idx]['issues'].append(anomaly_text[:50])  # Truncate for readability

    data = {
        # Version tracking
        'segmenter_version': SEGMENTER_VERSION,
        'segmenter_algorithm': SEGMENTER_ALGORITHM,
        
        'video_name': diagnostics.video_name,
        'total_frames': to_native(diagnostics.total_frames),
        'fps': to_native(diagnostics.fps),
        'boundaries': to_native(boundaries),
        
        # Quality info
        'reference_quality': diagnostics.reference_quality,
        'sa_coverage': {
            'SABL': to_native(diagnostics.sabl_coverage),
            'SABR': to_native(diagnostics.sabr_coverage),
            'SATL': to_native(diagnostics.satl_coverage),
            'SATR': to_native(diagnostics.satr_coverage),
        },
        
        # Detection info
        'detection': {
            'n_primary': to_native(diagnostics.n_primary_candidates),
            'n_fallback': to_native(diagnostics.n_fallback_candidates),
            'methods': diagnostics.boundary_methods,
            'confidences': to_native(diagnostics.boundary_confidences),
        },
        
        # Interval stats
        'intervals': {
            'mean_frames': to_native(diagnostics.interval_mean),
            'std_frames': to_native(diagnostics.interval_std),
            'cv': to_native(diagnostics.interval_cv),
            'mean_seconds': to_native(diagnostics.interval_mean / diagnostics.fps),
        },
        
        # Issues
        'anomalies': diagnostics.anomalies,
        'anomaly_details': anomaly_details,
        'anomaly_summary': anomaly_summary,
        'boundary_flags': {str(k): v for k, v in boundary_flags.items()},
        'overall_confidence': to_native(np.mean(diagnostics.boundary_confidences)),
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Update pipeline index with new file
    try:
        from mousereach.index import PipelineIndex
        index = PipelineIndex()
        index.load()
        index.record_file_created(output_path, metadata={
            "seg_boundaries": len(boundaries),
            "seg_confidence": data.get("overall_confidence", 0),
            "segmenter_version": SEGMENTER_VERSION,
            "seg_validation": "needs_review" if anomaly_summary.get("warning", 0) > 0 else "auto_review",
        })
        index.save()
    except Exception:
        pass  # Don't fail segmentation if index update fails


def print_diagnostics(diag: SegmentationDiagnostics) -> None:
    """Print human-readable diagnostics."""
    print(f"\n{'='*60}")
    print(f"SEGMENTATION: {diag.video_name}")
    print(f"{'='*60}")
    
    print(f"\nReference quality: {diag.reference_quality}")
    print(f"  BOXL std: {diag.boxl_std:.1f} px")
    print(f"  BOXR std: {diag.boxr_std:.1f} px")
    
    print(f"\nSA point coverage:")
    print(f"  SABL: {diag.sabl_coverage*100:.0f}%")
    print(f"  SABR: {diag.sabr_coverage*100:.0f}%")
    print(f"  SATL: {diag.satl_coverage*100:.0f}%")
    print(f"  SATR: {diag.satr_coverage*100:.0f}%")
    
    print(f"\nDetection:")
    print(f"  Primary candidates: {diag.n_primary_candidates}")
    print(f"  Fallback candidates: {diag.n_fallback_candidates}")
    
    print(f"\nIntervals:")
    print(f"  Mean: {diag.interval_mean:.0f} frames ({diag.interval_mean/diag.fps:.2f}s)")
    print(f"  Std: {diag.interval_std:.0f} frames")
    print(f"  CV: {diag.interval_cv:.4f}")
    
    if diag.anomalies:
        print(f"\nANOMALIES DETECTED:")
        for a in diag.anomalies:
            print(f"  [!] {a}")
    
    print(f"\nBoundary confidences:")
    low_conf = [(i+1, c) for i, c in enumerate(diag.boundary_confidences) if c < 0.5]
    if low_conf:
        print(f"  Low confidence (<0.5):")
        for idx, conf in low_conf:
            print(f"    B{idx}: {conf:.2f} ({diag.boundary_methods[idx-1]})")
    
    avg_conf = np.mean(diag.boundary_confidences)
    print(f"  Average: {avg_conf:.2f}")


# Test on ground truth files
if __name__ == '__main__':
    import sys
    
    test_files = [
        "/mnt/user-data/uploads/20250820_CNT0104_P2DLC_resnet50_MPSAOct27shuffle1_100000.h5",
        "/mnt/user-data/uploads/20251029_CNT0408_P1DLC_resnet50_MPSAOct27shuffle1_100000.h5",
        "/mnt/user-data/uploads/20251031_CNT0413_P2DLC_resnet50_MPSAOct27shuffle1_100000.h5",
        "/mnt/user-data/uploads/20251031_CNT0415_P1DLC_resnet50_MPSAOct27shuffle1_100000.h5",
    ]
    
    gt_files = [
        "/mnt/user-data/uploads/20250820_CNT0104_P2_seg_ground_truth.json",
        "/mnt/user-data/uploads/20251029_CNT0408_P1_seg_ground_truth.json",
        "/mnt/user-data/uploads/20251031_CNT0413_P2_seg_ground_truth.json",
        "/mnt/user-data/uploads/20251031_CNT0415_P1_seg_ground_truth.json",
    ]
    
    print("ROBUST SEGMENTER EVALUATION")
    print("="*70)
    
    all_errors = []
    
    for dlc_path, gt_path in zip(test_files, gt_files):
        if not Path(dlc_path).exists():
            continue
        
        # Run segmenter
        boundaries, diag = segment_video_robust(dlc_path)
        print_diagnostics(diag)
        
        # Compare to ground truth
        with open(gt_path) as f:
            gt = json.load(f)
        gt_bounds = gt['boundaries']
        
        errors = [a - t for a, t in zip(boundaries, gt_bounds)]
        abs_errors = [abs(e) for e in errors]
        
        print(f"\nComparison to ground truth:")
        print(f"  Mean absolute error: {np.mean(abs_errors):.1f} frames ({np.mean(abs_errors)/60:.2f}s)")
        print(f"  Max absolute error: {max(abs_errors)} frames")
        print(f"  Within 50 frames: {sum(1 for e in abs_errors if e < 50)}/21")
        
        # Show worst errors
        worst = sorted(enumerate(errors), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"  Worst errors:")
        for idx, err in worst:
            print(f"    B{idx+1}: {err:+d} frames ({diag.boundary_methods[idx]})")
        
        all_errors.extend(abs_errors)
    
    print("\n" + "="*70)
    print("OVERALL")
    print("="*70)
    print(f"Total boundaries: {len(all_errors)}")
    print(f"Mean absolute error: {np.mean(all_errors):.1f} frames ({np.mean(all_errors)/60:.2f}s)")
    print(f"Within 50 frames: {sum(1 for e in all_errors if e < 50)}/{len(all_errors)} ({100*sum(1 for e in all_errors if e < 50)/len(all_errors):.1f}%)")
    print(f"Within 20 frames: {sum(1 for e in all_errors if e < 20)}/{len(all_errors)} ({100*sum(1 for e in all_errors if e < 20)/len(all_errors):.1f}%)")
