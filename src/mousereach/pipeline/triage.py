"""
triage.py - Unified video triage for the MouseReach pipeline.

Replaces the per-step triage modules with a single, evidence-based system.
Flags videos that genuinely need human review based on:

1. DLC tracking coherence (fixed-point stability, ruler consistency, velocity outliers)
2. Structural integrity (segment count, missing files)
3. Cross-step consistency (outcome vs reach count mismatches)
4. Statistical outlier detection (segments that don't match peers)
5. Detector-reported uncertainty (explicit "I don't know" from algorithms)

Design principles:
- Data defines normal. Anomaly thresholds are derived from the video's own
  distributions, not hardcoded values. Only extreme-tail outliers are flagged.
- GT-validated accuracy: the per-step algorithms are 98-99% accurate on
  classification. Triage does NOT second-guess confident classifications.
- Flag real problems: DLC tracking failures, experimenter interventions,
  equipment issues, cross-step contradictions.

v3.0 (2026-02): Complete rewrite based on GT mismatch analysis.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Structural constants
EXPECTED_SEGMENTS_MIN = 20
EXPECTED_SEGMENTS_MAX = 21

# Outlier detection: how many IQR from median to consider extreme
# At 4.0, for a normal distribution this catches ~0.007% of data —
# only truly extreme points. Tuned conservatively to avoid false flags.
OUTLIER_IQR_MULTIPLIER = 4.0


@dataclass
class TriageFlag:
    """A single reason a video was flagged."""
    category: str       # dlc_coherence, structural, cross_step, statistical, uncertainty
    severity: str       # 'critical' or 'warning'
    description: str
    segment_num: Optional[int] = None  # None = video-level flag

    def to_dict(self):
        return asdict(self)


@dataclass
class TriageResult:
    """Complete triage result for one video."""
    video_id: str
    verdict: str          # 'auto_approved' or 'needs_review'
    flags: List[TriageFlag] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)

    @property
    def n_critical(self) -> int:
        return sum(1 for f in self.flags if f.severity == 'critical')

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.flags if f.severity == 'warning')

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['n_critical'] = self.n_critical
        d['n_warnings'] = self.n_warnings
        return d

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Layer 1: DLC Tracking Coherence
# ---------------------------------------------------------------------------

# Fixed points that should be rigid-body with the tray
TRAY_FIXED_POINTS = ['BOXL', 'BOXR', 'SABL', 'SABR']

# Points that form the ruler (distance should be constant)
RULER_POINTS = ('SABL', 'SABR')


def check_dlc_coherence(
    h5_path: Path,
    seg_boundaries: List[int],
    likelihood_threshold: float = 0.6,
) -> List[TriageFlag]:
    """
    Check DLC tracking coherence by analyzing fixed-point behavior.

    Detects:
    - Frame-to-frame velocity outliers on tray-fixed points (jumps/teleportation)
    - Ruler length instability (SABL-SABR distance changing = DLC error)
    - Correlated fixed-point jumps (experimenter tray manipulation)
    - Individual fixed-point divergence (DLC identity swap)

    Uses the video's own velocity distribution to define "extreme" —
    only flags the absolute tail of the distribution.
    """
    import pandas as pd

    flags = []

    try:
        df = pd.read_hdf(h5_path)
    except Exception as e:
        flags.append(TriageFlag(
            category='dlc_coherence',
            severity='critical',
            description=f'Could not read DLC file: {e}'
        ))
        return flags

    scorer = df.columns.get_level_values(0)[0]

    # --- Fixed-point velocity analysis ---
    # For each fixed point, compute frame-to-frame displacement.
    # Use 99.9th percentile + minimum absolute floor to define "extreme."
    # The IQR method fails on zero-inflated distributions (fixed points
    # barely move, so IQR ≈ 0 and everything looks extreme).
    MIN_VELOCITY_THRESHOLD = 3.0  # px/frame — below this is sub-pixel jitter, not a real jump

    anomalous_frames_by_point = {}

    for bp in TRAY_FIXED_POINTS:
        try:
            x = df[(scorer, bp, 'x')].values
            y = df[(scorer, bp, 'y')].values
            conf = df[(scorer, bp, 'likelihood')].values
        except KeyError:
            continue

        # Frame-to-frame displacement (only where both frames are high-confidence)
        high_conf = conf > likelihood_threshold
        dx = np.diff(x)
        dy = np.diff(y)
        velocity = np.sqrt(dx**2 + dy**2)

        # Both current and next frame must be high-confidence
        both_conf = high_conf[:-1] & high_conf[1:]
        good_vel = velocity[both_conf]

        if len(good_vel) < 100:
            continue

        # Threshold: 99.9th percentile of velocity, but no lower than MIN_VELOCITY_THRESHOLD
        p999 = np.percentile(good_vel, 99.9)
        threshold = max(p999, MIN_VELOCITY_THRESHOLD)

        # Find frames with extreme jumps
        extreme_mask = np.zeros(len(velocity), dtype=bool)
        conf_indices = np.where(both_conf)[0]
        extreme_in_good = good_vel > threshold
        extreme_mask[conf_indices[extreme_in_good]] = True

        anomalous_frames_by_point[bp] = np.where(extreme_mask)[0]

    # --- Correlated jumps: multiple fixed points jumping simultaneously ---
    # This indicates tray manipulation or camera bump.
    # Only report as aggregate events, not individual frames.
    if len(anomalous_frames_by_point) >= 2:
        # Find frames where 3+ fixed points have extreme velocity simultaneously
        frame_counts = {}
        for bp, frames in anomalous_frames_by_point.items():
            for fr in frames:
                # Use a small window (±2 frames) for "simultaneous"
                for offset in range(-2, 3):
                    key = fr + offset
                    frame_counts[key] = frame_counts.get(key, 0) + 1

        correlated_frames = sorted(
            fr for fr, count in frame_counts.items() if count >= 3
        )

        if correlated_frames:
            # Group into contiguous events
            events = _group_contiguous_frames(correlated_frames, gap=30)

            # Only flag if there are a meaningful number of events
            # A few across a 30k+ frame video is normal tray operation
            if len(events) > 20:
                flags.append(TriageFlag(
                    category='dlc_coherence',
                    severity='warning',
                    description=(
                        f'{len(events)} correlated fixed-point jump events detected '
                        f'— possible frequent tray manipulation or tracking instability'
                    )
                ))

            # Flag events that land during a segment's active period
            # (not between segments) — these could contaminate reach kinematics.
            # Aggregate when there are many to avoid per-event spam.
            in_segment_events = []
            for event_frames in events:
                mid_frame = event_frames[len(event_frames) // 2]
                seg_num = _frame_to_segment(mid_frame, seg_boundaries)
                if seg_num is not None and len(event_frames) > 5:
                    in_segment_events.append((mid_frame, len(event_frames), seg_num))

            if len(in_segment_events) > 5:
                # Many events — summarize rather than listing each one
                affected_segs = sorted(set(s for _, _, s in in_segment_events))
                flags.append(TriageFlag(
                    category='dlc_coherence',
                    severity='warning',
                    description=(
                        f'{len(in_segment_events)} tracking disruptions during active segments '
                        f'(segments {", ".join(str(s) for s in affected_segs[:8])}'
                        f'{"..." if len(affected_segs) > 8 else ""})'
                    )
                ))
            else:
                for mid_frame, n_frames, seg_num in in_segment_events:
                    flags.append(TriageFlag(
                        category='dlc_coherence',
                        severity='warning',
                        description=(
                            f'Tracking disruption ({n_frames} frames) '
                            f'during active segment at frame {mid_frame}'
                        ),
                        segment_num=seg_num
                    ))

    # --- Ruler stability ---
    # SABL-SABR distance should be nearly constant
    try:
        sabl_x = df[(scorer, 'SABL', 'x')].values
        sabl_y = df[(scorer, 'SABL', 'y')].values
        sabl_conf = df[(scorer, 'SABL', 'likelihood')].values
        sabr_x = df[(scorer, 'SABR', 'x')].values
        sabr_y = df[(scorer, 'SABR', 'y')].values
        sabr_conf = df[(scorer, 'SABR', 'likelihood')].values

        both_good = (sabl_conf > likelihood_threshold) & (sabr_conf > likelihood_threshold)
        if both_good.sum() > 100:
            ruler_dist = np.sqrt(
                (sabr_x[both_good] - sabl_x[both_good])**2 +
                (sabr_y[both_good] - sabl_y[both_good])**2
            )
            ruler_median = np.median(ruler_dist)
            ruler_std = np.std(ruler_dist)
            ruler_cv = ruler_std / ruler_median if ruler_median > 0 else 0

            if ruler_cv > 0.15:
                flags.append(TriageFlag(
                    category='dlc_coherence',
                    severity='critical',
                    description=(
                        f'Ruler (SABL-SABR) highly unstable: '
                        f'CV={ruler_cv:.3f}, std={ruler_std:.1f}px, median={ruler_median:.1f}px'
                    )
                ))
            elif ruler_cv > 0.08:
                flags.append(TriageFlag(
                    category='dlc_coherence',
                    severity='warning',
                    description=(
                        f'Ruler (SABL-SABR) moderately unstable: '
                        f'CV={ruler_cv:.3f}, std={ruler_std:.1f}px'
                    )
                ))

            # Also check for sudden ruler changes (frame-to-frame)
            # Use 99.9th percentile with minimum floor (same approach as velocity)
            ruler_diff = np.abs(np.diff(ruler_dist))
            p999 = np.percentile(ruler_diff, 99.9)
            ruler_jump_threshold = max(p999, 3.0)  # At least 3px change
            n_ruler_jumps = (ruler_diff > ruler_jump_threshold).sum()
            if n_ruler_jumps > len(ruler_diff) * 0.005:
                flags.append(TriageFlag(
                    category='dlc_coherence',
                    severity='warning',
                    description=(
                        f'Ruler length has {n_ruler_jumps} sudden changes '
                        f'(>{ruler_jump_threshold:.1f}px/frame)'
                    )
                ))
    except KeyError:
        flags.append(TriageFlag(
            category='dlc_coherence',
            severity='critical',
            description='Missing SABL or SABR — cannot compute ruler'
        ))

    # --- Hand point clustering ---
    # RightHand, RHLeft, RHOut, RHRight should be spatially close when all visible
    rh_points = ['RightHand', 'RHLeft', 'RHOut', 'RHRight']
    try:
        rh_data = {}
        for rh in rh_points:
            rh_data[rh] = {
                'x': df[(scorer, rh, 'x')].values,
                'y': df[(scorer, rh, 'y')].values,
                'conf': df[(scorer, rh, 'likelihood')].values,
            }

        # For frames where all 4 hand points are high-confidence,
        # check that they're spatially clustered
        all_conf = np.ones(len(df), dtype=bool)
        for rh in rh_points:
            all_conf &= (rh_data[rh]['conf'] > likelihood_threshold)

        if all_conf.sum() > 50:
            # Compute max pairwise distance among the 4 hand points per frame
            max_spread = np.zeros(all_conf.sum())
            xs = np.array([rh_data[rh]['x'][all_conf] for rh in rh_points])
            ys = np.array([rh_data[rh]['y'][all_conf] for rh in rh_points])

            for i in range(4):
                for j in range(i + 1, 4):
                    dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    max_spread = np.maximum(max_spread, dist)

            # Flag extreme spreads
            q1, q3 = np.percentile(max_spread, [25, 75])
            iqr = q3 - q1
            spread_threshold = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            n_scattered = (max_spread > spread_threshold).sum()
            pct = n_scattered / len(max_spread) * 100

            if pct > 1.0:
                flags.append(TriageFlag(
                    category='dlc_coherence',
                    severity='warning',
                    description=(
                        f'Hand points abnormally scattered in {n_scattered} frames '
                        f'({pct:.1f}%, spread>{spread_threshold:.0f}px)'
                    )
                ))
    except KeyError:
        pass  # Missing hand points is handled by DLC quality report

    return flags


# ---------------------------------------------------------------------------
# Layer 2: Structural Integrity
# ---------------------------------------------------------------------------

def check_structural(
    video_id: str,
    processing_dir: Path,
) -> List[TriageFlag]:
    """Check for missing files and wrong segment counts."""
    flags = []

    seg_path = processing_dir / f"{video_id}_segments.json"
    reach_path = processing_dir / f"{video_id}_reaches.json"
    outcome_path = processing_dir / f"{video_id}_pellet_outcomes.json"

    # Missing files
    for name, path in [('segments', seg_path), ('reaches', reach_path), ('outcomes', outcome_path)]:
        if not path.exists():
            flags.append(TriageFlag(
                category='structural',
                severity='critical',
                description=f'Missing {name} file: {path.name}'
            ))

    # Segment count
    if seg_path.exists():
        try:
            with open(seg_path) as f:
                data = json.load(f)
            boundaries = data.get('boundaries', [])
            n_boundaries = len(boundaries)
            if n_boundaries < EXPECTED_SEGMENTS_MIN or n_boundaries > EXPECTED_SEGMENTS_MAX + 1:
                # boundaries = segment count + 1 (fencepost), but some files
                # store boundary count differently. Check both interpretations.
                n_segments = n_boundaries  # could also be n_boundaries - 1
                flags.append(TriageFlag(
                    category='structural',
                    severity='critical',
                    description=f'Unexpected boundary count: {n_boundaries} (expected {EXPECTED_SEGMENTS_MIN}-{EXPECTED_SEGMENTS_MAX + 1})'
                ))
        except Exception as e:
            flags.append(TriageFlag(
                category='structural',
                severity='critical',
                description=f'Could not read segments file: {e}'
            ))

    return flags


# ---------------------------------------------------------------------------
# Layer 3: Cross-Step Consistency
# ---------------------------------------------------------------------------

def check_cross_step(
    video_id: str,
    processing_dir: Path,
) -> List[TriageFlag]:
    """
    Check consistency between pipeline step outputs.

    Catches contradictions that no single algorithm can self-detect:
    - Outcome implies interaction but no reaches detected
    - Outcome says untouched but many reaches detected
    - Detector explicitly reported uncertainty
    """
    flags = []

    reach_path = processing_dir / f"{video_id}_reaches.json"
    outcome_path = processing_dir / f"{video_id}_pellet_outcomes.json"

    if not reach_path.exists() or not outcome_path.exists():
        return flags  # Can't do cross-step without both files

    try:
        with open(reach_path) as f:
            reach_data = json.load(f)
        with open(outcome_path) as f:
            outcome_data = json.load(f)
    except Exception:
        return flags

    # Build reach count per segment
    reach_by_seg = {}
    for seg in reach_data.get('segments', []):
        reach_by_seg[seg['segment_num']] = seg.get(
            'n_reaches', len(seg.get('reaches', []))
        )

    for seg in outcome_data.get('segments', []):
        seg_num = seg.get('segment_num')
        outcome = seg.get('outcome', '')
        confidence = seg.get('confidence') or 0.0
        n_reaches = reach_by_seg.get(seg_num, 0)

        # Detector said "I don't know"
        if outcome == 'uncertain':
            flags.append(TriageFlag(
                category='cross_step',
                severity='critical',
                description=f'Outcome detector reported uncertain (conf={confidence:.2f})',
                segment_num=seg_num
            ))

        # Very low confidence = detector failure
        if confidence < 0.50:
            flags.append(TriageFlag(
                category='cross_step',
                severity='critical',
                description=f'Outcome confidence extremely low ({confidence:.2f})',
                segment_num=seg_num
            ))

        # Interaction outcome but no reaches
        if outcome in ('displaced_sa', 'displaced_outside', 'retrieved') and n_reaches == 0:
            flags.append(TriageFlag(
                category='cross_step',
                severity='warning',
                description=f'Outcome={outcome} but 0 reaches detected',
                segment_num=seg_num
            ))

        # Untouched but excessive reaches
        if outcome == 'untouched' and n_reaches > 15:
            flags.append(TriageFlag(
                category='cross_step',
                severity='warning',
                description=f'Outcome=untouched but {n_reaches} reaches detected',
                segment_num=seg_num
            ))

    return flags


# ---------------------------------------------------------------------------
# Layer 4: Statistical Outlier Detection
# ---------------------------------------------------------------------------

def check_statistical_outliers(
    video_id: str,
    processing_dir: Path,
) -> List[TriageFlag]:
    """
    Flag segments whose features are extreme outliers compared to
    their peers within the same video.

    Uses IQR-based outlier detection — the data defines what's normal.
    Only flags the absolute tail of the distribution.
    """
    flags = []

    reach_path = processing_dir / f"{video_id}_reaches.json"
    outcome_path = processing_dir / f"{video_id}_pellet_outcomes.json"

    # --- Reach-level outliers ---
    if reach_path.exists():
        try:
            with open(reach_path) as f:
                reach_data = json.load(f)

            # Collect per-segment features
            seg_features = []
            for seg in reach_data.get('segments', []):
                reaches = seg.get('reaches', [])
                n_reaches = seg.get('n_reaches', len(reaches))

                if reaches:
                    extents = [abs(r.get('max_extent_ruler', 0) or 0) for r in reaches]
                    confs = [r.get('confidence', 0) or 0 for r in reaches]
                    durations = [r.get('duration_frames', 0) or 0 for r in reaches]
                else:
                    extents = [0]
                    confs = [0]
                    durations = [0]

                seg_features.append({
                    'segment_num': seg['segment_num'],
                    'n_reaches': n_reaches,
                    'mean_extent': np.mean(extents),
                    'mean_conf': np.mean(confs),
                    'mean_duration': np.mean(durations),
                })

            if len(seg_features) >= 5:
                # Check each feature for outliers
                for feature_name in ['n_reaches', 'mean_extent', 'mean_duration']:
                    values = np.array([sf[feature_name] for sf in seg_features])
                    outlier_indices = _find_outliers_iqr(values, OUTLIER_IQR_MULTIPLIER)

                    for idx in outlier_indices:
                        sf = seg_features[idx]
                        flags.append(TriageFlag(
                            category='statistical',
                            severity='warning',
                            description=(
                                f'Reach {feature_name}={sf[feature_name]:.2f} '
                                f'is an extreme outlier (median={np.median(values):.2f})'
                            ),
                            segment_num=sf['segment_num']
                        ))

        except Exception:
            pass

    # --- Outcome-level outliers ---
    if outcome_path.exists():
        try:
            with open(outcome_path) as f:
                outcome_data = json.load(f)

            segments = outcome_data.get('segments', [])

            # Check distance_from_pillar_start — pellet starting off pillar is unusual
            start_dists = []
            for seg in segments:
                d = seg.get('distance_from_pillar_start')
                if d is not None:
                    start_dists.append((seg.get('segment_num'), d))

            if len(start_dists) >= 5:
                values = np.array([d for _, d in start_dists])
                outlier_indices = _find_outliers_iqr(values, OUTLIER_IQR_MULTIPLIER)

                for idx in outlier_indices:
                    seg_num, dist = start_dists[idx]
                    flags.append(TriageFlag(
                        category='statistical',
                        severity='warning',
                        description=(
                            f'Pellet start distance={dist:.3f} is an extreme outlier '
                            f'(median={np.median(values):.3f}) — possible setup issue'
                        ),
                        segment_num=seg_num
                    ))

        except Exception:
            pass

    return flags


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def triage_video(
    video_id: str,
    processing_dir: Path,
    h5_path: Optional[Path] = None,
    skip_dlc_check: bool = False,
) -> TriageResult:
    """
    Run unified triage on a single video.

    Args:
        video_id: Video identifier (e.g. "20250624_CNT0115_P2")
        processing_dir: Directory containing all output files
        h5_path: Path to DLC h5 file. Auto-detected if None.
                 Set skip_dlc_check=True to skip DLC coherence (faster).
        skip_dlc_check: Skip DLC coherence analysis (avoids loading h5)

    Returns:
        TriageResult with verdict and detailed flags
    """
    processing_dir = Path(processing_dir)
    result = TriageResult(video_id=video_id, verdict='auto_approved')

    # --- Layer 1: DLC Coherence ---
    if not skip_dlc_check:
        if h5_path is None:
            h5_files = list(processing_dir.glob(f"{video_id}*DLC*.h5"))
            if h5_files:
                h5_path = h5_files[0]

        if h5_path and h5_path.exists():
            # Need segment boundaries for mapping anomalies to segments
            seg_boundaries = _load_boundaries(processing_dir, video_id)
            dlc_flags = check_dlc_coherence(h5_path, seg_boundaries)
            result.flags.extend(dlc_flags)
            result.checks_run.append('dlc_coherence')

    # --- Layer 2: Structural Integrity ---
    structural_flags = check_structural(video_id, processing_dir)
    result.flags.extend(structural_flags)
    result.checks_run.append('structural')

    # --- Layer 3: Cross-Step Consistency ---
    cross_flags = check_cross_step(video_id, processing_dir)
    result.flags.extend(cross_flags)
    result.checks_run.append('cross_step')

    # --- Layer 4: Statistical Outliers ---
    stat_flags = check_statistical_outliers(video_id, processing_dir)
    result.flags.extend(stat_flags)
    result.checks_run.append('statistical')

    # --- Verdict ---
    # Any critical flag → needs_review
    # Warnings alone don't trigger review (they're informational)
    if result.n_critical > 0:
        result.verdict = 'needs_review'

    return result


def triage_batch(
    processing_dir: Path,
    skip_dlc_check: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run unified triage on all videos in a directory.

    Returns summary dict with per-video results.
    """
    processing_dir = Path(processing_dir)

    # Find all videos by looking for outcome files (last pipeline step)
    outcome_files = sorted(processing_dir.glob("*_pellet_outcomes.json"))
    if not outcome_files:
        # Fall back to reach files
        outcome_files = sorted(processing_dir.glob("*_reaches.json"))
    if not outcome_files:
        outcome_files = sorted(processing_dir.glob("*_segments.json"))

    # Extract video IDs
    video_ids = set()
    for f in outcome_files:
        stem = f.stem
        for suffix in ['_pellet_outcomes', '_reaches', '_segments', '_segments_v2', '_seg_validation']:
            if stem.endswith(suffix):
                video_ids.add(stem[:-len(suffix)])
                break

    video_ids = sorted(video_ids)

    if verbose:
        print(f"Triaging {len(video_ids)} video(s)...")
        print("-" * 70)

    results = []
    counts = {'auto_approved': 0, 'needs_review': 0}

    for vid in video_ids:
        result = triage_video(vid, processing_dir, skip_dlc_check=skip_dlc_check)
        results.append(result)
        counts[result.verdict] += 1

        if verbose:
            status = "OK" if result.verdict == 'auto_approved' else "REVIEW"
            flag_str = ""
            if result.flags:
                crits = [f for f in result.flags if f.severity == 'critical']
                warns = [f for f in result.flags if f.severity == 'warning']
                parts = []
                if crits:
                    parts.append(f"{len(crits)} critical")
                if warns:
                    parts.append(f"{len(warns)} warning")
                flag_str = f" ({', '.join(parts)})"
            print(f"  [{status:6s}] {vid}{flag_str}")

            # Show critical flags
            for f in result.flags:
                if f.severity == 'critical':
                    seg_str = f" [seg {f.segment_num}]" if f.segment_num else ""
                    print(f"           ! {f.description}{seg_str}")

    if verbose:
        print("-" * 70)
        print(f"Auto-approved: {counts['auto_approved']}, Needs review: {counts['needs_review']}")

    return {
        'total': len(video_ids),
        'auto_approved': counts['auto_approved'],
        'needs_review': counts['needs_review'],
        'results': [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_outliers_iqr(values: np.ndarray, multiplier: float) -> List[int]:
    """Find indices of extreme outliers using IQR method."""
    if len(values) < 5:
        return []
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        # All values are the same — nothing is an outlier
        return []
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return [i for i, v in enumerate(values) if v < lower or v > upper]


def _group_contiguous_frames(frames: List[int], gap: int = 10) -> List[List[int]]:
    """Group frame numbers into contiguous events (within gap tolerance)."""
    if not frames:
        return []
    groups = [[frames[0]]]
    for f in frames[1:]:
        if f - groups[-1][-1] <= gap:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


def _frame_to_segment(frame: int, boundaries: List[int]) -> Optional[int]:
    """Map a frame number to a segment number using boundaries."""
    if not boundaries:
        return None
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= frame < boundaries[i + 1]:
            return i + 1
    return None


def _load_boundaries(processing_dir: Path, video_id: str) -> List[int]:
    """Load segment boundaries from the segments file."""
    for pattern in [
        f"{video_id}_segments.json",
        f"{video_id}_seg_validation.json",
        f"{video_id}_segments_v2.json",
    ]:
        path = processing_dir / pattern
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return data.get('boundaries', [])
            except Exception:
                pass
    return []
