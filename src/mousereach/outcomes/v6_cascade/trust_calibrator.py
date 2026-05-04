"""
Per-stage trust calibrator.

For a given stage, run it on every segment in the GT corpus, compare
its commits to GT, and compute the empirical "trust" score:

    trust = fraction of stage's commits where:
        - committed_class == gt_outcome AND
        - |committed_outcome_known_frame - gt_outcome_known_frame| <= TOL_OKF AND
        - (if class is touched) |committed_interaction_frame - gt_interaction_frame| <= TOL_IFR

Trust is computed PER COMMITTED CLASS (a stage that commits multiple
classes gets a trust per class). Stages that defer everything have no
trust score for that class.

The trust score is what determines triage at runtime: if a stage's
trust on the class it would commit is below the user-set threshold
(e.g., 0.95), the case falls through instead of being committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .stage_base import SegmentInput, Stage, StageDecision


# Default tolerances per the user's 2026-05-01 specification:
# Stage 1 commits at +/-3 frames; we adopt that as the default for all
# stages until per-stage tolerances are explicitly set.
DEFAULT_OKF_TOLERANCE = 3
DEFAULT_IFR_TOLERANCE = 3


@dataclass
class CalibrationCase:
    """One stage decision evaluated against GT."""
    video_id: str
    segment_num: int
    decision: str
    committed_class: Optional[str]
    gt_class: Optional[str]
    committed_outcome_known_frame: Optional[int]
    gt_outcome_known_frame: Optional[int]
    walked_back_gt_okf: Optional[int]    # GT OKF clipped to clean zone
    committed_interaction_frame: Optional[int]
    gt_interaction_frame: Optional[int]
    okf_delta: Optional[int]              # cascade - walked_back_gt
    ifr_delta: Optional[int]
    class_match: bool
    okf_within_tol: bool
    ifr_within_tol: bool
    trust_pass: bool
    reason: str


@dataclass
class StageCalibration:
    stage_name: str
    okf_tolerance: int
    ifr_tolerance: int
    cases: List[CalibrationCase]
    trust_per_class: Dict[str, float] = field(default_factory=dict)
    n_committed_per_class: Dict[str, int] = field(default_factory=dict)
    n_correct_per_class: Dict[str, int] = field(default_factory=dict)
    yield_per_class: Dict[str, float] = field(default_factory=dict)
    n_continued: int = 0
    residual_gt_distribution: Dict[str, int] = field(default_factory=dict)
    n_total: int = 0


def calibrate_stage(
    stage: Stage,
    seg_inputs: List[SegmentInput],
    gt_lookup: Dict,
    okf_tolerance: int = DEFAULT_OKF_TOLERANCE,
    ifr_tolerance: int = DEFAULT_IFR_TOLERANCE,
    transition_zone_half: int = 5,
) -> StageCalibration:
    """Run `stage` on every SegmentInput, compare to GT, compute trust.

    Trust framework (`cascade_trust_framework.md`):
    - Class match against GT outcome class is always required.
    - For UNTOUCHED commits: GT OKF lives in the new-segment /
      transition-zone semantic space; we walk it back into the
      current segment's clean zone via
      `walked_back_gt_okf = min(gt_okf, seg_end - transition_zone_half)`.
      Per the user's logic: if outcome was knowable at GT_OKF, it was
      knowable at every earlier frame in the current segment too. The
      walked-back value sits in clean territory and CAN be compared
      frame-for-frame to the cascade OKF.
    - For TOUCHED commits: GT OKF and GT IFR both live in segment-
      interior clean zone (near the reach), so direct frame-match is
      valid. No walk-back needed.

    `gt_lookup`: dict keyed by (video_id, segment_num) returning a dict
    with keys gt_outcome, gt_outcome_known_frame, gt_interaction_frame.
    """
    cases: List[CalibrationCase] = []
    for seg in seg_inputs:
        decision = stage.decide(seg)
        gt_key = (seg.video_id, seg.segment_num)
        gt = gt_lookup.get(gt_key, {})
        gt_class = gt.get("gt_outcome")
        gt_okf = gt.get("gt_outcome_known_frame")
        gt_ifr = gt.get("gt_interaction_frame")

        committed_class = decision.committed_class
        committed_okf = decision.whens.get("outcome_known_frame") if decision.whens else None
        committed_ifr = decision.whens.get("interaction_frame") if decision.whens else None

        # Walk back GT OKF to the latest clean-zone frame for untouched
        # commits. For touched commits, GT OKF lives in clean territory
        # already (near the reach end), so no walk-back.
        walked_back_gt_okf = None
        if gt_okf is not None:
            if committed_class == "untouched":
                clean_zone_max = seg.seg_end - transition_zone_half
                walked_back_gt_okf = min(int(gt_okf), int(clean_zone_max))
            else:
                walked_back_gt_okf = int(gt_okf)

        okf_delta = None
        ifr_delta = None
        if (committed_okf is not None and walked_back_gt_okf is not None):
            okf_delta = int(committed_okf) - int(walked_back_gt_okf)
        if (committed_ifr is not None and gt_ifr is not None):
            ifr_delta = int(committed_ifr) - int(gt_ifr)

        class_match = (committed_class is not None and committed_class == gt_class)
        okf_within_tol = (okf_delta is not None and abs(okf_delta) <= okf_tolerance)
        ifr_within_tol = True
        if committed_class in ("retrieved", "displaced_sa", "displaced_outside"):
            ifr_within_tol = (ifr_delta is not None and abs(ifr_delta) <= ifr_tolerance)

        if decision.decision == "commit":
            if committed_class == "untouched":
                trust_pass = class_match and okf_within_tol
            else:
                trust_pass = class_match and okf_within_tol and ifr_within_tol
        else:
            trust_pass = False

        cases.append(CalibrationCase(
            video_id=seg.video_id, segment_num=seg.segment_num,
            decision=decision.decision,
            committed_class=committed_class,
            gt_class=gt_class,
            committed_outcome_known_frame=committed_okf,
            gt_outcome_known_frame=gt_okf,
            walked_back_gt_okf=walked_back_gt_okf,
            committed_interaction_frame=committed_ifr,
            gt_interaction_frame=gt_ifr,
            okf_delta=okf_delta, ifr_delta=ifr_delta,
            class_match=class_match,
            okf_within_tol=okf_within_tol,
            ifr_within_tol=ifr_within_tol,
            trust_pass=trust_pass,
            reason=decision.reason,
        ))

    # Compute per-class trust on commits
    n_committed_per_class: Dict[str, int] = {}
    n_correct_per_class: Dict[str, int] = {}
    n_gt_per_class: Dict[str, int] = {}
    n_continued = 0
    residual_gt_distribution: Dict[str, int] = {}

    for c in cases:
        if c.gt_class:
            n_gt_per_class[c.gt_class] = n_gt_per_class.get(c.gt_class, 0) + 1
        if c.decision == "commit":
            cls = c.committed_class
            n_committed_per_class[cls] = n_committed_per_class.get(cls, 0) + 1
            if c.trust_pass:
                n_correct_per_class[cls] = n_correct_per_class.get(cls, 0) + 1
        else:
            n_continued += 1
            if c.gt_class:
                residual_gt_distribution[c.gt_class] = residual_gt_distribution.get(c.gt_class, 0) + 1

    trust_per_class = {}
    yield_per_class = {}
    for cls, n_com in n_committed_per_class.items():
        trust_per_class[cls] = (n_correct_per_class.get(cls, 0) / n_com) if n_com else 0.0
        n_gt_for_cls = n_gt_per_class.get(cls, 0)
        yield_per_class[cls] = (n_correct_per_class.get(cls, 0) / n_gt_for_cls) if n_gt_for_cls else 0.0

    return StageCalibration(
        stage_name=stage.name,
        okf_tolerance=okf_tolerance,
        ifr_tolerance=ifr_tolerance,
        cases=cases,
        trust_per_class=trust_per_class,
        n_committed_per_class=n_committed_per_class,
        n_correct_per_class=n_correct_per_class,
        yield_per_class=yield_per_class,
        n_continued=n_continued,
        residual_gt_distribution=residual_gt_distribution,
        n_total=len(cases),
    )


def report_calibration(cal: StageCalibration) -> str:
    """Render a human-readable summary of stage calibration."""
    lines = []
    lines.append(f"--- {cal.stage_name} (okf_tol=+/-{cal.okf_tolerance}f, ifr_tol=+/-{cal.ifr_tolerance}f) ---")
    lines.append(f"  Total segments: {cal.n_total}")
    n_commit_total = sum(cal.n_committed_per_class.values())
    lines.append(f"  Committed: {n_commit_total}  Continued: {cal.n_continued}")
    lines.append("")
    lines.append("  TRUST per committed class:")
    if not cal.n_committed_per_class:
        lines.append("    (no commits)")
    for cls, n_com in sorted(cal.n_committed_per_class.items()):
        n_correct = cal.n_correct_per_class.get(cls, 0)
        trust = cal.trust_per_class.get(cls, 0.0)
        yield_ = cal.yield_per_class.get(cls, 0.0)
        lines.append(f"    {cls:>22s}: trust = {n_correct}/{n_com} ({100*trust:.1f}%)  "
                     f"yield-on-target = {100*yield_:.1f}%")
    lines.append("")
    lines.append("  RESIDUAL pool GT distribution:")
    for cls, n in sorted(cal.residual_gt_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"    {cls:>22s}: {n}")
    return "\n".join(lines)
