"""
Outcome cascade experiment: three stage fixes targeting missed retrievals.

Adds:
  - Stage 16 subclass: global post-causal vanish veto (override commit -> continue
    if pellet sustains >= 60 frames of low lk after the causal reach end)
  - Stage 27 subclass: same global vanish veto
  - New Stage 30: "fallback retrieved via sustained vanish" inserted BEFORE
    Stage 99, catches strong-vanish cases that no other retrieved-commit
    stage fires on

================================================================
Pre-experiment checklist (per memory pre_experiment_checklist.md)
================================================================

1. Cumulative-stacking check (verified 2026-06-02 against current state)

   Reach detector: v8.0.4 production (BSW + leading-trim + apex-split +
   trailing-trim). Same as 2026-06-01 experiment.

   Outcome cascade BASELINE: v6.0.0 unmodified, accepted 2026-06-01
   snapshot Improvement_Snapshots/outcome/v6.0.0_e2e_v804_reaches_2026-06-01/
   on the 20-video 2026-05-11 generalization corpus. Result was 377/400 raw
   correct, 11 triages, retrieved 58/67. Master at c4ca5dd as of this run.

   This experiment LAYERS THREE FIXES on top of that cumulative best:
     - Stage 16 vanish veto
     - Stage 27 vanish veto
     - Stage 30 fallback retrieved

   Comparison baseline: the 2026-06-01 algo_outputs re-scored against the
   current (post-2026-06-01 GT edit on CNT0216_P1 s18) GT. The GT edit
   alone moves 1 commit from wrong to correct, so the corrected baseline
   is approximately 378/400 not 377/400.

   No v6 cascade pending integrations exist (cascade hasn't been edited
   since 2026-05-04 ship). My three new modifications are NOT yet
   integrated into detector.py; they live inline in this runner.

   Verification method: snapshot/algo_outputs comparison, code inspection
   of v6_cascade/detector.py and stage files.

2. Existing-code-modification check: NO. This runner imports existing
   v6_cascade stage classes from src/mousereach/outcomes/v6_cascade/ and
   subclasses them inline. Does not modify any module under src/mousereach/.

3. Unverified hypotheses
   - H1: Stage 27 vanish veto catches CNT0316_P3 s13 (355f vanish) and
     CNT0414_P4 s17 (100f vanish) -- both currently wrongly committed as
     displaced. After veto, expect them to fall through to Stage 28 (or new
     Stage 30) and commit retrieved.
   - H2: Stage 16 vanish veto: no specific case in current error set fires
     this (all 3 Stage 16 errors have vanish < 7f). Defensive measure.
   - H3: Stage 30 catches CNT0214_P1 s15 (vanish=400, no competing
     displacement) and CNT0402_P4 s17 (vanish=132, displacement=25px --
     borderline depending on threshold).
   - H4: No regression on untouched (Stage 30 requires reaches present, so
     0-reach untouched segments cannot trigger it).
   - H5: Stage 30 may CREATE new wrong commits on displaced_outside cases
     that look identical in DLC. Risk: 1-3 events flipped from triage to
     wrong-retrieved. Tradeoff acceptable since current handling is also
     wrong (triage = no commit info downstream).

4. FN-direction-reporting (planned RESULTS.md lead-line structure)
   Lead: "Triage count and total correct vs cumulative-best baseline
   (v6.0.0 unmodified, 2026-06-01 run re-scored against current GT):
   <direction by magnitude>."
   Two deltas: (a) vs cumulative best (the 2026-06-01 run), (b) vs pure
   baseline (the 2026-05-11 v7.2.0+v6 baseline, 240/400).

5. Framework-not-adhoc
   Snapshot dir: Improvement_Snapshots/outcome/v6.0.1_three_stage_fixes_2026-06-02/
   Schema: matches compute_outcome_metrics canonical scorer output.
   Figures: will be generated post-run via canonical
   mousereach.improvement.outcome._run_notebooks.

6. Branch + tag
   - Pre-experiment tag: outcome-pre-three-stage-fixes-2026-06-02
   - Feature branch: feature/outcome-three-stage-fixes
   - Set on master c4ca5dd before this runner executes (handled by user
     or invoking shell before calling this script).

7. Decision rule
   - ACCEPT: total correct rises by >= 3 from corrected-baseline AND
     retrieved class correct rises by >= 2 AND no class regresses by > 2.
   - REJECT: total correct does not rise, OR retrieved class does not
     improve, OR any class regresses by > 2.
   - NEUTRAL: changes within +/-2 on every class, no clear win or loss.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mousereach.reach.v8 import detect_reaches_v8
from mousereach.reach.v8.features import load_dlc_h5
from mousereach.outcomes.v6_cascade.detector import _build_production_stages
from mousereach.outcomes.v6_cascade.stage_base import (
    SegmentInput, Stage, StageDecision)
from mousereach.outcomes.v6_cascade.stage_16_displaced_via_max_displacement_reach import (
    Stage16DisplacedViaMaxDisplacement)
from mousereach.outcomes.v6_cascade.stage_27_displaced_sa_via_unique_high_displacement_reach import (
    Stage27DisplacedSaViaUniqueHighDisplacement)
from mousereach.improvement.outcome.metrics import compute_outcome_metrics


# ---------------------------------------------------------------------------
# Helper: max consecutive frames where pellet_lk < threshold
# ---------------------------------------------------------------------------

def _max_consecutive_low_lk(dlc_df, start: int, end: int, thr: float = 0.5) -> int:
    if end <= start:
        return 0
    lk = dlc_df["Pellet_likelihood"].iloc[start:end].to_numpy(dtype=float)
    below = lk < thr
    max_run = 0
    cur = 0
    for b in below:
        if b:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


# ---------------------------------------------------------------------------
# Fix #1 + #2: Stage 16 and Stage 27 subclasses with global post-causal
# vanish veto. The subclass calls super().decide() and if the parent
# would commit, runs a post-check: if pellet sustains >= 60 frames of
# low lk anywhere after the causal reach end, refuse to commit and
# return continue instead (letting later retrieved stages handle).
# ---------------------------------------------------------------------------

VANISH_VETO_FRAMES = 60


class Stage16WithVanishVeto(Stage16DisplacedViaMaxDisplacement):
    name = "stage_16_with_vanish_veto"

    def decide(self, seg: SegmentInput) -> StageDecision:
        d = super().decide(seg)
        if d.decision != "commit":
            return d
        causal_idx = d.features.get("causal_idx")
        if causal_idx is None:
            return d
        reaches = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if causal_idx >= len(reaches):
            return d
        _, causal_re = reaches[causal_idx]
        max_run = _max_consecutive_low_lk(seg.dlc_df,
                                            causal_re + 1,
                                            seg.seg_end + 1,
                                            thr=0.5)
        feats = dict(d.features)
        feats["post_causal_max_vanish_run"] = int(max_run)
        if max_run >= VANISH_VETO_FRAMES:
            feats["vanish_veto_fired"] = True
            return StageDecision(
                decision="continue",
                reason=(f"global_post_causal_vanish_veto "
                        f"({max_run}f >= {VANISH_VETO_FRAMES}f sustained vanish)"),
                features=feats,
            )
        return StageDecision(
            decision=d.decision,
            committed_class=d.committed_class,
            whens=d.whens,
            reason=d.reason,
            features=feats,
        )


class Stage27WithVanishVeto(Stage27DisplacedSaViaUniqueHighDisplacement):
    name = "stage_27_with_vanish_veto"

    def decide(self, seg: SegmentInput) -> StageDecision:
        d = super().decide(seg)
        if d.decision != "commit":
            return d
        causal_idx = d.features.get("causal_reach_idx")
        if causal_idx is None:
            return d
        reaches = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows])
        if causal_idx >= len(reaches):
            return d
        _, causal_re = reaches[causal_idx]
        max_run = _max_consecutive_low_lk(seg.dlc_df,
                                            causal_re + 1,
                                            seg.seg_end + 1,
                                            thr=0.5)
        feats = dict(d.features)
        feats["post_causal_max_vanish_run"] = int(max_run)
        if max_run >= VANISH_VETO_FRAMES:
            feats["vanish_veto_fired"] = True
            return StageDecision(
                decision="continue",
                reason=(f"global_post_causal_vanish_veto "
                        f"({max_run}f >= {VANISH_VETO_FRAMES}f sustained vanish)"),
                features=feats,
            )
        return StageDecision(
            decision=d.decision,
            committed_class=d.committed_class,
            whens=d.whens,
            reason=d.reason,
            features=feats,
        )


# ---------------------------------------------------------------------------
# Fix #3: Stage 30 fallback retrieved via sustained vanish. Inserted BEFORE
# Stage 99 (residual triage). Commits retrieved if:
#   - sustained pellet vanish (>= 100 frames) somewhere after the last reach
#   - pillar lk pre-reach low, post-reach high (clear transition)
#   - no competing high-displacement reach (max disp < 12px among all reaches)
# ---------------------------------------------------------------------------

STAGE30_MIN_VANISH = 100
STAGE30_PILLAR_PRE_MAX = 0.5
STAGE30_PILLAR_POST_MIN = 0.85
STAGE30_MAX_COMPETING_DISP_PX = 12.0


class Stage30FallbackRetrievedViaSustainedVanish(Stage):
    name = "stage_30_retrieved_via_sustained_vanish_fallback"
    target_class = "retrieved"

    def decide(self, seg: SegmentInput) -> StageDecision:
        reaches = sorted([(int(rs), int(re)) for (rs, re) in seg.reach_windows
                           if rs is not None and re is not None])
        feats = {"n_reaches": len(reaches)}
        if not reaches:
            return StageDecision(decision="continue",
                                  reason="no_reaches",
                                  features=feats)

        last_rs, last_re = reaches[-1]
        scan_lo = last_re + 1
        scan_hi = seg.seg_end + 1
        if scan_hi - scan_lo < STAGE30_MIN_VANISH:
            return StageDecision(
                decision="continue",
                reason=(f"post_last_reach_window_too_short "
                        f"({scan_hi - scan_lo} < {STAGE30_MIN_VANISH})"),
                features=feats,
            )

        max_run = _max_consecutive_low_lk(seg.dlc_df, scan_lo, scan_hi, thr=0.5)
        feats["max_vanish_run"] = int(max_run)
        if max_run < STAGE30_MIN_VANISH:
            return StageDecision(
                decision="continue",
                reason=(f"insufficient_sustained_vanish "
                        f"({max_run} < {STAGE30_MIN_VANISH})"),
                features=feats,
            )

        # Pillar lk transition check
        first_rs = reaches[0][0]
        if first_rs > seg.seg_start:
            pre_pillar = seg.dlc_df["Pillar_likelihood"].iloc[
                seg.seg_start:first_rs].to_numpy(dtype=float)
        else:
            pre_pillar = np.array([], dtype=float)
        post_pillar = seg.dlc_df["Pillar_likelihood"].iloc[
            last_re + 1:seg.seg_end + 1].to_numpy(dtype=float)
        if len(pre_pillar) == 0 or len(post_pillar) == 0:
            return StageDecision(
                decision="continue",
                reason="empty_pillar_lk_windows",
                features=feats,
            )
        pre_mean = float(np.mean(pre_pillar))
        post_mean = float(np.mean(post_pillar))
        feats["pillar_lk_pre_mean"] = round(pre_mean, 3)
        feats["pillar_lk_post_mean"] = round(post_mean, 3)
        if pre_mean >= STAGE30_PILLAR_PRE_MAX:
            return StageDecision(
                decision="continue",
                reason=(f"pillar_pre_lk_too_high (pre={pre_mean:.2f} >= "
                        f"{STAGE30_PILLAR_PRE_MAX}); pellet probably not "
                        f"on pillar to begin with"),
                features=feats,
            )
        if post_mean <= STAGE30_PILLAR_POST_MIN:
            return StageDecision(
                decision="continue",
                reason=(f"pillar_post_lk_too_low (post={post_mean:.2f} <= "
                        f"{STAGE30_PILLAR_POST_MIN}); pillar not clearly "
                        f"revealed after vanish"),
                features=feats,
            )

        # Competing high-displacement reach check
        max_disp = 0.0
        for rs, re in reaches:
            pre_window_start = max(seg.seg_start, rs - 30)
            post_window_end = min(seg.seg_end + 1, re + 1 + 30)
            pre_lk = seg.dlc_df["Pellet_likelihood"].iloc[
                pre_window_start:rs].to_numpy(dtype=float)
            post_lk = seg.dlc_df["Pellet_likelihood"].iloc[
                re + 1:post_window_end].to_numpy(dtype=float)
            pre_c = pre_lk >= 0.7
            post_c = post_lk >= 0.7
            if not pre_c.any() or not post_c.any():
                continue
            pre_x = seg.dlc_df["Pellet_x"].iloc[
                pre_window_start:rs].to_numpy(dtype=float)
            pre_y = seg.dlc_df["Pellet_y"].iloc[
                pre_window_start:rs].to_numpy(dtype=float)
            post_x = seg.dlc_df["Pellet_x"].iloc[
                re + 1:post_window_end].to_numpy(dtype=float)
            post_y = seg.dlc_df["Pellet_y"].iloc[
                re + 1:post_window_end].to_numpy(dtype=float)
            pmx = float(np.median(pre_x[pre_c]))
            pmy = float(np.median(pre_y[pre_c]))
            qmx = float(np.median(post_x[post_c]))
            qmy = float(np.median(post_y[post_c]))
            d = float(np.sqrt((qmx - pmx) ** 2 + (qmy - pmy) ** 2))
            if d > max_disp:
                max_disp = d
        feats["max_competing_disp_px"] = round(max_disp, 2)
        if max_disp >= STAGE30_MAX_COMPETING_DISP_PX:
            return StageDecision(
                decision="continue",
                reason=(f"competing_displacement_signal "
                        f"({max_disp:.1f}px >= "
                        f"{STAGE30_MAX_COMPETING_DISP_PX}); ambiguous "
                        f"between retrieved and displaced"),
                features=feats,
            )

        # Commit retrieved. Use last reach as causal.
        bout_length = last_re - last_rs + 1
        interaction_frame = last_rs + bout_length // 2
        okf = last_re + 5
        return StageDecision(
            decision="commit",
            committed_class="retrieved",
            whens={
                "interaction_frame": int(interaction_frame),
                "outcome_known_frame": int(okf),
            },
            reason=(f"sustained_vanish_fallback "
                    f"(vanish={max_run}f >= {STAGE30_MIN_VANISH}f, "
                    f"pillar {pre_mean:.2f}->{post_mean:.2f}, "
                    f"max_disp={max_disp:.1f}px)"),
            features=feats,
        )


# ---------------------------------------------------------------------------
# Cascade with modifications
# ---------------------------------------------------------------------------

def build_modified_stages(video_dir=None):
    """Build the v6 cascade stage list with three modifications:
      - Stage 16 -> Stage 16 with vanish veto
      - Stage 27 -> Stage 27 with vanish veto
      - Insert Stage 30 before Stage 99
    """
    base = _build_production_stages(video_dir=video_dir)
    modified = []
    for label, stage in base:
        if isinstance(stage, Stage16DisplacedViaMaxDisplacement):
            modified.append((label.replace("stage_16_displaced_via_max_displacement",
                                            "stage_16_with_vanish_veto"),
                              Stage16WithVanishVeto()))
        elif isinstance(stage, Stage27DisplacedSaViaUniqueHighDisplacement):
            modified.append((label.replace("stage_27_displaced_sa_via_unique_high_displacement",
                                            "stage_27_with_vanish_veto"),
                              Stage27WithVanishVeto()))
        elif "stage_99_residual_triage" in label:
            modified.append(("stage_30_retrieved_via_sustained_vanish_fallback",
                              Stage30FallbackRetrievedViaSustainedVanish()))
            modified.append((label, stage))
        else:
            modified.append((label, stage))
    return modified


def run_cascade_on_segments(seg_inputs, stages):
    """Replicate detector.py's cascade loop with a custom stage list."""
    output_segments_per_video = {}
    for seg_input in seg_inputs:
        decision = None
        committing_stage = "residual (auto-triage)"
        for label, stage in stages:
            dec = stage.decide(seg_input)
            if dec.decision in ("commit", "triage"):
                decision = dec
                committing_stage = label
                break
        if decision is not None and decision.decision == "commit":
            rec = {
                "segment_num": seg_input.segment_num,
                "outcome": decision.committed_class,
                "outcome_known_frame": decision.whens.get("outcome_known_frame"),
                "interaction_frame": decision.whens.get("interaction_frame"),
                "stage": committing_stage,
                "flagged_for_review": False,
            }
        else:
            reason = decision.reason if decision is not None else "fell_through_all_stages"
            rec = {
                "segment_num": seg_input.segment_num,
                "outcome": "triaged",
                "outcome_known_frame": None,
                "interaction_frame": None,
                "stage": committing_stage,
                "flagged_for_review": True,
                "flag_reason": reason,
            }
        output_segments_per_video.setdefault(seg_input.video_id, []).append(rec)
    return output_segments_per_video


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11"
)
DLC_DIR = CORPUS_DIR / "algo_outputs_current"
GT_DIR = CORPUS_DIR / "gt"

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\outcome\v6.0.1_three_stage_fixes_2026-06-02"
)


def find_dlc(video_id: str) -> Path:
    matches = sorted(DLC_DIR.glob(f"{video_id}DLC_*.h5"))
    if not matches:
        raise FileNotFoundError(f"No DLC h5 for {video_id} in {DLC_DIR}")
    return matches[0]


def load_gt_segments(video_id: str) -> List[Tuple[int, int]]:
    gt = json.loads(
        (GT_DIR / f"{video_id}_unified_ground_truth.json").read_text(encoding="utf-8")
    )
    boundary_frames = [int(b["frame"]) for b in
                        gt.get("segmentation", {}).get("boundaries", [])]
    if len(boundary_frames) < 2:
        return []
    segs = []
    for i in range(len(boundary_frames) - 1):
        segs.append((boundary_frames[i], boundary_frames[i + 1] - 1))
    return segs


def save_reaches_segmented(video_id, reaches, segments, out_path):
    video_segments_data = []
    global_reach_id = 0
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        seg_num = seg_idx + 1
        seg_reaches = []
        reach_num = 0
        for r_start, r_end in reaches:
            if seg_start <= r_start <= seg_end:
                reach_num += 1
                global_reach_id += 1
                seg_reaches.append({
                    "reach_id": global_reach_id,
                    "reach_num": reach_num,
                    "start_frame": int(r_start),
                    "end_frame": int(r_end),
                    "duration_frames": int(r_end - r_start + 1),
                })
        video_segments_data.append({
            "segment_num": seg_num,
            "start_frame": int(seg_start),
            "end_frame": int(seg_end),
            "n_reaches": len(seg_reaches),
            "reaches": seg_reaches,
        })
    data = {
        "video_id": video_id,
        "detector": "v8.0.4_production",
        "n_segments": len(segments),
        "segments": video_segments_data,
    }
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    t0 = time.time()
    print(f"Three-stage-fix experiment (Stage 16 veto + Stage 27 veto + Stage 30)")
    print(f"Snapshot dir: {SNAPSHOT_DIR}")

    algo_outputs_dir = SNAPSHOT_DIR / "algo_outputs"
    metrics_dir = SNAPSHOT_DIR / "metrics"
    algo_outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    video_ids = sorted({p.stem.replace("_unified_ground_truth", "")
                         for p in GT_DIR.glob("*_unified_ground_truth.json")})
    print(f"Found {len(video_ids)} videos")

    # Build modified stage list ONCE (stateless stages)
    stages = build_modified_stages(video_dir=None)

    for i, vid in enumerate(video_ids, 1):
        t_vid = time.time()
        print(f"[{i}/{len(video_ids)}] {vid}", flush=True)

        dlc_path = find_dlc(vid)
        dlc_df = load_dlc_h5(dlc_path)
        segments = load_gt_segments(vid)

        algo_reaches = detect_reaches_v8(dlc_df)
        print(f"   detected {len(algo_reaches)} reaches", flush=True)

        save_reaches_segmented(vid, algo_reaches, segments,
                                algo_outputs_dir / f"{vid}_reaches.json")

        # Build SegmentInput list
        seg_inputs = []
        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_num = seg_idx + 1
            seg_reaches = [(rs, re) for rs, re in algo_reaches
                            if seg_start <= rs <= seg_end]
            seg_inputs.append(SegmentInput(
                video_id=vid, segment_num=seg_num,
                seg_start=seg_start, seg_end=seg_end,
                dlc_df=dlc_df, reach_windows=seg_reaches,
            ))

        outs = run_cascade_on_segments(seg_inputs, stages)
        if vid in outs:
            data = {
                "video_id": vid,
                "detector": "v6_cascade_three_stage_fixes",
                "detector_version": "6.0.1_three_stage_fixes",
                "reach_detector_version": "v8.0.4_production",
                "segments": outs[vid],
            }
            (algo_outputs_dir / f"{vid}_pellet_outcomes.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        print(f"   ({time.time() - t_vid:.1f}s)", flush=True)

    print(f"\nScoring with compute_outcome_metrics...", flush=True)
    scalars = compute_outcome_metrics(
        gt_dir=GT_DIR,
        algo_dir=algo_outputs_dir,
        output_dir=metrics_dir,
        video_ids=video_ids,
        reaches_dir=algo_outputs_dir,
    )

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Snapshot complete at: {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
