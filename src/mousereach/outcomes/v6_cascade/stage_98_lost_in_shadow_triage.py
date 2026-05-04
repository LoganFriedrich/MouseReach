"""
Stage 98: CV-based "lost in shadow" triage.

Question this stage answers:
    "For a residual segment where DLC lost track of the pellet, did
    the SA contain a dark/shadow region where DLC's visibility floor
    would have prevented continued tracking? If yes -> triage with a
    specific reason (instead of generic catch-all triage in Stage 99)."

Why:
    Per GT-er hand annotations on the v4.0.0 walkthrough abnormal
    cases: "no algorithm in the outcome layer can reasonably classify
    them correctly given the DLC tracking limitations (dirty tray
    artifacts, dark SA / DLC visibility floor, video artifacts)...
    Goal: the algo triage path eventually captures these via a
    dedicated triage rule (cross-segment artifact / dark-SA
    detection)".

    This stage uses computer vision (cv2) to look directly at the
    SA region pixels and detect dark areas that would explain DLC
    failure.

How:
    1. Sample frames in the segment's clean zone (sparse sampling
       for speed -- e.g., 10 evenly-spaced frames).
    2. For each frame, extract the SA region via SA bodypart
       coordinates (SABL/SABR/SATL/SATR define a quadrilateral).
    3. Compute pixel intensity statistics in the SA region:
       - Min intensity (darkest spot)
       - Fraction of pixels below a "DLC visibility floor" threshold
    4. If a significant fraction of SA pixels is below the visibility
       floor AND DLC pellet detection rate in this segment is low,
       triage with reason "lost_in_shadow".

Triage criteria:
    - DLC pellet detection rate (overall fraction of frames at
      Pellet_lk >= 0.7) is BELOW a threshold (DLC struggling)
    - At least 1 sampled frame's SA region has >= SHADOW_AREA_FRAC
      of pixels below SHADOW_INTENSITY_THRESHOLD
    - This stage runs LATE in the cascade (Stage 98) so only fires
      on residuals; commits would have already happened at earlier
      stages

Cascade emit on commit:
    - decision: "triage"
    - committed_class: None
    - reason: "lost_in_shadow_dark_sa_region"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from .stage_base import SegmentInput, Stage, StageDecision


PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
TRANSITION_ZONE_HALF = 5

# Shadow/visibility thresholds (8-bit grayscale, 0=black 255=white).
# Below this intensity, DLC's pellet tracking is unreliable in the
# trained model's experience. 60 = roughly the bottom 25% of typical
# tray brightness.
SHADOW_INTENSITY_THRESHOLD = 60
SHADOW_AREA_FRAC_THRESHOLD = 0.30  # >= 30% of SA region in shadow

# DLC failure threshold: very low (10%) means DLC barely tracked the
# pellet at all -- the unrecoverable case. Higher rates (10-30%) might
# be retrieval where some in-mouth tracking happens; we don't triage
# those.
MAX_DLC_PELLET_DETECTION_RATE = 0.10

# Positive-evidence-of-displacement requirement: the pellet must have
# been observed OFF-PILLAR IN SA at some point. Without this, low
# DLC detection rate is EXPECTED (retrieval) and not a shadow-loss
# failure -- pure retrievals should NOT triage here.
MIN_OFF_PILLAR_IN_SA_OBS = 30     # sustained off-pillar in-SA pellet
                                   # observations -- 30 sustained frames
                                   # is the real-displacement signature.
                                   # Brief transient in-SA fires during
                                   # retrieval (paw transit) wouldn't
                                   # reach this threshold.
MIN_SUSTAINED_RUN = 3

# Frame sampling
N_SAMPLE_FRAMES = 10

PELLET_LK_THR = 0.7
PELLET_LK_OFF_PILLAR = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0


class Stage98LostInShadowTriage(Stage):
    name = "stage_98_lost_in_shadow_triage"
    target_class = None  # triage only

    def __init__(
        self,
        video_dir: Optional[Path] = None,
        shadow_intensity_threshold: int = SHADOW_INTENSITY_THRESHOLD,
        shadow_area_frac_threshold: float = SHADOW_AREA_FRAC_THRESHOLD,
        max_dlc_pellet_detection_rate: float = MAX_DLC_PELLET_DETECTION_RATE,
        min_off_pillar_in_sa_obs: int = MIN_OFF_PILLAR_IN_SA_OBS,
        n_sample_frames: int = N_SAMPLE_FRAMES,
        pellet_lk_threshold: float = PELLET_LK_THR,
        transition_zone_half: int = TRANSITION_ZONE_HALF,
    ):
        self.video_dir = Path(video_dir) if video_dir else None
        self.shadow_intensity_threshold = shadow_intensity_threshold
        self.shadow_area_frac_threshold = shadow_area_frac_threshold
        self.max_dlc_pellet_detection_rate = max_dlc_pellet_detection_rate
        self.min_off_pillar_in_sa_obs = min_off_pillar_in_sa_obs
        self.n_sample_frames = n_sample_frames
        self.pellet_lk_threshold = pellet_lk_threshold
        self.transition_zone_half = transition_zone_half

    def _video_path(self, video_id: str) -> Optional[Path]:
        if self.video_dir is None:
            return None
        candidate = self.video_dir / f"{video_id}.mp4"
        if candidate.exists():
            return candidate
        return None

    def decide(self, seg: SegmentInput) -> StageDecision:
        clean_end = seg.seg_end - self.transition_zone_half
        if clean_end <= seg.seg_start:
            return StageDecision(decision="continue",
                                 reason="segment_too_short")

        sub_raw = seg.dlc_df.iloc[seg.seg_start:clean_end + 1]
        n = len(sub_raw)
        if n == 0:
            return StageDecision(decision="continue", reason="empty_segment")

        # DLC pellet detection rate gate.
        pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
        det_rate = float((pellet_lk >= self.pellet_lk_threshold).mean())
        feats = {
            "pellet_detection_rate": det_rate,
            "n_clean_zone_frames": int(n),
        }
        if det_rate >= self.max_dlc_pellet_detection_rate:
            return StageDecision(
                decision="continue",
                reason=(
                    f"dlc_pellet_detection_rate_too_high "
                    f"({det_rate:.2f} >= {self.max_dlc_pellet_detection_rate}; "
                    f"DLC tracked the pellet, this isn't a shadow-loss case)"
                ),
                features=feats,
            )

        # Positive-evidence gate: pellet must have been observed
        # OFF-PILLAR IN SA at some point in the segment. Without this
        # evidence, low DLC detection rate is consistent with retrieval
        # (pellet eaten = no pellet to detect) and shouldn't triage as
        # shadow-loss. Stage 98 fires only when there's affirmative
        # evidence the pellet was in the SA but DLC couldn't sustain
        # tracking.
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
        geom = compute_pillar_geometry_series(sub)
        pillar_cx_arr = geom["pillar_cx"].to_numpy(dtype=float)
        pillar_cy_arr = geom["pillar_cy"].to_numpy(dtype=float)
        pillar_r_arr = geom["pillar_r"].to_numpy(dtype=float)
        slit_y_line_arr = pillar_cy_arr + pillar_r_arr

        pellet_x_arr = sub["Pellet_x"].to_numpy(dtype=float)
        pellet_y_arr = sub["Pellet_y"].to_numpy(dtype=float)
        dist_radii_arr = (np.sqrt((pellet_x_arr - pillar_cx_arr) ** 2
                                  + (pellet_y_arr - pillar_cy_arr) ** 2)
                          / np.maximum(pillar_r_arr, 1e-6))

        paw_past_y_arr = np.zeros(n, dtype=bool)
        for bp in PAW_BODYPARTS:
            py = sub[f"{bp}_y"].to_numpy(dtype=float)
            pl = sub_raw[f"{bp}_likelihood"].to_numpy(dtype=float)
            paw_past_y_arr |= (py <= slit_y_line_arr) & (pl >= PAW_LK_THR)

        # SA bounding box for pellet-in-SA check.
        sabl_x_med = float(np.median(sub["SABL_x"].to_numpy()))
        sabl_y_med = float(np.median(sub["SABL_y"].to_numpy()))
        sabr_x_med = float(np.median(sub["SABR_x"].to_numpy()))
        sabr_y_med = float(np.median(sub["SABR_y"].to_numpy()))
        satl_x_med = float(np.median(sub["SATL_x"].to_numpy()))
        satl_y_med = float(np.median(sub["SATL_y"].to_numpy()))
        satr_x_med = float(np.median(sub["SATR_x"].to_numpy()))
        satr_y_med = float(np.median(sub["SATR_y"].to_numpy()))
        sa_top_y_med = (satl_y_med + satr_y_med) / 2.0
        sa_bot_y_med = (sabl_y_med + sabr_y_med) / 2.0
        sa_left_x_med = min(sabl_x_med, satl_x_med)
        sa_right_x_med = max(sabr_x_med, satr_x_med)

        in_sa_arr = (
            (pellet_y_arr >= sa_top_y_med) & (pellet_y_arr <= sa_bot_y_med)
            & (pellet_x_arr >= sa_left_x_med) & (pellet_x_arr <= sa_right_x_med)
        )
        off_pillar_in_sa_arr = (
            (pellet_lk >= PELLET_LK_OFF_PILLAR)
            & (dist_radii_arr > ON_PILLAR_RADII)
            & in_sa_arr
            & (~paw_past_y_arr)
        )
        # Compute sustained run count (3+ consecutive frames).
        def sustained_count(arr):
            total = 0
            r = 0
            for v in arr:
                if v:
                    r += 1
                else:
                    if r >= MIN_SUSTAINED_RUN:
                        total += r
                    r = 0
            if r >= MIN_SUSTAINED_RUN:
                total += r
            return total
        n_off_pillar_in_sa = sustained_count(off_pillar_in_sa_arr)
        feats["n_off_pillar_in_sa_sustained"] = n_off_pillar_in_sa

        # Late-zone evidence drop kept lenient -- real shadow-loss
        # cases may have pellet completely obscured by shadow late
        # in segment (no late observations at all). The DLC <10% +
        # severe shadow combination is what defines this triage --
        # late-zone evidence isn't required.
        late_start_idx = int(n * 0.75)
        late_off_in_sa = sustained_count(off_pillar_in_sa_arr[late_start_idx:])
        feats["late_off_in_sa_sustained"] = late_off_in_sa

        # Retrieval-plausibility check: defer if any in-mouth track
        # signal (pellet briefly observed ABOVE slit-y-line) is
        # present. In-mouth tracks indicate retrieval, not shadow-
        # loss. Per user 2026-05-03: "for retrievals we would actually
        # expect that the pellet disappear so this triager catching
        # retrieved is bad".
        above_slit_arr = (
            (pellet_lk >= PELLET_LK_OFF_PILLAR)
            & (pellet_y_arr < slit_y_line_arr - 5)
            & (~paw_past_y_arr)
        )
        n_above_slit = sustained_count(above_slit_arr)
        feats["n_above_slit_sustained"] = n_above_slit
        if n_above_slit >= 3:
            return StageDecision(
                decision="continue",
                reason=(
                    f"in_mouth_track_signal_present "
                    f"({n_above_slit} sustained above-slit pellet frames; "
                    f"this looks like retrieval -- shadow-loss triage "
                    f"would be wrong)"
                ),
                features=feats,
            )

        # Even with severe shadow, displaced cases tend to have at
        # LEAST a few brief off-pillar in-SA observations before DLC
        # loses tracking. Pure retrievals have ZERO (pellet was eaten,
        # never in SA). Require >= 3 sustained observations.
        if n_off_pillar_in_sa < 3:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_off_pillar_in_sa_evidence "
                    f"(sustained off-pillar in-SA = {n_off_pillar_in_sa}; "
                    f"pellet was never in SA -- consistent with retrieval, "
                    f"not shadow-loss; defer)"
                ),
                features=feats,
            )

        # Late-zone evidence: for real shadow-loss displacement, the
        # pellet stays in SA throughout (it's just obscured by shadow).
        # For brief retrieval-transit-then-eaten, late-zone has no
        # off-pillar evidence. Require at least 1 sustained late
        # off-pillar in-SA observation.
        late_start_idx = int(n * 0.5)
        late_off_in_sa = sustained_count(off_pillar_in_sa_arr[late_start_idx:])
        feats["late_off_in_sa_sustained"] = late_off_in_sa
        if late_off_in_sa < 3:
            return StageDecision(
                decision="continue",
                reason=(
                    f"no_late_zone_off_pillar_in_sa_evidence "
                    f"(late sustained off-pillar in-SA = {late_off_in_sa}; "
                    f"pellet not seen in SA late -- consistent with "
                    f"retrieval-transit, not real shadow-loss; defer)"
                ),
                features=feats,
            )

        # Need video file to do CV.
        video_path = self._video_path(seg.video_id)
        if video_path is None:
            return StageDecision(
                decision="continue",
                reason=(
                    f"video_unavailable "
                    f"(no .mp4 found for {seg.video_id} in {self.video_dir})"
                ),
                features=feats,
            )

        # Compute SA region from SA bodyparts (use clean-zone median).
        sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
        sabl_x = float(np.median(sub["SABL_x"].to_numpy()))
        sabl_y = float(np.median(sub["SABL_y"].to_numpy()))
        sabr_x = float(np.median(sub["SABR_x"].to_numpy()))
        sabr_y = float(np.median(sub["SABR_y"].to_numpy()))
        satl_x = float(np.median(sub["SATL_x"].to_numpy()))
        satl_y = float(np.median(sub["SATL_y"].to_numpy()))
        satr_x = float(np.median(sub["SATR_x"].to_numpy()))
        satr_y = float(np.median(sub["SATR_y"].to_numpy()))

        # Bounding box of SA quadrilateral.
        sa_left = int(min(sabl_x, satl_x))
        sa_right = int(max(sabr_x, satr_x))
        sa_top = int(min(satl_y, satr_y))
        sa_bot = int(max(sabl_y, sabr_y))
        if sa_right <= sa_left or sa_bot <= sa_top:
            return StageDecision(
                decision="continue",
                reason="sa_bounding_box_invalid",
                features=feats,
            )

        # Sample frames in segment's clean zone.
        sample_frame_idxs = np.linspace(
            seg.seg_start, clean_end, self.n_sample_frames, dtype=int)

        # Use cv2 to load frames and check intensity.
        try:
            import cv2
        except ImportError:
            return StageDecision(
                decision="continue",
                reason="cv2_unavailable",
                features=feats,
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return StageDecision(
                decision="continue",
                reason=f"cv2_failed_to_open_{video_path.name}",
                features=feats,
            )

        max_shadow_frac = 0.0
        sample_results = []
        try:
            for fidx in sample_frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    sample_results.append((int(fidx), None))
                    continue
                # Convert to grayscale.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                # Extract SA bounding box (clip to image bounds).
                x0 = max(0, sa_left)
                x1 = min(w, sa_right)
                y0 = max(0, sa_top)
                y1 = min(h, sa_bot)
                if x1 <= x0 or y1 <= y0:
                    sample_results.append((int(fidx), None))
                    continue
                sa_region = gray[y0:y1, x0:x1]
                # Fraction of pixels below shadow threshold.
                shadow_pixels = (sa_region < self.shadow_intensity_threshold).sum()
                total_pixels = sa_region.size
                shadow_frac = shadow_pixels / max(total_pixels, 1)
                sample_results.append((int(fidx), float(shadow_frac)))
                if shadow_frac > max_shadow_frac:
                    max_shadow_frac = shadow_frac
        finally:
            cap.release()

        feats["max_shadow_frac"] = max_shadow_frac
        feats["sample_results"] = sample_results

        if max_shadow_frac < self.shadow_area_frac_threshold:
            return StageDecision(
                decision="continue",
                reason=(
                    f"sa_shadow_below_threshold "
                    f"(max shadow frac {max_shadow_frac:.2f} < "
                    f"{self.shadow_area_frac_threshold})"
                ),
                features=feats,
            )

        # Triage with specific reason.
        return StageDecision(
            decision="triage",
            committed_class=None,
            reason=(
                f"lost_in_shadow_dark_sa_region "
                f"(DLC pellet detection rate {det_rate:.2f} below floor "
                f"{self.max_dlc_pellet_detection_rate}; SA region has "
                f"{max_shadow_frac:.2%} pixels below intensity "
                f"{self.shadow_intensity_threshold}; pellet likely rolled "
                f"into shadow where DLC tracking floor was exceeded -- "
                f"unrecoverable algorithmically; route to human review)"
            ),
            features=feats,
        )
