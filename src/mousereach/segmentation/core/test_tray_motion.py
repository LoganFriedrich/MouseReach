"""
Unit tests for the tray-motion gate (Signal 27).

Tests use synthetic DLC dataframes that simulate the four documented
patterns:
  - Real ASPA cycle: SA corner sweeps >= 30 px AND pillar lk drops
  - Operator adjustment: small SA corner shift, pillar lk stays high
  - Pellet-only motion: SA corners static, pillar lk drops (rare)
  - Total noise: neither signal present
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mousereach.segmentation.core.tray_motion import (
    DEFAULT_EXCURSION_THRESHOLD,
    DEFAULT_PILLAR_LK_DROP_THRESHOLD,
    apply_tray_motion_gate,
    replace_invalid_boundaries,
    validate_tray_motion,
)


def _make_df(n_frames: int = 1000) -> pd.DataFrame:
    """Build an empty DLC-shaped dataframe."""
    cols = {}
    for bp in ("SABL", "SABR", "SATL", "SATR"):
        cols[f"{bp}_x"] = np.full(n_frames, 100.0)
        cols[f"{bp}_y"] = np.full(n_frames, 200.0)
        cols[f"{bp}_likelihood"] = np.full(n_frames, 1.0)
    cols["Pillar_x"] = np.full(n_frames, 165.0)
    cols["Pillar_y"] = np.full(n_frames, 460.0)
    cols["Pillar_likelihood"] = np.full(n_frames, 1.0)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# validate_tray_motion -- single-boundary checks
# ---------------------------------------------------------------------------

class TestValidateTrayMotion:

    def test_real_aspa_cycle_passes(self):
        """SA excursion >= 30 px AND pillar lk drop >= 0.3 -> valid.

        Pre/post pillar lk windows are OFFSET from the boundary
        (default 2x window each side: pre=[F-150,F-50], post=[F+50,F+150])
        because the boundary frame F is mid-cycle and pillar lk doesn't
        drop until the new pellet settles.
        """
        df = _make_df(1000)
        # Inject a 40 px SABL excursion across the motion window
        # [F-50, F+50] = [450, 550] for boundary at 500.
        df.loc[470:480, "SABL_x"] = 80.0   # left swing
        df.loc[490:510, "SABL_x"] = 130.0  # right swing
        # Pillar lk pre window [F-150, F-50] = [350, 450]: high
        # Pillar lk post window [F+50, F+150] = [550, 650]: low
        df.loc[350:450, "Pillar_likelihood"] = 1.0
        df.loc[550:650, "Pillar_likelihood"] = 0.05
        is_valid, reasons = validate_tray_motion(df, 500)
        assert is_valid, f"unexpected reasons: {reasons}"
        assert reasons == []

    def test_operator_adjustment_fails(self):
        """Small SA shift fails on excursion alone.

        v2.2.1: only the excursion test gates validity (pillar lk drop
        test is hard-disabled after corpus over-rejection findings).
        Operator adjustments produce only 5-15 px tray shifts, well
        below the 30 px threshold.
        """
        df = _make_df(1000)
        df.loc[490:510, "SABL_x"] = 105.0  # 5 px shift, < threshold
        is_valid, reasons = validate_tray_motion(df, 500)
        assert not is_valid
        assert any("insufficient_tray_excursion" in r for r in reasons)

    def test_excursion_present_no_pillar_drop_passes(self):
        """SA excursion present and pillar lk static -> valid.

        v2.2.1: pillar lk test is disabled. Excursion alone determines
        validity. This test was previously named test_excursion_only_fails
        and inverted -- pillar-only failures no longer invalidate.
        """
        df = _make_df(1000)
        df.loc[470:480, "SABL_x"] = 60.0
        df.loc[490:510, "SABL_x"] = 140.0
        # Pillar lk stays at 1.0 -- no drop, but test is disabled
        is_valid, reasons = validate_tray_motion(df, 500)
        assert is_valid, f"unexpected reasons: {reasons}"

    def test_pillar_drop_only_fails(self):
        """Pillar lk drops but SA stays static -> invalid (excursion fails).

        With the pillar test disabled, this case still fails because the
        excursion test detects no SA motion. The pillar-only signal does
        not rescue an SA-static boundary.
        """
        df = _make_df(1000)
        # SA static (no excursion)
        df.loc[350:450, "Pillar_likelihood"] = 1.0
        df.loc[550:650, "Pillar_likelihood"] = 0.05
        is_valid, reasons = validate_tray_motion(df, 500)
        assert not is_valid
        assert any("insufficient_tray_excursion" in r for r in reasons)

    def test_excursion_uses_max_across_corners(self):
        """If SABR has the excursion, gate should still pass on excursion."""
        df = _make_df(1000)
        # Only SABR moves
        df.loc[470:480, "SABR_x"] = 60.0
        df.loc[490:510, "SABR_x"] = 140.0
        # Pillar lk in offset windows
        df.loc[350:450, "Pillar_likelihood"] = 1.0
        df.loc[550:650, "Pillar_likelihood"] = 0.05
        is_valid, reasons = validate_tray_motion(df, 500)
        assert is_valid, f"unexpected reasons: {reasons}"

    def test_boundary_at_video_edge(self):
        """Boundary near frame 0 or end: truncated window still works.

        With boundary at F=10 the pre window is empty (F-150..F-50 < 0)
        so the pillar test has no data and silently defaults to pass.
        Excursion test uses motion window [F-50, F+50] which clips to
        [0, 60] and still sees the synthetic excursion.
        """
        df = _make_df(1000)
        df.loc[0:30, "SABL_x"] = 60.0
        df.loc[31:60, "SABL_x"] = 140.0
        is_valid, reasons = validate_tray_motion(df, 10)
        assert is_valid, f"unexpected reasons: {reasons}"

    def test_boundary_out_of_range(self):
        df = _make_df(100)
        is_valid, reasons = validate_tray_motion(df, 200)
        assert not is_valid
        assert any("boundary_out_of_range" in r for r in reasons)

    def test_missing_pillar_column_passes_test_2(self):
        """If Pillar_likelihood column absent, test 2 defaults to pass."""
        df = _make_df(1000)
        df = df.drop(columns=["Pillar_likelihood"])
        df.loc[470:480, "SABL_x"] = 60.0
        df.loc[490:510, "SABL_x"] = 140.0
        is_valid, reasons = validate_tray_motion(df, 500)
        # Excursion test passes, pillar test silently skipped
        assert is_valid


# ---------------------------------------------------------------------------
# replace_invalid_boundaries
# ---------------------------------------------------------------------------

class TestReplaceInvalidBoundaries:

    def test_all_valid_unchanged(self):
        boundaries = [100, 200, 300, 400]
        flags = [True, True, True, True]
        result = replace_invalid_boundaries(boundaries, flags, 1000, 100.0)
        assert result == boundaries

    def test_single_invalid_in_middle_uses_median_cadence(self):
        boundaries = [100, 250, 300, 400]  # 250 is wrong
        flags = [True, False, True, True]
        result = replace_invalid_boundaries(boundaries, flags, 1000, 100.0)
        # Median of valid intervals = median([200, 100]) = 150
        # Project from prev valid (100) + 1 step * 150 = 250 -- by coincidence
        # the result might equal the original. Let's use a clearer test:
        boundaries = [100, 999, 300, 400]
        flags = [True, False, True, True]
        result = replace_invalid_boundaries(boundaries, flags, 2000, 100.0)
        # Valid intervals: 200, 100 -> median 150
        # New boundary[1] = 100 + 150 = 250
        assert result == sorted([100, 250, 300, 400])

    def test_clamp_to_total_frames(self):
        boundaries = [100, 5000, 300]
        flags = [True, False, True]
        result = replace_invalid_boundaries(boundaries, flags, 1000, 100.0)
        assert all(0 <= b < 1000 for b in result)

    def test_no_valid_anchors_uses_expected_interval(self):
        boundaries = [100, 200, 300]
        flags = [False, False, False]
        # All invalid -- with no anchors the function leaves the
        # original frames in place (no projection possible).
        result = replace_invalid_boundaries(boundaries, flags, 1000, 100.0)
        assert sorted(result) == sorted(boundaries)


# ---------------------------------------------------------------------------
# apply_tray_motion_gate -- end-to-end on synthetic data
# ---------------------------------------------------------------------------

class TestApplyTrayMotionGate:

    def test_all_valid_no_rejections(self):
        df = _make_df(2000)
        # Two real cycles at frames 500 and 1500. Pillar lk windows are
        # offset (pre=[F-150,F-50], post=[F+50,F+150]) so place the high
        # and low values accordingly.
        for f in (500, 1500):
            df.loc[f - 30:f - 20, "SABL_x"] = 60.0
            df.loc[f - 10:f + 10, "SABL_x"] = 140.0
            df.loc[f - 150:f - 50, "Pillar_likelihood"] = 1.0
            df.loc[f + 50:f + 150, "Pillar_likelihood"] = 0.05
        boundaries = [500, 1500]
        filtered, rejections = apply_tray_motion_gate(
            df, boundaries, total_frames=2000, expected_interval=1000.0,
        )
        assert filtered == boundaries
        assert rejections == []

    def test_invalid_substituted(self):
        df = _make_df(3000)
        # Real cycles at 500 and 2500. Spurious boundary at 1500 (no signal).
        for f in (500, 2500):
            df.loc[f - 30:f - 20, "SABL_x"] = 60.0
            df.loc[f - 10:f + 10, "SABL_x"] = 140.0
            df.loc[f - 150:f - 50, "Pillar_likelihood"] = 1.0
            df.loc[f + 50:f + 150, "Pillar_likelihood"] = 0.05
        boundaries = [500, 1500, 2500]
        filtered, rejections = apply_tray_motion_gate(
            df, boundaries, total_frames=3000, expected_interval=1000.0,
        )
        assert len(rejections) == 1
        idx, original, reasons = rejections[0]
        assert idx == 1
        assert original == 1500
        # Substitution: median valid interval = 2000, projected from 500
        assert filtered[1] == 2500  # 500 + 2000
