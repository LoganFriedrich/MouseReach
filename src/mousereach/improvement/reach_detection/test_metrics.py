"""
Unit tests for reach detection accuracy metrics.

Tests cover the pure ``match_reaches`` function which is the core
matching logic. No I/O, no file system access.
"""
from __future__ import annotations

import pytest

from mousereach.improvement.reach_detection.metrics import (
    Reach,
    ReachMatchResult,
    match_reaches,
)


def _count(results, status):
    return sum(1 for r in results if r.status == status)


def _start_deltas(results):
    return sorted(
        r.start_delta for r in results if r.status == "matched"
    )


def _end_deltas(results):
    return sorted(
        r.end_delta for r in results if r.status == "matched"
    )


class TestMatchReaches:
    """Tests for the match_reaches function."""

    def test_perfect_match(self):
        """Every algo reach matches exactly one GT, all deltas = 0."""
        gt = [Reach(100, 110, 0), Reach(200, 215, 1), Reach(300, 308, 2)]
        algo = [Reach(100, 110, 0), Reach(200, 215, 1), Reach(300, 308, 2)]
        results = match_reaches(algo, gt, window=10)

        assert _count(results, "matched") == 3
        assert _count(results, "fn") == 0
        assert _count(results, "fp") == 0
        assert all(r.start_delta == 0 for r in results if r.status == "matched")
        assert all(r.end_delta == 0 for r in results if r.status == "matched")

    def test_uniform_positive_3f_bias_on_start(self):
        """Algo starts are all +3f late. All matched, all start_delta = 3."""
        gt = [Reach(100, 110, 0), Reach(200, 215, 1), Reach(300, 308, 2)]
        algo = [Reach(103, 112, 0), Reach(203, 218, 1), Reach(303, 310, 2)]
        results = match_reaches(algo, gt, window=10)

        assert _count(results, "matched") == 3
        assert _count(results, "fn") == 0
        assert _count(results, "fp") == 0
        assert all(r.start_delta == 3 for r in results if r.status == "matched")

    def test_one_fn_algo_emits_fewer(self):
        """Algo emits one fewer than GT -> 2 matched + 1 fn + 0 fp."""
        gt = [Reach(100, 110, 0), Reach(200, 215, 1), Reach(300, 308, 2)]
        algo = [Reach(100, 110, 0), Reach(300, 308, 1)]
        results = match_reaches(algo, gt, window=10)

        assert _count(results, "matched") == 2
        assert _count(results, "fn") == 1
        assert _count(results, "fp") == 0

        # The fn should be GT index 1 (start=200)
        fns = [r for r in results if r.status == "fn"]
        assert fns[0].gt_reach_index == 1
        assert fns[0].gt_start == 200

    def test_one_fp_algo_emits_more(self):
        """Algo emits one more than GT -> 3 matched + 0 fn + 1 fp."""
        gt = [Reach(100, 110, 0), Reach(200, 215, 1), Reach(300, 308, 2)]
        algo = [Reach(100, 110, 0), Reach(150, 160, 1), Reach(200, 215, 2), Reach(300, 308, 3)]
        results = match_reaches(algo, gt, window=10)

        assert _count(results, "matched") == 3
        assert _count(results, "fn") == 0
        assert _count(results, "fp") == 1

        # The fp should be the spurious reach at 150
        fps = [r for r in results if r.status == "fp"]
        assert fps[0].algo_start == 150

    def test_two_algo_within_window_of_one_gt_closest_wins(self):
        """Two algo reaches within +/-10f of same GT -> closest wins, other fp."""
        gt = [Reach(200, 215, 0)]
        # Two algo near GT[0]=200: one at 197, one at 205
        algo = [Reach(197, 212, 0), Reach(205, 220, 1)]
        results = match_reaches(algo, gt, window=10)

        assert _count(results, "matched") == 1
        assert _count(results, "fp") == 1
        assert _count(results, "fn") == 0

        # The closer one (197, delta=-3) should match
        matched = [r for r in results if r.status == "matched"]
        assert matched[0].algo_start == 197
        assert matched[0].start_delta == -3

        # The farther one (205) should be fp
        fps = [r for r in results if r.status == "fp"]
        assert fps[0].algo_start == 205

    def test_boundary_at_window_edge_inclusive(self):
        """Reach exactly at +/-10f window edge -> matched (inclusive)."""
        gt = [Reach(100, 110, 0)]

        # +10f
        algo = [Reach(110, 120, 0)]
        results = match_reaches(algo, gt, window=10)
        assert _count(results, "matched") == 1
        matched = [r for r in results if r.status == "matched"]
        assert matched[0].start_delta == 10

        # -10f
        algo = [Reach(90, 100, 0)]
        results = match_reaches(algo, gt, window=10)
        assert _count(results, "matched") == 1
        matched = [r for r in results if r.status == "matched"]
        assert matched[0].start_delta == -10

    def test_boundary_outside_window(self):
        """Reach at +/-11f -> fp + fn (outside window)."""
        gt = [Reach(100, 110, 0)]

        # +11f, outside
        algo = [Reach(111, 121, 0)]
        results = match_reaches(algo, gt, window=10)
        assert _count(results, "matched") == 0
        assert _count(results, "fn") == 1
        assert _count(results, "fp") == 1

        # -11f, outside
        algo = [Reach(89, 99, 0)]
        results = match_reaches(algo, gt, window=10)
        assert _count(results, "matched") == 0
        assert _count(results, "fn") == 1
        assert _count(results, "fp") == 1

    def test_empty_inputs(self):
        """Empty algo or GT -> all fn/fp respectively."""
        gt = [Reach(100, 110, 0), Reach(200, 215, 1)]
        results = match_reaches([], gt, window=10)
        assert _count(results, "fn") == 2
        assert _count(results, "matched") == 0
        assert _count(results, "fp") == 0

        results = match_reaches([Reach(100, 110, 0), Reach(200, 215, 1)], [], window=10)
        assert _count(results, "fp") == 2
        assert _count(results, "matched") == 0
        assert _count(results, "fn") == 0

    def test_end_deltas_tracked(self):
        """End deltas are computed for matched reaches."""
        gt = [Reach(100, 110, 0), Reach(200, 215, 1)]
        algo = [Reach(102, 115, 0), Reach(198, 220, 1)]
        results = match_reaches(algo, gt, window=10)

        assert _count(results, "matched") == 2
        matched = sorted(
            [r for r in results if r.status == "matched"],
            key=lambda r: r.gt_reach_index,
        )
        assert matched[0].start_delta == 2
        assert matched[0].end_delta == 5   # 115 - 110
        assert matched[1].start_delta == -2
        assert matched[1].end_delta == 5   # 220 - 215

    def test_large_realistic_scenario(self):
        """20 reaches with mixed offsets -- verify counts add up."""
        gt = [Reach(i * 50, i * 50 + 12, i) for i in range(20)]
        # Algo: mostly close, one way off (fn), one extra (fp)
        algo_frames = [(i * 50 + (i % 5) - 2, i * 50 + 12 + (i % 3))
                       for i in range(20)]
        algo_frames[10] = (10 * 50 + 500, 10 * 50 + 512)  # way outside
        algo_frames.append((5 * 50 + 3, 5 * 50 + 15))  # duplicate near gt[5]
        algo = [Reach(s, e, i) for i, (s, e) in enumerate(algo_frames)]

        results = match_reaches(algo, gt, window=10)
        n_matched = _count(results, "matched")
        n_fn = _count(results, "fn")
        n_fp = _count(results, "fp")

        # Total results = matched + fn + fp
        assert len(results) == n_matched + n_fn + n_fp
        # Every GT is either matched or fn
        assert n_matched + n_fn == len(gt)
        # Every algo is either matched or fp
        assert n_matched + n_fp == len(algo)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
