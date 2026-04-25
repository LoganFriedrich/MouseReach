"""
Unit tests for segmentation boundary-accuracy metrics.

Tests cover the pure ``match_boundaries`` function which is the core
matching logic. No I/O, no file system access.
"""
from __future__ import annotations

import pytest

from mousereach.improvement.segmentation.metrics import (
    MatchResult,
    match_boundaries,
)


def _count(results, status):
    return sum(1 for r in results if r.status == status)


def _deltas(results):
    return sorted(
        r.signed_delta for r in results if r.status == "matched"
    )


class TestMatchBoundaries:
    """Tests for the match_boundaries function."""

    def test_perfect_match(self):
        """All 21 algo frames == GT frames -> all matched, delta 0."""
        gt = list(range(0, 21000, 1000))  # 21 boundaries
        algo = list(gt)  # identical
        results = match_boundaries(algo, gt, window=20)

        assert _count(results, "matched") == 21
        assert _count(results, "miss") == 0
        assert _count(results, "phantom") == 0
        assert all(r.signed_delta == 0 for r in results if r.status == "matched")

    def test_all_early_by_7_bias(self):
        """Algo frames = GT - 7 -> all matched, all delta == -7."""
        gt = list(range(0, 21000, 1000))
        algo = [g - 7 for g in gt]
        results = match_boundaries(algo, gt, window=20)

        assert _count(results, "matched") == 21
        assert _count(results, "miss") == 0
        assert _count(results, "phantom") == 0
        assert all(r.signed_delta == -7 for r in results if r.status == "matched")

    def test_one_miss(self):
        """Algo emits 20 (missing middle one) -> 20 matched + 1 miss + 0 phantom."""
        gt = list(range(0, 21000, 1000))  # 21 boundaries
        algo = list(gt)
        del algo[10]  # remove the 11th boundary (index 10)
        assert len(algo) == 20

        results = match_boundaries(algo, gt, window=20)

        assert _count(results, "matched") == 20
        assert _count(results, "miss") == 1
        assert _count(results, "phantom") == 0

        # The miss should be GT index 10
        misses = [r for r in results if r.status == "miss"]
        assert misses[0].gt_boundary_index == 10
        assert misses[0].gt_frame == 10000

    def test_one_phantom(self):
        """Algo emits 22 (extra spurious) -> 21 matched + 1 phantom + 0 miss."""
        gt = list(range(0, 21000, 1000))  # 21 boundaries
        algo = list(gt) + [500]  # extra spurious at frame 500
        algo.sort()
        assert len(algo) == 22

        results = match_boundaries(algo, gt, window=20)

        assert _count(results, "matched") == 21
        assert _count(results, "phantom") == 1
        assert _count(results, "miss") == 0

        # The phantom should be the spurious one at frame 500
        phantoms = [r for r in results if r.status == "phantom"]
        assert phantoms[0].algo_frame == 500

    def test_two_algo_close_to_one_gt(self):
        """Two algo boundaries close to one GT -> closest matched, other phantom."""
        gt = [1000, 2000, 3000]
        # Two algo boundaries near GT[1]=2000: one at 1998, one at 2005
        algo = [1000, 1998, 2005, 3000]

        results = match_boundaries(algo, gt, window=20)

        assert _count(results, "matched") == 3
        assert _count(results, "phantom") == 1
        assert _count(results, "miss") == 0

        # The closer one (1998, delta=-2) should match GT[1]=2000
        matched_to_gt1 = [r for r in results if r.status == "matched" and r.gt_boundary_index == 1]
        assert len(matched_to_gt1) == 1
        assert matched_to_gt1[0].algo_frame == 1998
        assert matched_to_gt1[0].signed_delta == -2

        # The farther one (2005) should be phantom
        phantoms = [r for r in results if r.status == "phantom"]
        assert phantoms[0].algo_frame == 2005

    def test_boundary_at_window_edge_inclusive(self):
        """Boundary exactly at +/-20f window edge -> matched (inclusive)."""
        gt = [1000]
        algo = [1020]  # exactly +20
        results = match_boundaries(algo, gt, window=20)
        assert _count(results, "matched") == 1
        assert results[0].signed_delta == 20

        algo = [980]  # exactly -20
        results = match_boundaries(algo, gt, window=20)
        assert _count(results, "matched") == 1
        assert results[0].signed_delta == -20

    def test_boundary_outside_window(self):
        """Boundary at +/-21f -> miss (outside window)."""
        gt = [1000]
        algo = [1021]  # +21, outside
        results = match_boundaries(algo, gt, window=20)
        assert _count(results, "matched") == 0
        assert _count(results, "miss") == 1
        assert _count(results, "phantom") == 1

        algo = [979]  # -21, outside
        results = match_boundaries(algo, gt, window=20)
        assert _count(results, "matched") == 0
        assert _count(results, "miss") == 1
        assert _count(results, "phantom") == 1

    def test_empty_inputs(self):
        """Empty algo or GT -> all miss/phantom respectively."""
        gt = [1000, 2000, 3000]
        results = match_boundaries([], gt, window=20)
        assert _count(results, "miss") == 3
        assert _count(results, "matched") == 0
        assert _count(results, "phantom") == 0

        results = match_boundaries([1000, 2000], [], window=20)
        assert _count(results, "phantom") == 2
        assert _count(results, "matched") == 0
        assert _count(results, "miss") == 0

    def test_two_gt_claim_same_algo(self):
        """Two GT boundaries equidistant from one algo -> closer wins."""
        # GT at 990 and 1010, algo at 1000 -- both within window
        # 990 is 10 away, 1010 is 10 away -- tie broken by sort stability
        # but both should not match -- one must be a miss
        gt = [990, 1010]
        algo = [1000]
        results = match_boundaries(algo, gt, window=20)

        assert _count(results, "matched") == 1
        assert _count(results, "miss") == 1
        assert _count(results, "phantom") == 0

    def test_large_realistic_scenario(self):
        """21 boundaries with mixed offsets -- verify counts add up."""
        gt = list(range(500, 500 + 21 * 1800, 1800))  # 21 boundaries
        # Algo: mostly close, but one is way off (miss), one extra (phantom)
        algo = [g + (i % 5) - 2 for i, g in enumerate(gt)]
        algo[10] = gt[10] + 500  # way outside window -> will be miss + phantom
        algo.append(gt[5] + 3)   # duplicate near gt[5] -> phantom

        results = match_boundaries(algo, gt, window=20)
        n_matched = _count(results, "matched")
        n_miss = _count(results, "miss")
        n_phantom = _count(results, "phantom")

        # Total results = matched + miss + phantom
        assert len(results) == n_matched + n_miss + n_phantom
        # Every GT boundary is either matched or missed
        assert n_matched + n_miss == len(gt)
        # Every algo boundary is either matched or phantom
        assert n_matched + n_phantom == len(algo)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
