"""Duplicate-boundary repair: a duplicate boundary value becomes a zero-length
segment downstream (no reaches, nan calibration, automatic residual triage),
so segmentation must never emit one. 21 videos across the corpus carried one
as of 2026-09-01, from three producers: SABL double-snap, end-clamp padding,
tray-gate projection."""

from mousereach.segmentation.core.segmenter_multi import _dedupe_boundaries


def _strictly_increasing(vals):
    return all(b > a for a, b in zip(vals, vals[1:]))


def test_no_duplicates_is_a_passthrough():
    b = [100, 200, 300, 400, 500]
    out, n_dup = _dedupe_boundaries(b, 5, 100.0)
    assert out == b
    assert n_dup == 0


def test_mid_video_duplicate_reinserted_in_largest_gap():
    # The observed shape: a missed tray advance leaves a double-length gap and
    # the duplicate sits elsewhere. The replacement belongs inside the gap.
    b = [100, 200, 300, 300, 500, 600]
    out, n_dup = _dedupe_boundaries(b, 6, 100.0)
    assert n_dup == 1
    assert len(out) == 6
    assert _strictly_increasing(out)
    assert 300 < out[3] < 500  # replacement landed inside the double gap


def test_end_clamp_multiple_duplicates():
    # The padding path clamps at the final frame, duplicating it on every
    # further iteration (the E-tray triple-duplicate shape).
    b = [100, 200, 300, 400, 500, 500, 500]
    out, n_dup = _dedupe_boundaries(b, 7, 100.0)
    assert n_dup == 2
    assert len(out) == 7
    assert _strictly_increasing(out)


def test_pathological_adjacent_values_cannot_loop():
    # When there is genuinely nowhere to insert, return a short count rather
    # than spin: an honest 2 boundaries beats an invented duplicate.
    out, n_dup = _dedupe_boundaries([5, 6, 6], 3, 1.0)
    assert out == [5, 6]
    assert n_dup == 1
