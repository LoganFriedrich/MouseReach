"""compatible_versions: a declared-compatible manifest version must not stale
a video. WHY: two August 2026 declaration edits with no algorithm change
behind them marked 1,255 rows outdated, and 'outdated' was a one-way door.
A bugfix bump whose output only changes for pathological videos (which are
re-marked by hand) must not re-run the whole corpus."""

from mousereach.pipeline.versions import compare_manifest_to_current


def _manifest(seg_version):
    return {"pipeline_versions": {"segmenter": seg_version}, "dlc_model": {}}


def test_compatible_version_is_current():
    current = {
        "versions": {"segmenter": "2.2.4"},
        "compatible_versions": {"segmenter": ["2.2.3"]},
    }
    r = compare_manifest_to_current(_manifest("2.2.3"), current)
    assert r["is_current"] is True
    assert "segmenter" not in r["stale_components"]
    # The caller that un-marks staleness must be able to see that this row is
    # current only BY COMPAT -- a hand-mark on it must survive the two-way door.
    assert r["compat_used"] == ["segmenter"]


def test_non_compatible_version_is_stale():
    current = {
        "versions": {"segmenter": "2.2.4"},
        "compatible_versions": {"segmenter": ["2.2.3"]},
    }
    r = compare_manifest_to_current(_manifest("2.2.2"), current)
    assert r["is_current"] is False
    assert "segmenter" in r["stale_components"]


def test_exact_match_still_current_without_compat_key():
    current = {"versions": {"segmenter": "2.2.4"}}
    r = compare_manifest_to_current(_manifest("2.2.4"), current)
    assert r["is_current"] is True
    assert r["compat_used"] == []  # outright current; the door may un-mark
