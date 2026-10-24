"""A recording retires when its videos' reviews have been APPLIED, judged the
way the re-run scan judges it (content first, file times last).

Until 2026-10-24 retirement compared modified times: a review saved seconds
after the run that applied it looked unapplied forever, and 196 recordings sat
in the intake with every review applied.
"""
import json
import os

from mousereach.pipeline.fsutil import sha256_file
from mousereach.video_prep.core.collage_provenance import review_applied


def _review(tmp_path, reviewed_at="2026-10-01T14:51:31", touch=None):
    p = tmp_path / "x_causal_review.json"
    p.write_text(json.dumps({"reviewed_at": reviewed_at, "segments": []}))
    if touch is not None:
        os.utime(p, (touch, touch))
    return p


def test_the_applied_stamp_decides_even_when_the_review_file_is_newer(tmp_path):
    r = _review(tmp_path, touch=2_000_000_000)          # file time far after the run
    manifest = {"created_at": "2026-10-01T15:01:30",
                "applied_review": {"sha256": sha256_file(r)}}
    assert review_applied(manifest, r, review_mtime=2_000_000_000,
                          features_mtime=1_000_000_000)


def test_a_review_changed_since_the_run_is_not_applied(tmp_path):
    r = _review(tmp_path)
    manifest = {"created_at": "2026-10-01T15:01:30",
                "applied_review": {"sha256": "0" * 64}}
    assert not review_applied(manifest, r)


def test_without_a_stamp_the_reviews_own_time_is_compared_with_the_run(tmp_path):
    r = _review(tmp_path, reviewed_at="2026-10-01T14:51:31", touch=2_000_000_000)
    assert review_applied({"created_at": "2026-10-01T15:01:30"}, r,
                          features_mtime=1_000_000_000)
    r = _review(tmp_path, reviewed_at="2026-10-01T15:30:00")
    assert not review_applied({"created_at": "2026-10-01T15:01:30"}, r)


def test_file_times_only_when_nothing_better_is_known(tmp_path):
    p = tmp_path / "y_causal_review.json"
    p.write_text("{}")
    os.utime(p, (1_000, 1_000))
    assert review_applied({}, p, features_mtime=2_000)
    assert not review_applied({}, p, features_mtime=500)
    assert not review_applied({}, p)                     # nothing to compare: stays


def test_a_hand_cut_video_is_not_staled_by_the_segmenter(tmp_path):
    import json as _json
    from mousereach.video_prep.core.collage_provenance import _drop_human_seg_staleness
    (tmp_path / "v_segments.json").write_text(_json.dumps({"boundary_source": "human"}))
    cmp = {"is_current": False, "stale_components": ["segmenter"], "needs_full_reprocess": False}
    assert _drop_human_seg_staleness("v", tmp_path, cmp)["is_current"]
    (tmp_path / "w_segments.json").write_text(_json.dumps({"boundary_source": "algorithm"}))
    assert not _drop_human_seg_staleness("w", tmp_path, cmp)["is_current"]


def test_an_easy_tray_video_is_complete_once_its_session_file_exists(tmp_path):
    """Easy/Flat trays get no outcomes, so their manifests never record an
    outcome or assignment version; the re-run scan never re-runs them for
    that, and retirement must not wait for it either."""
    import json as _json
    from mousereach.video_prep.core.collage_provenance import build_complete_stems
    nas = tmp_path / "nas"
    nas.mkdir()
    (nas / "pipeline_versions.json").write_text(_json.dumps({"versions": {
        "segmenter": "2.2.4", "reach_detector": "8.1.0", "outcome_detector": "6.1.0",
        "assignment": "2.1.0", "kinematic_extractor": "2.1.0"}}))
    d = tmp_path / "Analyzed" / "PROJ" / "C1"
    d.mkdir(parents=True)
    man = {"pipeline_versions": {"segmenter": "2.2.4", "reach_detector": "8.1.0",
                                 "kinematic_extractor": "2.1.0"}}
    for stem in ("20250101_ABCD0101_E1", "20250101_ABCD0102_E1", "20250101_ABCD0103_P1"):
        (d / f"{stem}_processing_manifest.json").write_text(_json.dumps(man))
    (d / "20250101_ABCD0101_E1_features.json").write_text("{}")
    (d / "20250101_ABCD0103_P1_features.json").write_text("{}")
    done = build_complete_stems(analyzed_root=tmp_path / "Analyzed", nas_root=nas)
    assert "20250101_ABCD0101_E1" in done          # Easy, session file present
    assert "20250101_ABCD0102_E1" not in done      # Easy, no session file yet
    assert "20250101_ABCD0103_P1" not in done      # Pillar still needs every version
