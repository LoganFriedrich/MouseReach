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
