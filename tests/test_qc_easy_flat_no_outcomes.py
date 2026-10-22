"""Easy and Flat trays are not failed for the outcomes file they never get.

WHY: outcome detection is skipped for Easy (E) and Flat (F) trays by design
(watcher/router.SKIP_STEPS), but the quality check's structural layer still
demanded the outcomes file and called its absence CRITICAL -- so the gate sent
every Easy/Flat video to deep review on a false alarm (100 Flat-tray videos in
three hours on 2026-10-22). A Pillar video without its outcomes file is still
a critical failure.
"""
import json

from mousereach.pipeline.triage import EXPECTED_SEGMENTS_MIN, triage_video


def _outputs(d, stem, outcomes):
    (d / f"{stem}_segments.json").write_text(
        json.dumps({"boundaries": list(range(0, 1000 * EXPECTED_SEGMENTS_MIN, 1000))}),
        encoding="utf-8")
    (d / f"{stem}_reaches.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
    if outcomes:
        (d / f"{stem}_pellet_outcomes.json").write_text(json.dumps({"segments": []}),
                                                          encoding="utf-8")


def _missing_outcomes(result):
    return [f for f in result.flags if "Missing outcomes" in f.description]


def test_flat_and_easy_trays_do_not_need_an_outcomes_file(tmp_path):
    for stem in ("20261012_CNT0615_F4", "20261012_CNT0615_E2"):
        _outputs(tmp_path, stem, outcomes=False)
        r = triage_video(stem, tmp_path, skip_dlc_check=True)
        assert not _missing_outcomes(r), stem
        assert r.verdict == "auto_approved", (stem, [f.description for f in r.flags])


def test_a_pillar_tray_still_needs_its_outcomes_file(tmp_path):
    stem = "20261012_CNT0615_P4"
    _outputs(tmp_path, stem, outcomes=False)
    r = triage_video(stem, tmp_path, skip_dlc_check=True)
    assert _missing_outcomes(r) and _missing_outcomes(r)[0].severity == "critical"
    assert r.verdict == "needs_review"


def test_an_unrecognised_name_is_held_to_the_full_set(tmp_path):
    stem = "odd_name"
    _outputs(tmp_path, stem, outcomes=False)
    assert _missing_outcomes(triage_video(stem, tmp_path, skip_dlc_check=True))
