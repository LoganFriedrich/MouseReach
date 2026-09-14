"""mousereach-reconcile: every video judged against the definition of done.

Done = analysis current (or declared compatible) with every saved review
reflected, AND sitting beside the video where its name says it belongs. The
check reads files only and writes nothing.
"""
import hashlib
import json

import pytest

import mousereach.review.causal_review_io as cr_io
from mousereach.archive.core import get_archive_destination
from mousereach.watcher import reconcile as rc

VID = "20250101_CNT0101_P1"
VID2 = "20250101_CNT0102_P1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    dirs = {
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "TRIAGE_REVIEW": nas / "Processing" / "Review" / "triage",
        "DEEP_REVIEW": nas / "Processing" / "Review" / "flagged_for_review",
        "FAILED": nas / "Processing" / "Failed",
    }
    for d in dirs.values():
        d.mkdir(parents=True)
    monkeypatch.setattr(rc_paths(), "NAS_ROOT", nas)
    for name, d in dirs.items():
        monkeypatch.setattr(rc_paths(), name, d)
    quarantine = nas / "Processing" / "Quarantine"
    quarantine.mkdir(parents=True)
    monkeypatch.setattr(rc, "_quarantine_dir", lambda: quarantine)
    reviews = {}
    monkeypatch.setattr(cr_io, "resolve_review_path",
                        lambda stem, primary_dir=None: reviews.get(stem))
    declare(nas, {"dlc_scorer": "NEW", "segmenter": "2.0"})
    return type("Env", (), {"nas": nas, "reviews": reviews, "tmp": tmp_path, **dirs})


def rc_paths():
    from mousereach.config import Paths
    return Paths


def declare(nas, versions, compat=None):
    doc = {"versions": versions}
    if compat:
        doc["compatible_versions"] = compat
    (nas / "pipeline_versions.json").write_text(json.dumps(doc))


def put_analyzed(stem, seg="2.0", scorer="NEW", folder=None, video=True, **manifest_extra):
    folder = folder or get_archive_destination(stem)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"pipeline_versions": {"segmenter": seg},
                "dlc_model": {"dlc_scorer": scorer},
                "created_at": "2026-01-01T00:00:00", **manifest_extra}
    (folder / f"{stem}_processing_manifest.json").write_text(json.dumps(manifest))
    (folder / f"{stem}_features.json").write_text("{}")
    if video:
        (folder / f"{stem}.mp4").write_bytes(b"video")
    return folder


def verdict_of(stem):
    rows = {r["video_id"]: r for r in rc.reconcile()["rows"]}
    return rows[stem]


def test_current_video_in_its_cohort_folder_is_done(env):
    put_analyzed(VID)
    assert verdict_of(VID)["verdict"] == rc.DONE


def test_outdated_stage_is_a_mismatch_named_per_video(env):
    put_analyzed(VID, seg="1.0")
    row = verdict_of(VID)
    assert row["verdict"] == rc.NOT_CURRENT
    assert "segmenter" in row["detail"]
    assert [r["video_id"] for r in rc.reconcile()["mismatches"]] == [VID]


def test_old_pose_model_is_not_current(env):
    put_analyzed(VID, scorer="OLD")
    row = verdict_of(VID)
    assert row["verdict"] == rc.NOT_CURRENT
    assert "older model" in row["detail"]


def test_declared_compatible_version_counts_as_current(env):
    declare(env.nas, {"dlc_scorer": "NEW", "segmenter": "2.0"}, compat={"segmenter": ["1.9"]})
    put_analyzed(VID, seg="1.9")
    assert verdict_of(VID)["verdict"] == rc.DONE


def test_human_segmentation_is_never_outdated(env):
    folder = put_analyzed(VID, seg="1.0")
    (folder / f"{VID}_segments.json").write_text(json.dumps({"boundary_source": "human"}))
    assert verdict_of(VID)["verdict"] == rc.DONE


def test_saved_review_not_reflected_is_a_mismatch(env):
    put_analyzed(VID)
    review = env.tmp / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"reviewed_at": "2026-02-01T00:00:00"}))
    env.reviews[VID] = review
    row = verdict_of(VID)
    assert row["verdict"] == rc.NOT_CURRENT
    assert "review" in row["detail"]


def test_saved_review_applied_by_content_is_done(env):
    review = env.tmp / f"{VID}_causal_review.json"
    review.write_text(json.dumps({"reviewed_at": "2026-02-01T00:00:00"}))
    sha = hashlib.sha256(review.read_bytes()).hexdigest()
    put_analyzed(VID, applied_review={"sha256": sha})
    env.reviews[VID] = review
    assert verdict_of(VID)["verdict"] == rc.DONE


def test_analysis_in_another_cohort_folder_is_wrong_place(env):
    wrong = env.ANALYZED_OUTPUT / "Connectome" / "CNT09"
    put_analyzed(VID, folder=wrong)
    row = verdict_of(VID)
    assert row["verdict"] == rc.WRONG_PLACE
    assert "CNT09" in row["detail"]


def test_second_copy_elsewhere_is_wrong_place_even_when_current(env):
    put_analyzed(VID)
    put_analyzed(VID, folder=env.ANALYZED_OUTPUT / "Connectome" / "CNT09")
    assert verdict_of(VID)["verdict"] == rc.WRONG_PLACE


def test_current_analysis_without_its_video_is_wrong_place(env):
    put_analyzed(VID, video=False)
    assert verdict_of(VID)["verdict"] == rc.WRONG_PLACE


def test_video_held_for_a_person_is_fine(env):
    put_analyzed(VID, seg="1.0")
    (env.TRIAGE_REVIEW / VID).mkdir()
    (env.DEEP_REVIEW / VID2).mkdir()
    assert verdict_of(VID)["verdict"] == rc.HELD
    assert verdict_of(VID2)["verdict"] == rc.HELD
    assert rc.reconcile()["mismatches"] == []


def test_done_video_still_in_a_queue_is_a_stray_bundle(env):
    put_analyzed(VID)
    (env.DEEP_REVIEW / VID).mkdir()
    assert verdict_of(VID)["verdict"] == rc.STRAY_BUNDLE


def test_failed_and_quarantined_need_a_person(env):
    (env.FAILED / f"{VID}.mp4").write_bytes(b"v")
    (env.nas / "Processing" / "Quarantine" / f"{VID2}.mp4").write_bytes(b"v")
    assert verdict_of(VID)["verdict"] == rc.NEEDS_PERSON
    assert verdict_of(VID2)["verdict"] == rc.NEEDS_PERSON


def test_copy_in_an_old_folder_is_a_leftover_not_a_mismatch(env):
    put_analyzed(VID)
    old = env.nas / "Processing" / "Single_Animal"
    old.mkdir(parents=True)
    (old / f"{VID}.mp4").write_bytes(b"v")
    result = rc.reconcile()
    assert verdict_of(VID)["verdict"] == rc.DONE
    assert result["leftover_copies"] == [VID]
    assert result["mismatches"] == []


def test_only_copy_in_an_old_folder_is_listed_per_video(env):
    old = env.nas / "Processing" / "DLC_Complete"
    old.mkdir(parents=True)
    (old / f"{VID}.mp4").write_bytes(b"v")
    row = verdict_of(VID)
    assert row["verdict"] == rc.ONLY_IN_OLD_FOLDER
    assert "Processing/DLC_Complete" in row["detail"]
    assert rc.reconcile()["leftover_copies"] == []


def test_single_waiting_in_a_stage_folder(env):
    waiting = env.nas / "Unanalyzed" / "Single_Animal"
    waiting.mkdir(parents=True)
    (waiting / f"{VID}.mp4").write_bytes(b"v")
    assert verdict_of(VID)["verdict"] == rc.WAITING


def test_unsupported_tray_is_out_of_scope(env):
    stem = "20250101_CNT0101_E1"
    put_analyzed(stem, seg="1.0")
    assert verdict_of(stem)["verdict"] == rc.UNSUPPORTED_TRAY
    assert rc.reconcile()["mismatches"] == []


def test_skipped_folders_do_not_invent_videos(env):
    cohort = get_archive_destination(VID)
    for sub in ("DLC Model 4", "Multi-Animal", "_retired", ".inflight"):
        put_analyzed(VID, seg="1.0", folder=cohort / sub)
    (env.TRIAGE_REVIEW / "_Problematic").mkdir()
    (env.TRIAGE_REVIEW / ".return_claims").mkdir()
    assert rc.reconcile()["rows"] == []


def test_collages_are_not_judged_as_singles(env):
    collage = "20250101_CNT0101,CNT0102,CNT0103_P1"
    quarantine = env.nas / "Processing" / "Quarantine"
    (quarantine / f"{collage}.mkv").write_bytes(b"c")
    (quarantine / f"{collage}.mkv.quarantine.json").write_text("{}")
    (quarantine / f"{VID}.mp4").write_bytes(b"v")
    assert [r["video_id"] for r in rc.reconcile()["rows"]] == [VID]


def test_video_without_manifest_names_its_folder(env):
    folder = get_archive_destination(VID)
    folder.mkdir(parents=True)
    (folder / f"{VID}.mp4").write_bytes(b"v")
    row = verdict_of(VID)
    assert row["verdict"] == rc.NOT_CURRENT
    assert "Analyzed/Connectome/CNT01" in row["detail"]


def test_nothing_is_written(env):
    put_analyzed(VID, seg="1.0")
    (env.TRIAGE_REVIEW / VID2).mkdir()

    def snapshot():
        return {p: p.stat().st_mtime_ns for p in env.nas.rglob("*")}

    before = snapshot()
    rc.reconcile()
    rc.main(["--json"])
    assert snapshot() == before


def test_exit_codes(env, capsys):
    put_analyzed(VID)
    assert rc.main([]) == 0
    put_analyzed(VID2, seg="1.0")
    assert rc.main([]) == 1
    out = capsys.readouterr().out
    assert VID2 in out
    out.encode("ascii")
    (env.nas / "pipeline_versions.json").unlink()
    assert rc.main([]) == 2
