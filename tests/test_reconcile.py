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
        "DEEP_REVIEW": nas / "Processing" / "Review" / "deep_review",
        "FAILED": nas / "Processing" / "Failed",
        # The waiting folders are patched like every other root: reconcile
        # reads them from Paths at call time, so the tests never depend on
        # the machine's configured share.
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
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


LEFTOVERS = "Processing/_leftovers_pending_cleanup_2026-09-14"


def leftovers(env, name):
    folder = env.nas / LEFTOVERS / name
    folder.mkdir(parents=True)
    return folder


def test_copy_in_an_old_folder_is_a_leftover_not_a_mismatch(env):
    put_analyzed(VID)
    (leftovers(env, "Single_Animal") / f"{VID}.mp4").write_bytes(b"v")
    result = rc.reconcile()
    assert verdict_of(VID)["verdict"] == rc.DONE
    assert result["leftover_copies"] == [VID]
    assert result["mismatches"] == []


def test_only_copy_in_an_old_folder_is_listed_per_video(env):
    (leftovers(env, "DLC_Complete") / f"{VID}.mp4").write_bytes(b"v")
    row = verdict_of(VID)
    assert row["verdict"] == rc.ONLY_IN_OLD_FOLDER
    assert f"{LEFTOVERS}/DLC_Complete" in row["detail"]
    assert rc.reconcile()["leftover_copies"] == []


def test_every_leftovers_folder_is_read_and_stray_files_are_skipped(env):
    # Any dated leftovers folder counts (a pattern, not one name), and a file
    # at either level of that walk is skipped rather than listed or crashed on.
    (leftovers(env, "DLC_Complete") / f"{VID}.mp4").write_bytes(b"v")
    later = env.nas / "Processing" / "_leftovers_pending_cleanup_2027-01-01" / "Single_Animal"
    later.mkdir(parents=True)
    (later / f"{VID2}.mp4").write_bytes(b"v")
    (env.nas / LEFTOVERS / "README.txt").write_text("kept until cleanup")
    (env.nas / "Processing" / "_leftovers_pending_cleanup_note.txt").write_text("x")
    rows = {r["video_id"]: r for r in rc.reconcile()["rows"]}
    assert set(rows) == {VID, VID2}
    assert rows[VID]["found_in"] == [f"old:{LEFTOVERS}/DLC_Complete"]
    assert rows[VID2]["found_in"] == [
        "old:Processing/_leftovers_pending_cleanup_2027-01-01/Single_Animal"]
    assert all(r["verdict"] == rc.ONLY_IN_OLD_FOLDER for r in rows.values())


@pytest.mark.parametrize("stage, rel", [
    ("SINGLE_ANIMAL_OUTPUT", "Unanalyzed/Single_Animal"),
    ("DLC_STAGING", "Processing/Posed"),
])
def test_single_waiting_in_a_stage_folder(env, stage, rel):
    (getattr(env, stage) / f"{VID}.mp4").write_bytes(b"v")
    row = verdict_of(VID)
    assert row["verdict"] == rc.WAITING
    assert row["found_in"] == ["waiting:" + rel]


def test_waiting_folders_come_from_paths_at_call_time(env, monkeypatch):
    # The watcher moves singles into Paths' folders; reconcile must look in
    # the same ones, not in a copy of their names. Repoint Paths after import:
    # the video there is waiting, and the folder Paths no longer names is not read.
    elsewhere = env.nas / "Processing" / "Posed_elsewhere"
    elsewhere.mkdir()
    (elsewhere / f"{VID}.mp4").write_bytes(b"v")
    (env.DLC_STAGING / f"{VID2}.mp4").write_bytes(b"v")
    monkeypatch.setattr(rc_paths(), "DLC_STAGING", elsewhere)
    rows = {r["video_id"]: r for r in rc.reconcile()["rows"]}
    assert set(rows) == {VID}
    assert rows[VID]["verdict"] == rc.WAITING
    assert rows[VID]["found_in"] == ["waiting:Processing/Posed_elsewhere"]


def test_guard_files_at_retired_folder_names_do_not_crash(env, monkeypatch):
    # The layout change leaves a plain FILE at each retired folder name so old
    # code fails loudly. Reconcile must neither crash on them nor read them.
    processing = env.nas / "Processing"
    guards = {
        "SINGLE_ANIMAL_OUTPUT": processing / "Single_Animal",
        "DLC_STAGING": processing / "DLC_Complete",
        "DEEP_REVIEW": processing / "Review" / "flagged_for_review",
    }
    for guard in guards.values():
        guard.write_text("retired folder -- see the new layout")
    put_analyzed(VID)
    (env.DLC_STAGING / f"{VID2}.mp4").write_bytes(b"v")
    rows = {r["video_id"]: r for r in rc.reconcile()["rows"]}
    assert rows[VID]["verdict"] == rc.DONE
    assert rows[VID2]["verdict"] == rc.WAITING

    # Even a stale config that still names the retired folders reads them as
    # empty instead of raising on iterdir.
    for name, guard in guards.items():
        monkeypatch.setattr(rc_paths(), name, guard)
    result = rc.reconcile()
    assert [r["video_id"] for r in result["rows"]] == [VID]
    assert result["mismatches"] == []
    assert result["unmigrated_folders"] == []   # a guard file is the migrated state
    assert rc.main([]) == 0


def test_unmigrated_share_is_reported_not_silently_ignored(env, capsys):
    # The retired names are still real FOLDERS: this share was never migrated,
    # and work waiting there is invisible to every stage walk. Reconcile must
    # say so loudly -- without listing those folders -- and not exit clean.
    from mousereach.pipeline.pipe_structure import RETIRED_DIRS
    put_analyzed(VID)
    for rel in RETIRED_DIRS:
        (env.nas / rel).mkdir(parents=True)
    (env.nas / "Processing" / "DLC_Complete" / f"{VID2}.mp4").write_bytes(b"v")

    result = rc.reconcile()
    assert result["unmigrated_folders"] == list(RETIRED_DIRS)
    assert [r["video_id"] for r in result["rows"]] == [VID]   # never listed
    assert result["mismatches"] == []

    assert rc.main([]) == 1
    out = capsys.readouterr().out
    out.encode("ascii")
    for rel in RETIRED_DIRS:
        assert rel in out
    assert "not been migrated" in out
    # The warning comes before the verdicts, so it is read first.
    assert out.index("not been migrated") < out.index("MISMATCHES")

    assert rc.main(["--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["unmigrated_folders"] == list(RETIRED_DIRS)
    assert "not been migrated" in captured.err


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


def test_superseded_analysis_under_archive_is_not_read(env):
    # REGRESSION GUARD, not proof of a fix: reconcile already pruned "Archive"
    # at every depth before this branch (it was literally in _ANALYZED_SKIP).
    # The branch only rebuilt _ANALYZED_SKIP from the shared
    # analyzed_tree.SUPERSEDED_DIR_NAMES; this pins that the refactor kept
    # Archive pruned. Why it matters: superseded outputs keep their ORIGINAL
    # names under Analyzed/Archive/. If the walk entered Archive, the older
    # generation would be a second manifest for VID outside its cohort folder
    # -> WRONG_PLACE for a video that is done, and a stem that exists ONLY in
    # Archive would be invented as a video.
    live = put_analyzed(VID)
    archive = env.ANALYZED_OUTPUT / "Archive"
    put_analyzed(VID, seg="1.0", folder=archive / "superseded_processing_root_3.1")
    put_analyzed(VID, seg="1.0", folder=archive / "Connectome" / "CNT01")
    put_analyzed(VID2, folder=archive / "superseded_processing_root_3.1")

    walked = rc.walk_analyzed(env.ANALYZED_OUTPUT)
    assert walked[VID]["manifest"] == [live / f"{VID}_processing_manifest.json"]
    assert VID2 not in walked

    result = rc.reconcile()
    row = {r["video_id"]: r for r in result["rows"]}[VID]
    assert row["verdict"] == rc.DONE
    assert VID2 not in {r["video_id"] for r in result["rows"]}
    assert result["mismatches"] == []


# --- ASPA discovery helpers: the same Archive rule, over Analyzed/<cohort> ---
# They live here because they are small iterdir walks of Analyzed with no data
# dependency. Each builds a live cohort AND a same-shaped decoy under
# Analyzed/Archive; the old iterdir listed "Archive" as a cohort, so each
# assertion below fails against it.

def test_aspa_feed_skips_archive_as_a_cohort(tmp_path):
    from mousereach.aspa.feed import find_single_animal_videos

    analyzed = tmp_path / "Analyzed"
    live = analyzed / "H" / "Single_Animal"
    live.mkdir(parents=True)
    (live / "H0101_P1.mp4").write_bytes(b"v")
    decoy = analyzed / "Archive" / "Single_Animal"
    decoy.mkdir(parents=True)
    (decoy / "H0101_P1.mp4").write_bytes(b"old")
    # A superseded folder nested inside a live cohort is not walked either
    # (the old rglob found this one).
    nested = live / "Archive"
    nested.mkdir()
    (nested / "H0102_P1.mp4").write_bytes(b"old")

    found = list(find_single_animal_videos(tmp_path))
    assert found == [("H", live / "H0101_P1.mp4")]


def test_aspa_importer_skips_archive_as_a_cohort(tmp_path):
    from mousereach.aspa.importer import find_post_processing_dirs

    analyzed = tmp_path / "Analyzed"
    live = analyzed / "H" / "Post-Processing"
    live.mkdir(parents=True)
    (live / "H0101_P1.xlsx").write_bytes(b"x")
    decoy = analyzed / "Archive" / "Post-Processing"
    decoy.mkdir(parents=True)
    (decoy / "H0101_P1.xlsx").write_bytes(b"old")

    assert list(find_post_processing_dirs(tmp_path)) == [("H", live)]


def test_aspa_sync_skips_superseded_folders_as_cohort_and_as_video(tmp_path):
    from mousereach.aspa.sync import find_reprocessed_videos

    stem = "20220811_H01_P3"
    aspa = tmp_path / "Analyzed" / "ASPA"
    cohort = aspa / "H"
    cohort.mkdir(parents=True)
    (cohort / f"{stem}_reaches.json").write_text("{}")          # live, flat
    (cohort / "Archive").mkdir()
    (cohort / "Archive" / f"{stem}_reaches.json").write_text("{}")  # decoy
    (aspa / "Archive").mkdir()
    (aspa / "Archive" / f"{stem}_reaches.json").write_text("{}")    # decoy

    # Old code yielded ("Archive", stem, "H01", ASPA/Archive) -- archived
    # reaches synced as a cohort -- and ("H", "Archive", "UNKNOWN", H/Archive),
    # which also made H look per-video and hid the live flat file.
    expected = [("H", stem, "H01", cohort)]
    assert list(find_reprocessed_videos(tmp_path)) == expected
    assert list(find_reprocessed_videos(tmp_path, cohort="H")) == expected
