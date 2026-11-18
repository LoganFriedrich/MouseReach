"""A review bundle parked for a new pose is seen by every reader, and the
parked folder is cleared once the round trip has ended.

WHY: mousereach-repose-videos takes a held video's bundle out of its review
queue into Processing/Repose_Queue/_parked/<video>/, which then holds the
video's only copy until a GPU node has posed it again. The census and the
Pipeline Dashboard did not read that folder, so the video vanished from both
for the whole round trip. And nothing ever removed the parked folder after
the round trip: the re-run left a leftover copy behind for good, and a later
re-pose of the same video was refused because the folder existed.

The lab's rule for machine leftovers: deleted once a current copy elsewhere
is verified (same size and SHA-256); never set aside; anything that may be a
person's work is never deleted.

tmp_path only, invented ids; Paths is patched.
"""
import json
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_io as cr_io
import mousereach.watcher.repose as repose
import mousereach.watcher.repose_videos as rv
from mousereach.watcher import reconcile as rc
from mousereach.watcher.db import WatcherDB

VID = "20990102_ABC0102_P2"
SC = "DLC_resnet101_TestModelshuffle1_100"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from mousereach.config import Paths
    import mousereach.config as cfg
    nas = tmp_path / "nas"
    work = tmp_path / "work"
    dirs = {
        "NAS_ROOT": nas,
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "MULTI_ANIMAL_SOURCE": nas / "Unanalyzed" / "Multi-Animal",
        "SINGLE_ANIMAL_OUTPUT": nas / "Unanalyzed" / "Single_Animal",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "FAILED": nas / "Processing" / "Failed",
        "TRIAGE_REVIEW": nas / "Processing" / "Review" / "triage",
        "DEEP_REVIEW": nas / "Processing" / "Review" / "deep_review",
        "REPOSE_QUEUE": nas / "Processing" / "Repose_Queue",
        "PROCESSING_ROOT": work,
        "PROCESSING": work / "Processing",
        "DLC_QUEUE": work / "DLC_Queue",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, name, d, raising=False)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", dirs["TRIAGE_REVIEW"].parent, raising=False)
    monkeypatch.setattr(Paths, "NAS_DRIVE", None, raising=False)

    class _NoWatcherConfig:
        @staticmethod
        def load():
            raise RuntimeError("no watcher config in tests")
    monkeypatch.setattr(cfg, "WatcherConfig", _NoWatcherConfig)
    quarantine = nas / "Processing" / "Quarantine"
    quarantine.mkdir(parents=True)
    monkeypatch.setattr(rc, "_quarantine_dir", lambda: quarantine)
    monkeypatch.setattr(cr_io, "resolve_review_path", lambda stem, primary_dir=None: None)
    monkeypatch.setattr(cr_io, "durable_review_dir", lambda: None)
    (nas / "pipeline_versions.json").write_text(
        json.dumps({"versions": {"dlc_scorer": SC, "segmenter": "2.0"}}))
    results = nas / "Analyzed" / "X" / "C01"
    results.mkdir(parents=True)
    monkeypatch.setattr(rv, "_results_dir", lambda vid: results)
    monkeypatch.setattr(rv, "_durable_review", lambda vid: None)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path=f"C:/x/{VID}.mp4",
                      current_path=f"C:/x/{VID}.mp4")
    parked = dirs["REPOSE_QUEUE"] / rv.PARKED / VID
    return SimpleNamespace(db=db, parked=parked, results=results, **{
        k.lower(): v for k, v in dirs.items()})


def _park(env, video=b"the-video", posed_copy=True):
    """What mousereach-repose-videos leaves in the parked folder."""
    env.parked.mkdir(parents=True)
    (env.parked / f"{VID}.mp4").write_bytes(video)
    (env.parked / f"{VID}_manifest.json").write_text("{}", encoding="utf-8")
    (env.parked / f"{VID}_routing.json").write_text("{}", encoding="utf-8")
    if posed_copy:
        (env.parked / "posed_copy").mkdir()
        (env.parked / "posed_copy" / f"{VID}.mp4").write_bytes(video)


def _bundle(env, queue="triage", video=b"the-video"):
    """The new bundle the re-run routed into a review queue."""
    b = (env.triage_review if queue == "triage" else env.deep_review) / VID
    b.mkdir(parents=True)
    (b / f"{VID}.mp4").write_bytes(video)
    (b / f"{VID}{SC}.h5").write_bytes(b"gpu-pose")
    (b / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    return b


def _state(env, state):
    env.db.force_state(VID, state, reason="test setup")


# ---------------------------------------------------------------- (a) seen

def test_the_dashboard_scan_names_a_parked_video(env):
    from mousereach.dashboard import folder_scan
    _park(env)
    out = folder_scan.scan_pipeline_folders()
    assert out.get(VID, {}).get("current_stage") == "parked_for_repose"
    assert out[VID]["locations"][0]["path"] == str(env.parked)


def test_a_parked_leftover_never_hides_the_videos_new_queue(env):
    from mousereach.dashboard import folder_scan
    _park(env)
    _bundle(env)
    assert folder_scan.scan_pipeline_folders()[VID]["current_stage"] == "triage"


def test_the_census_counts_a_parked_video_as_waiting_for_its_pose(env):
    from mousereach.census import runner
    _park(env)
    c = runner.run_census()
    assert c["sessions"].get(VID, {}).get("element") == "crop_dlc"
    assert c["roots"]["repose_parked"]["exists"] is True


def test_the_dashboard_lists_parked_videos_under_in_progress():
    from mousereach.dashboard.widget import STAGE_FILTERS, stage_label
    assert "parked_for_repose" in dict(STAGE_FILTERS)["In progress"]
    assert stage_label("parked_for_repose")[0] != "parked_for_repose"


# ---------------------------------------------------------------- (c) cleared

def test_a_later_re_pose_clears_a_finished_leftover_instead_of_refusing(env):
    """The round trip ended with the video held in triage again; a person
    re-poses it once more. The old parked copy is verified against the new
    bundle and deleted, and the new bundle is parked in its place."""
    _park(env)
    _bundle(env)
    _state(env, "triage")
    r = rv.plan_and_move(env.db, VID, label="L", reason="again", hostname="srv",
                         declared=SC, dry_run=False)
    assert r["status"] == "requested", r
    assert (env.parked / f"{VID}.mp4").read_bytes() == b"the-video"
    assert not (env.triage_review / VID).exists()


def test_a_leftover_verified_against_the_filed_copy_is_deleted(env):
    _park(env)
    (env.results / f"{VID}.mp4").write_bytes(b"the-video")
    _state(env, "archived")
    r = rv.clear_parked(env.db, VID, dry_run=False)
    assert r["status"] == "cleared", r
    assert not env.parked.exists()
    assert (env.results / f"{VID}.mp4").read_bytes() == b"the-video"


def test_a_dry_run_deletes_nothing(env):
    _park(env)
    _bundle(env, queue="deep_review")
    _state(env, "deep_review")
    r = rv.clear_parked(env.db, VID, dry_run=True)
    assert r["status"] == "would_clear"
    assert (env.parked / f"{VID}.mp4").is_file()


@pytest.mark.parametrize("case", ["differs", "posed_copy_differs", "unknown_file",
                                  "request_out", "mid_run", "no_new_copy"])
def test_anything_unverified_keeps_the_parked_folder(env, case):
    _park(env)
    _bundle(env)
    _state(env, "triage")
    if case == "differs":
        (env.triage_review / VID / f"{VID}.mp4").write_bytes(b"the-videX")
    elif case == "posed_copy_differs":
        (env.parked / "posed_copy" / f"{VID}.mp4").write_bytes(b"other")
    elif case == "unknown_file":
        (env.parked / f"{VID}_causal_review.json").write_text("{}", encoding="utf-8")
    elif case == "request_out":
        (env.repose_queue / f"{VID}.json").write_text("{}", encoding="utf-8")
    elif case == "mid_run":
        env.db.force_state(VID, "processing", reason="test")
    elif case == "no_new_copy":
        import shutil
        shutil.rmtree(env.triage_review / VID)
    r = rv.clear_parked(env.db, VID, dry_run=False)
    assert r["status"] == "kept" and r["why"]
    assert (env.parked / f"{VID}.mp4").read_bytes() == b"the-video"


def test_the_server_clears_finished_leftovers_by_itself(env):
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    _park(env)
    _bundle(env)
    _state(env, "triage")
    orch = SimpleNamespace(db=env.db)
    for name in ("_PARKED_CLEAR_INTERVAL_S", "_PARKED_CLEAR_PER_PASS",
                 "_PARKED_KEPT_RECHECK_S"):
        setattr(orch, name, getattr(ProcessingOrchestrator, name))
    ProcessingOrchestrator._maybe_clear_finished_parked(orch)
    assert not env.parked.exists()


def test_the_command_clears_on_request_and_dry_runs_first(env, monkeypatch, capsys):
    import mousereach.watcher.db_location as dbl
    _park(env)
    _bundle(env)
    _state(env, "triage")
    monkeypatch.setattr(dbl, "open_watcher_db_if_present", lambda: (env.db, ""))
    assert rv.main(["--clear-finished", "--dry-run"]) == 0
    out = capsys.readouterr().out
    out.encode("ascii")
    assert f"{VID}: would_clear" in out and env.parked.exists()
    assert rv.main(["--clear-finished"]) == 0
    assert not env.parked.exists()
