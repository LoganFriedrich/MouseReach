"""The "Run All Steps" tab runs one chosen video through the watcher's own steps.

WHY: running by hand from a window must do what the watcher does -- same steps,
order, gates, holds and outputs, one implementation. The tab used to run its own
older chain in the chosen video's own folder: no hand-score comparison, a quality
check that let a video through when it could not run, a database push, and no
check of where the video was. Picking a filed video's mp4 rewrote its filed
results in place; picking a video in a triage bundle rewrote the bundle's
algorithm outputs under the reviewer.

These tests stand in the four algorithm steps with functions that WRITE (so a
run that reaches them changes files on disk) and the watcher's per-video
handlers with recorders, then press the tab's code. Video ids are invented.
"""
import json
from pathlib import Path

import pytest

# A date in the past: the watcher's intake check quarantines a future date.
V = "20200101_ABC9901_P1"
POSE = f"{V}DLC_resnet101_X_shuffle3_100000.h5"


@pytest.fixture
def env(tmp_path, monkeypatch):
    # Imported before require_processing_root is stood in (a module imported
    # afterwards would keep the stand-in by name for the rest of the run).
    from mousereach.watcher import orchestrator as orch_mod  # noqa: F401
    from mousereach.pipeline import run_all  # noqa: F401
    try:
        from mousereach.watcher import run_one  # noqa: F401
    except ImportError:
        pass

    from mousereach.config import Paths, WatcherConfig
    import mousereach.config as mr_config
    nas = tmp_path / "nas"
    roots = {
        "PROCESSING_ROOT": tmp_path / "local",
        "PROCESSING": tmp_path / "local" / "Processing",
        "ANALYZED_OUTPUT": nas / "Analyzed",
        "TRIAGE_REVIEW": nas / "Processing" / "Review" / "triage",
        "DEEP_REVIEW": nas / "Processing" / "Review" / "deep_review",
        "DLC_STAGING": nas / "Processing" / "Posed",
        "NAS_ROOT": None,
    }
    for k, v in roots.items():
        if v is not None:
            v.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Paths, k, v, raising=False)
    cfg = WatcherConfig({"mode": "processing_server",
                         "db_path": str(tmp_path / "local" / "watcher.db")})
    monkeypatch.setattr(WatcherConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(mr_config, "require_processing_root",
                        lambda: tmp_path / "local")
    monkeypatch.setattr(orch_mod, "require_processing_root",
                        lambda: tmp_path / "local", raising=False)

    from mousereach.watcher import singleton, health
    monkeypatch.setattr(singleton, "acquire_watcher_lock", lambda *a, **k: object())
    monkeypatch.setattr(singleton, "release_watcher_lock", lambda h: None)
    monkeypatch.setattr(singleton, "stop_flag_left", lambda p: None)
    monkeypatch.setattr(health, "watcher_running", lambda: None)

    # The algorithm steps, stood in: each one that runs WRITES its output.
    steps = []

    def _stem(dlc_path):
        return Path(dlc_path).name.split("DLC")[0]

    def seg(dlc_path, *a, **k):
        steps.append("segmentation")
        p = Path(dlc_path)
        (p.parent / f"{_stem(p)}_segments.json").write_text('{"rewritten": true}')
        return {"success": True}

    def reach(dlc_path, seg_path, *a, **k):
        steps.append("reach")
        p = Path(dlc_path)
        (p.parent / f"{_stem(p)}_reaches.json").write_text('{"rewritten": true}')
        return {}

    def outcome(dlc_path, seg_path, reach_path, *a, **k):
        steps.append("outcome")
        p = Path(dlc_path)
        (p.parent / f"{_stem(p)}_pellet_outcomes.json").write_text('{"rewritten": true}')
        return {}

    def assign(folder, video_id, dlc_path, *a, **k):
        steps.append("assignment")
        (Path(folder) / f"{video_id}_reach_assignments.json").write_text('{"rewritten": true}')
        return {}

    import mousereach.segmentation.core.batch as sb
    import mousereach.reach.core.batch as rb
    import mousereach.outcomes.core.batch as ob
    import mousereach.assignment.run as ar
    monkeypatch.setattr(sb, "process_single", seg)
    monkeypatch.setattr(rb, "process_single", reach)
    monkeypatch.setattr(ob, "process_single", outcome)
    monkeypatch.setattr(ar, "assign_reaches_for_video", assign)

    pushed = []
    try:
        import mousereach.sync.database as sync_db
        monkeypatch.setattr(sync_db, "sync_file_to_database",
                            lambda *a, **k: pushed.append(a))
    except ImportError:
        pass

    # The watcher's per-video handlers, stood in: each records that the
    # watcher's own dispatch called it and moves the row on as the real one
    # would on success.
    dispatched = []
    PO = orch_mod.ProcessingOrchestrator

    def _intake(self, work):
        dispatched.append("intake")
        self.db.update_state(work["id"], "processing")
        return True

    def _pipeline(self, work):
        dispatched.append("pipeline")
        self.db.update_state(work["id"], "processed")
        return True

    def _archive(self, work):
        dispatched.append("archive")
        self.db.update_state(work["id"], "archiving")
        self.db.update_state(work["id"], "archived")
        return True

    monkeypatch.setattr(PO, "_intake_from_staging", _intake)
    monkeypatch.setattr(PO, "_run_pipeline", _pipeline)
    monkeypatch.setattr(PO, "_archive_to_nas", _archive)

    from mousereach.watcher.db import WatcherDB

    def db():
        return WatcherDB(tmp_path / "local" / "watcher.db")

    return {"tmp": tmp_path, "roots": roots, "steps": steps, "dispatched": dispatched,
            "pushed": pushed, "db": db}


def _video_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{V}.mp4").write_bytes(b"video")
    (folder / POSE).write_bytes(b"pose")
    for suffix in ("_segments.json", "_reaches.json", "_pellet_outcomes.json"):
        (folder / f"{V}{suffix}").write_text('{"kept": true}')
    return folder / f"{V}.mp4"


def _kept(folder: Path):
    for suffix in ("_segments.json", "_reaches.json", "_pellet_outcomes.json"):
        assert json.loads((folder / f"{V}{suffix}").read_text()) == {"kept": True}, suffix
    assert not (folder / f"{V}_reach_assignments.json").exists()
    assert not (folder / f"{V}_features.json").exists()


def test_a_filed_video_is_refused_and_its_files_left_alone(env):
    from mousereach.pipeline.run_all import run_all_steps
    filed = env["roots"]["ANALYZED_OUTPUT"] / "PRJ" / "C01"
    mp4 = _video_folder(filed)
    res = run_all_steps(mp4)
    _kept(filed)
    assert env["steps"] == [] and env["dispatched"] == []
    assert not res.get("done")
    assert "filed" in (res.get("refused") or "")


def test_a_video_in_a_triage_bundle_is_refused_and_the_bundle_left_alone(env):
    from mousereach.pipeline.run_all import run_all_steps
    bundle = env["roots"]["TRIAGE_REVIEW"] / V
    mp4 = _video_folder(bundle)
    res = run_all_steps(mp4)
    _kept(bundle)
    assert env["steps"] == [] and env["dispatched"] == []
    assert "review" in (res.get("refused") or "")


def test_a_stray_copy_of_a_video_held_in_deep_review_is_refused(env):
    """Refused by the video, not only by the folder browsed to."""
    from mousereach.pipeline.run_all import run_all_steps
    _video_folder(env["roots"]["DEEP_REVIEW"] / V)
    stray = env["tmp"] / "desk"
    mp4 = _video_folder(stray)
    res = run_all_steps(mp4)
    _kept(stray)
    assert env["steps"] == [] and env["dispatched"] == []
    assert "deep review" in (res.get("refused") or "")


def test_a_video_in_processing_runs_the_watchers_own_steps_and_is_filed(env):
    from mousereach.pipeline.run_all import run_all_steps
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "processing", reason="test fixture")
    res = run_all_steps(mp4)
    assert env["dispatched"] == ["pipeline", "archive"], res
    assert env["steps"] == []          # no step outside the watcher's handlers
    assert env["pushed"] == []         # nothing pushed into any database
    assert res.get("done") is True and env["db"]().get_video(V)["state"] == "archived"


def test_a_posed_video_is_taken_in_analysed_and_filed_by_the_watcher(env):
    env["db"]()                      # the machine's watcher database exists
    from mousereach.pipeline.run_all import run_all_steps
    posed = env["roots"]["DLC_STAGING"]
    mp4 = _video_folder(posed)
    res = run_all_steps(mp4)
    assert env["dispatched"] == ["intake", "pipeline", "archive"], res
    assert env["steps"] == []
    assert res.get("done") is True


def test_a_crystallized_video_is_refused(env):
    from mousereach.pipeline.run_all import run_all_steps
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "crystallized", reason="test fixture", crystallized_label="X")
    res = run_all_steps(mp4)
    _kept(proc)
    assert env["steps"] == [] and env["dispatched"] == []
    assert "crystallized" in (res.get("refused") or "")


def test_a_video_the_watcher_does_not_know_is_refused(env):
    from mousereach.pipeline.run_all import run_all_steps
    env["db"]()                      # the machine's watcher database exists
    elsewhere = env["tmp"] / "elsewhere"
    mp4 = _video_folder(elsewhere)
    res = run_all_steps(mp4)
    _kept(elsewhere)
    assert env["steps"] == [] and env["dispatched"] == []
    assert "no record" in (res.get("refused") or "")


def test_no_watcher_database_is_refused_and_none_is_created(env):
    from mousereach.pipeline.run_all import run_all_steps
    elsewhere = env["tmp"] / "elsewhere"
    mp4 = _video_folder(elsewhere)
    res = run_all_steps(mp4)
    assert env["steps"] == [] and env["dispatched"] == []
    assert "no watcher database" in (res.get("refused") or "")
    assert not (env["tmp"] / "local" / "watcher.db").exists()


def test_a_graphics_card_pc_is_told_this_tab_runs_only_on_the_processing_machine(env,
                                                                               monkeypatch):
    from mousereach.config import WatcherConfig
    from mousereach.pipeline.run_all import run_all_steps
    cfg = WatcherConfig({"mode": "dlc_pc", "also_process": True,
                         "db_path": str(env["tmp"] / "local" / "watcher.db")})
    monkeypatch.setattr(WatcherConfig, "load", classmethod(lambda cls: cfg))
    mp4 = _video_folder(env["tmp"] / "elsewhere")
    why = run_all_steps(mp4).get("refused") or ""
    assert "runs only on the processing machine" in why
    assert "does not analyse" not in why


def test_nothing_runs_while_a_watcher_runs_on_this_machine(env, monkeypatch):
    from mousereach.watcher import health
    monkeypatch.setattr(health, "watcher_running", lambda: 424242)
    from mousereach.pipeline.run_all import run_all_steps
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "processing", reason="test fixture")
    res = run_all_steps(mp4)
    _kept(proc)
    assert env["steps"] == [] and env["dispatched"] == []
    assert "already running" in (res.get("refused") or "")


def test_a_held_video_stops_where_the_watcher_holds_it(env, monkeypatch):
    """A video the gate sends to triage stops there; it is not filed."""
    from mousereach.watcher import orchestrator as orch_mod

    def _pipeline_holds(self, work):
        env["dispatched"].append("pipeline")
        self.db.update_state(work["id"], "triage")
        return True

    monkeypatch.setattr(orch_mod.ProcessingOrchestrator, "_run_pipeline", _pipeline_holds)
    from mousereach.pipeline.run_all import run_all_steps
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "processing", reason="test fixture")
    res = run_all_steps(mp4)
    assert env["dispatched"] == ["pipeline"]
    assert res.get("held") == "triage" and not res.get("done")


def test_the_tab_never_says_the_database_was_pushed_to():
    # The Run All Steps tab is retired (2026-11-19); its line per video is now
    # run_one.chosen_summary_line, shown by "Run chosen videos..." and printed
    # by mousereach-run-videos.
    from mousereach.watcher.run_one import chosen_summary_line as status_text
    for res in ({"done": True}, {"held": "triage", "hold_reason": "x"},
                {"refused": "why"}, {"error": "e"}):
        text = status_text(res)
        text.encode("ascii")
        assert "database" not in text
    assert status_text({"refused": "why"}).startswith("[NOT RUN]")


def test_a_step_that_fails_is_run_once_and_reported_as_failed(env, monkeypatch):
    """A failure during the run stops it with [FAIL]; it is not reset and run
    again (that wiped the retry count and ran the failing step three times)."""
    from mousereach.watcher import orchestrator as orch_mod

    def _pipeline_fails(self, work):
        env["dispatched"].append("pipeline")
        raise RuntimeError("a step broke")

    monkeypatch.setattr(orch_mod.ProcessingOrchestrator, "_run_pipeline", _pipeline_fails)
    from mousereach.pipeline.run_all import run_all_steps
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "processing", reason="test fixture")
    res = run_all_steps(mp4)
    assert env["dispatched"] == ["pipeline"]
    row = env["db"]().get_video(V)
    assert row["state"] == "failed"
    assert res.get("error") and not res.get("held") and not res.get("done")


def test_a_video_that_failed_before_is_reset_once_then_run(env):
    from mousereach.pipeline.run_all import run_all_steps
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "failed", reason="test fixture", error_message="earlier error")
    res = run_all_steps(mp4)
    assert env["dispatched"] == ["pipeline", "archive"], res
    assert res.get("done") is True
