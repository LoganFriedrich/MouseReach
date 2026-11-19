"""The single-machine role ("standalone"): one PC cuts, poses, analyses, takes
reviewed videos back and files (watcher/standalone.py).

WHY: a lab with one PC could not run the whole pipeline. A machine ran exactly
one watcher program -- pose-only (dlc_pc) or processing (processing_server) --
and the pose-only program with also_process on has no review return, so every
video a person reviewed stayed in its queue for good. A fresh install also
defaulted to the pose-only role without asking. These tests pin:

  * the role builds BOTH existing programs (no third copy of any step), with
    separate databases, the posing half pose-only and handing on;
  * one pass runs the posing half, then the processing half;
  * the automatic watcher runs both, on one shared stop;
  * the wizard and the Setup tab's rules ask the role and write what it needs;
  * every role check that meant "the machine that analyses and files" accepts
    the single machine;
  * the lab-profile template has a single-machine profile; sqlalchemy is a
    core dependency.
"""
import json
import threading
from pathlib import Path

import pytest


class _FakeHalf:
    def __init__(self, config, db, log, name):
        self.config, self.db, self.log, self.name = config, db, log, name
        self.working_dir = None

    def run_once(self):
        self.log.append(("run_once", self.name))
        if getattr(self.config, "_fail", False):
            raise RuntimeError("boom")

    def run(self, evt):
        self.log.append(("run", self.name, evt))

    def shutdown(self):
        self.log.append(("shutdown", self.name))


@pytest.fixture
def fake_halves(tmp_path, monkeypatch):
    from mousereach.watcher import orchestrator as orch_mod
    import mousereach.config as mr_config
    log = []
    monkeypatch.setattr(orch_mod, "DLCOrchestrator",
                        lambda c, d: _FakeHalf(c, d, log, "posing"))
    monkeypatch.setattr(orch_mod, "ProcessingOrchestrator",
                        lambda c, d: _FakeHalf(c, d, log, "processing"))
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: tmp_path)
    return log


def _cfg(**kw):
    from mousereach.config import WatcherConfig
    d = {"mode": "standalone", "db_path": "x.db", "dlc_config_path": "m/config.yaml"}
    d.update(kw)
    return WatcherConfig(d)


def test_the_single_machine_role_builds_both_existing_programs(fake_halves, tmp_path):
    from mousereach.watcher.standalone import make_orchestrator, StandaloneOrchestrator
    main_db = object()
    cfg = _cfg(also_process=True)
    orch = make_orchestrator(cfg, main_db)
    assert isinstance(orch, StandaloneOrchestrator)
    # posing half: a pose-only GPU node that hands its poses on, own database
    assert orch.posing.config.mode == "dlc_pc"
    assert orch.posing.config.also_process is False
    assert Path(orch.posing.db.db_path) == tmp_path / "watcher_posing.db"
    # processing half: the processing server's program on THE database
    assert orch.processing.config.mode == "processing_server"
    assert orch.processing.db is main_db
    # the caller's settings are not changed; the DLC model reaches the posing half
    assert cfg.mode == "standalone" and cfg.also_process is True
    assert str(orch.posing.config.dlc_config_path).endswith("config.yaml")
    # separate temporary folders, so one half stopping never clears the other's
    assert orch.processing.working_dir == tmp_path / "watcher_working_processing"


def test_other_roles_still_get_their_own_program(fake_halves):
    from mousereach.watcher.standalone import make_orchestrator
    assert make_orchestrator(_cfg(mode="processing_server"), None).name == "processing"
    assert make_orchestrator(_cfg(mode="dlc_pc"), None).name == "posing"


def test_one_pass_poses_first_then_analyses(fake_halves):
    from mousereach.watcher.standalone import make_orchestrator
    make_orchestrator(_cfg(), object()).run_once()
    assert [e for e in fake_halves if e[0] == "run_once"] == [
        ("run_once", "posing"), ("run_once", "processing")]


def test_a_failing_posing_half_does_not_stop_the_analysis(fake_halves):
    from mousereach.watcher.standalone import make_orchestrator
    orch = make_orchestrator(_cfg(), object())
    orch.posing.config._fail = True
    orch.run_once()
    assert ("run_once", "processing") in fake_halves


def test_the_automatic_watcher_runs_both_halves_on_one_stop(fake_halves):
    from mousereach.watcher.standalone import make_orchestrator
    evt = threading.Event()
    make_orchestrator(_cfg(), object()).run(evt)
    runs = [e for e in fake_halves if e[0] == "run"]
    assert {e[1] for e in runs} == {"posing", "processing"}
    assert all(e[2] is evt for e in runs)


def test_role_helpers():
    from mousereach.watcher.standalone import analyses_and_files, poses, KNOWN_ROLES
    assert analyses_and_files("standalone") and analyses_and_files("processing_server")
    assert not analyses_and_files("dlc_pc") and not analyses_and_files(None)
    assert poses("standalone") and poses("dlc_pc") and not poses("processing_server")
    assert set(KNOWN_ROLES) == {"dlc_pc", "processing_server", "standalone"}


def test_the_dry_run_previews_both_halves(tmp_path, monkeypatch):
    import mousereach.config as mr_config
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: tmp_path)
    from mousereach.watcher.preview import preview
    text = "\n".join(preview(_cfg(), tmp_path / "watcher.db"))
    assert "posing half" in text and "processing half" in text
    assert not (tmp_path / "watcher.db").exists()
    assert not (tmp_path / "watcher_posing.db").exists()


# ---------------------------------------------------------------- setup

def test_a_new_lab_is_asked_and_defaults_to_the_single_machine():
    from mousereach.setup.wizard import ask_role
    assert ask_role({}, {}, ask=lambda _p: "") == "standalone"


def test_a_known_lab_machine_keeps_its_role_by_default():
    from mousereach.setup.wizard import ask_role
    answers = iter(["", ""])
    assert ask_role({"mode": "processing_server"}, {}, ask=lambda _p: next(answers)) \
        == "processing_server"


def test_no_then_a_role_letter_picks_that_role():
    from mousereach.setup.wizard import ask_role
    answers = iter(["n", "p"])
    assert ask_role({}, {}, ask=lambda _p: next(answers)) == "processing_server"
    answers = iter(["n", "g"])
    assert ask_role({}, {}, ask=lambda _p: next(answers)) == "dlc_pc"


def test_role_settings_name_the_database_and_switch_off_the_old_local_analysis(tmp_path):
    from mousereach.setup.wizard import role_settings
    s = role_settings("standalone", tmp_path, {}, {})
    assert s == {"mode": "standalone", "also_process": False, "enabled": True,
                 "db_path": str(tmp_path / "watcher.db")}
    # a database already named is never replaced
    assert "db_path" not in role_settings("standalone", tmp_path, {"db_path": "a.db"}, {})
    assert role_settings("dlc_pc", tmp_path, {}, {}) == {"mode": "dlc_pc"}


def test_the_pipeline_folder_is_built_with_its_versions_file(tmp_path):
    from mousereach.setup.wizard import make_pipeline_folder
    from mousereach.pipeline.pipe_structure import TARGET_DIRS
    root = tmp_path / "data"
    make_pipeline_folder(root)
    for rel in TARGET_DIRS:
        assert (root / rel).is_dir(), rel
    v = root / "pipeline_versions.json"
    assert json.loads(v.read_text())["versions"]
    # an existing declaration is never replaced
    v.write_text('{"mine": 1}')
    make_pipeline_folder(root)
    assert json.loads(v.read_text()) == {"mine": 1}


def test_the_processing_root_may_not_be_the_pipeline_folder(tmp_path):
    from mousereach.setup.wizard import same_folder
    assert same_folder(tmp_path, tmp_path)
    assert same_folder(tmp_path / "a", tmp_path / "b") is None


def test_the_profile_template_has_a_single_machine_profile():
    import mousereach.setup.wizard as wz
    data = json.loads(Path(wz.LAB_PROFILES_EXAMPLE).read_text(encoding="utf-8"))
    single = [p for p in data["profiles"]
              if (p.get("defaults", {}).get("watcher") or {}).get("mode") == "standalone"]
    assert single and single[0]["defaults"]["watcher"]["also_process"] is False


def test_sqlalchemy_is_a_core_dependency():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    core = text.split("[project.optional-dependencies]")[0]
    assert '"sqlalchemy' in core


# ---------------------------------------------------- role checks elsewhere

def test_publication_locks_are_set_on_a_single_machine():
    from mousereach.pipeline.versions import publication_lock_gate
    ok, note = publication_lock_gate(_cfg())
    assert ok and note is None


def test_the_run_pipeline_tab_explains_the_single_machine():
    pytest.importorskip("qtpy")
    from mousereach.pipeline.batch_widget import role_explanation
    t = role_explanation("standalone")
    assert "only MouseReach machine" in t and "files" in t


def test_run_all_steps_runs_on_a_single_machine(tmp_path, monkeypatch):
    """Run All Steps used to refuse every role but processing_server, so on a
    single machine a chosen video could not be run by hand at all."""
    from mousereach.config import WatcherConfig
    from mousereach.watcher import run_one
    cfg = _cfg(db_path=str(tmp_path / "nowhere" / "watcher.db"))
    monkeypatch.setattr(WatcherConfig, "load", classmethod(lambda cls: cfg))
    from mousereach.watcher import singleton, health
    monkeypatch.setattr(singleton, "acquire_watcher_lock", lambda *a, **k: object())
    monkeypatch.setattr(singleton, "release_watcher_lock", lambda h: None)
    monkeypatch.setattr(singleton, "stop_flag_left", lambda p: None)
    monkeypatch.setattr(health, "watcher_running", lambda: None)
    seen = {}

    def _drive(video_id, video, cfg, result, _p, _refuse):
        seen["mode"] = cfg.mode
        return result
    monkeypatch.setattr(run_one, "_drive", _drive)
    # Since the single machine poses a chosen video first (run_one.
    # _pose_on_this_machine), that step is stood in here; it is pinned in
    # test_run_chosen_videos.
    monkeypatch.setattr(run_one, "_pose_on_this_machine",
                        lambda vid, cfg, p: seen.setdefault("posed_first", vid) and None)
    res = run_one.run_one_video(tmp_path / "20200101_ABC9901_P1.mp4")
    assert not res.get("refused"), res
    assert seen["mode"] == "processing_server"     # the processing half's program
    assert seen["posed_first"] == "20200101_ABC9901_P1"
