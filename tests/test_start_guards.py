"""The checks every watcher start passes: one watcher per machine, the
machine's own role, and no stop request left behind (watcher/singleton.py).

WHY these tests:
- The Watcher Control panel started a watcher of whichever role its "Run as"
  drop-down showed, and took no one-watcher-per-machine lock, so it could run
  a second watcher, or one of the other role, on this machine's database.
- A stop flag left behind made every later start exit at once with one INFO
  line (exit code 0, so a scheduled task showed success) while nothing ran.

Nothing here takes the real machine-wide lock: every test replaces it, so a
watcher running on the test machine is never disturbed. Everything else lives
in tmp_path.
"""
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.config as mr_config


def _no_real_lock(monkeypatch, result="LOCK"):
    """Replace the machine-wide lock: the new shared helper, and the raw
    Windows call older code made, so no test ever takes the real one."""
    released = []
    try:
        from mousereach.watcher import singleton
        monkeypatch.setattr(singleton, "acquire_watcher_lock", lambda *a, **k: result)
        monkeypatch.setattr(singleton, "release_watcher_lock",
                            lambda h: released.append(h) if h else None)
    except ImportError:
        pass
    import ctypes

    class _K32:
        def CreateMutexW(self, *a):
            return 1

        def GetLastError(self):
            return 183 if result is None else 0

    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=_K32()), raising=False)
    return released


def _flag(root, name="watcher_stop.flag"):
    f = Path(root) / name
    f.write_text("stop requested\n", encoding="utf-8")
    return f


# ------------------------------------------------------------ command line

def _watch_env(tmp_path, monkeypatch):
    """Enough configuration for main_watch to reach its start checks; the
    database is never reached (it raises if it is)."""
    from mousereach.watcher import cli
    import mousereach.watcher.db as dbmod
    from mousereach.pipeline import pipe_structure
    root = tmp_path / "node"
    for d in ("node", "share", "share/Processing/Posed", "share/Processing"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(
        mode="processing_server", poll_interval_seconds=30, stability_wait_seconds=5,
        max_local_pending=2, auto_archive_approved=True, dlc_config_path=None,
        dlc_gpu_device=0, db_path=str(tmp_path / "w.db"),
        get_log_dir=lambda: tmp_path / "logs",
        get_quarantine_dir=lambda: tmp_path / "q")
    monkeypatch.setattr(mr_config.WatcherConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: tmp_path / "log.txt")
    monkeypatch.setattr(cli, "check_nas_root", lambda *a, **k: None)
    monkeypatch.setattr(pipe_structure, "retired_folders_present", lambda *a, **k: [])
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: root)
    for name, val in (("PROCESSING_ROOT", root), ("NAS_ROOT", tmp_path / "share"),
                      ("NAS_DRIVE", tmp_path / "share"), ("NAS_ROOT_ORIGIN", "config"),
                      ("DLC_STAGING", tmp_path / "share/Processing/Posed"),
                      ("PROCESSING", tmp_path / "share/Processing")):
        monkeypatch.setattr(mr_config.Paths, name, val, raising=False)

    def no_db(*a, **k):
        raise RuntimeError("the watcher went on past its start checks")
    monkeypatch.setattr(dbmod, "WatcherDB", no_db)
    monkeypatch.setattr(sys, "argv", ["mousereach-watch"])
    return cli, root


def test_the_watcher_refuses_to_start_on_a_stop_flag_left_behind(tmp_path, monkeypatch, capsys):
    cli, root = _watch_env(tmp_path, monkeypatch)
    _no_real_lock(monkeypatch)
    flag = _flag(root)
    with pytest.raises(SystemExit) as e:
        cli.main_watch()
    assert e.value.code == 3                         # not 0: a task shows it failed
    err = capsys.readouterr().err
    assert str(flag) in err and "stop request" in err
    err.encode("ascii")
    assert flag.exists()                             # never deleted


def test_without_a_stop_flag_the_watcher_goes_on_past_the_checks(tmp_path, monkeypatch):
    cli, _root = _watch_env(tmp_path, monkeypatch)
    _no_real_lock(monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli.main_watch()
    assert e.value.code == 1                         # reached the (refused) database


# ------------------------------------------------------------ the panel

cw = pytest.importorskip("mousereach.watcher.control_widget")
W = cw.WatcherControlWidget


class _Select:
    def __init__(self, value):
        self.value = value

    def currentData(self):
        return self.value


class _Panel:
    """A stand-in for the panel: its real Start / Run Once code, no Qt."""
    _start = W._start
    _run_once = W._run_once
    _run_orchestrator = W._run_orchestrator
    _take_start_guards = getattr(W, "_take_start_guards", None)

    def __init__(self, run_as):
        self._mode_select = _Select(run_as)
        self._thread = None
        self._orchestrator = None
        self._shutdown_event = None
        self._run_error = None

    def _is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def _refuse_if_external_watcher(self):
        return False

    def _refresh(self):
        pass

    def _current_pause_reason(self, cfg):
        return None


@pytest.fixture
def panel_env(tmp_path, monkeypatch):
    """The configured role is processing_server; any watcher the panel
    builds is recorded (and returns at once) instead of running."""
    import mousereach.watcher.db as dbmod
    import mousereach.watcher.orchestrator as orch
    root = tmp_path / "node"
    root.mkdir()
    cfg = SimpleNamespace(mode="processing_server", db_path=str(tmp_path / "w.db"))
    monkeypatch.setattr(mr_config.WatcherConfig, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: root)
    monkeypatch.setattr(dbmod, "WatcherDB", lambda p: SimpleNamespace(path=p))
    built = []

    def fake(role):
        class _O:
            def __init__(self, c, db):
                built.append(role)

            def run(self, evt):
                return None

            def run_once(self):
                return None
        return _O
    monkeypatch.setattr(orch, "ProcessingOrchestrator", fake("processing_server"))
    monkeypatch.setattr(orch, "DLCOrchestrator", fake("dlc_pc"))
    shown = []
    monkeypatch.setattr(cw, "show_info", shown.append)
    monkeypatch.setattr(cw, "show_error", lambda m: shown.append("ERROR " + m))
    return SimpleNamespace(root=root, built=built, shown=shown)


def _join(panel):
    if panel._thread is not None:
        panel._thread.join(5)


@pytest.mark.parametrize("button", ["_start", "_run_once"])
def test_the_panel_never_runs_a_watcher_of_the_other_role(panel_env, monkeypatch, button):
    _no_real_lock(monkeypatch)
    p = _Panel(run_as="dlc_pc")
    getattr(p, button)()
    _join(p)
    assert panel_env.built == []
    assert "Not started" in panel_env.shown[-1] and "processing_server" in panel_env.shown[-1]


@pytest.mark.parametrize("button", ["_start", "_run_once"])
def test_the_panel_refuses_while_another_watcher_holds_the_lock(panel_env, monkeypatch, button):
    _no_real_lock(monkeypatch, result=None)          # the lock is held elsewhere
    p = _Panel(run_as="processing_server")
    getattr(p, button)()
    _join(p)
    assert panel_env.built == []
    assert "already running" in panel_env.shown[-1]


@pytest.mark.parametrize("button", ["_start", "_run_once"])
def test_the_panel_refuses_on_a_stop_flag_left_behind(panel_env, monkeypatch, button):
    _no_real_lock(monkeypatch)
    flag = _flag(panel_env.root)
    p = _Panel(run_as="processing_server")
    getattr(p, button)()
    _join(p)
    assert panel_env.built == []
    assert str(flag) in panel_env.shown[-1]
    assert flag.exists()


@pytest.mark.parametrize("button", ["_start", "_run_once"])
def test_the_panel_holds_the_lock_while_its_watcher_runs_and_then_gives_it_back(
        panel_env, monkeypatch, button):
    released = _no_real_lock(monkeypatch)
    p = _Panel(run_as="processing_server")
    getattr(p, button)()
    _join(p)
    assert panel_env.built == ["processing_server"]
    assert released == ["LOCK"]


class _Choice:
    """A stand-in for a Qt drop-down: items carry data, one is current."""
    def __init__(self, items, current):
        self.items = list(items)
        self.index = self.items.index(current)

    def findData(self, value):
        return self.items.index(value) if value in self.items else -1

    def setCurrentIndex(self, i):
        self.index = i

    def currentData(self):
        return self.items[self.index]

    def currentText(self):
        return self.items[self.index]


def test_after_changing_mode_and_saving_the_panel_starts_in_the_new_role(
        tmp_path, monkeypatch):
    """The refusal tells the operator: change mode in the settings, press
    Save, start again. That must be enough -- Save also moves 'Run as' to
    the saved role, so the next start is not refused a second time."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    cfg_file = home / ".mousereach" / "config.json"
    cfg_file.parent.mkdir()
    cfg_file.write_text('{"watcher": {"mode": "dlc_pc"}}', encoding="utf-8")
    root = tmp_path / "node"
    root.mkdir()
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: root)
    _no_real_lock(monkeypatch)
    shown = []
    monkeypatch.setattr(cw, "show_info", shown.append)
    monkeypatch.setattr(cw, "show_error", lambda m: shown.append("ERROR " + m))

    roles = ["dlc_pc", "processing_server"]

    class _SavingPanel:
        _save_config = W._save_config
        _take_start_guards = W._take_start_guards
        _FORM_MANAGED_KEYS = W._FORM_MANAGED_KEYS

        def __init__(self):
            self._mode_select = _Choice(roles, "dlc_pc")
            self._f_mode = _Choice(roles, "dlc_pc")

        def _form_to_dict(self):
            return {"mode": self._f_mode.currentText()}

    p = _SavingPanel()
    p._f_mode.setCurrentIndex(roles.index("processing_server"))   # the operator's change
    p._save_config()
    assert not any(m.startswith("ERROR") for m in shown), shown
    cfg = mr_config.WatcherConfig.load()
    assert cfg.mode == "processing_server"
    ok, handle = p._take_start_guards(cfg)
    assert ok, shown[-1]
    assert p._mode_select.currentData() == "processing_server"


# ------------------------------------------------------------ the lock itself

@pytest.mark.skipif(sys.platform != "win32", reason="the lock is a Windows mutex")
def test_a_second_taker_of_the_same_lock_is_refused_until_the_first_lets_go():
    """On a private name, so the real watcher lock is never touched."""
    from mousereach.watcher import singleton
    name = "Local\\mousereach-test-lock-%d" % id(object())
    first = singleton.acquire_watcher_lock(name)
    assert first
    try:
        assert singleton.acquire_watcher_lock(name) is None
    finally:
        singleton.release_watcher_lock(first)
    again = singleton.acquire_watcher_lock(name)
    assert again
    singleton.release_watcher_lock(again)


# ------------------------------------------------------------ the CPU pool

def test_the_cpu_pool_refuses_to_start_on_a_stop_flag_left_behind(tmp_path, monkeypatch, capsys):
    import mousereach.watcher.cpu_pose_pool as pool_mod
    root = tmp_path / "node"
    root.mkdir()
    monkeypatch.setattr(pool_mod.Paths, "PROCESSING_ROOT", root, raising=False)
    flag = _flag(root, pool_mod.STOP_FLAG)
    p = object.__new__(pool_mod.CpuPosePool)
    p.workers, p.threads, p.dlc_config, p.holder = 1, 1, tmp_path / "c.yaml", "TEST-cpu"
    p.active, p.backoff, p.done_times, p.stopping = {}, {}, [], False
    recovered = []
    monkeypatch.setattr(p, "recover", lambda: recovered.append(True), raising=False)
    monkeypatch.setattr(p, "_status", lambda: None, raising=False)
    monkeypatch.setattr(p, "_start_one",
                        lambda: pytest.fail("a pose was started"), raising=False)
    monkeypatch.setattr(pool_mod.time, "sleep",
                        lambda s: pytest.fail("the pool went on running"))
    assert p.run() == 3
    assert recovered                                  # a previous run's leftovers handled
    assert str(flag) in capsys.readouterr().err
    assert flag.exists()


# ------------------------------------------------------------ the health line

def test_health_names_a_waiting_stop_flag_when_the_watcher_is_not_running(tmp_path, monkeypatch):
    from mousereach.watcher import health
    root = tmp_path / "node"
    root.mkdir()
    monkeypatch.setattr(mr_config, "require_processing_root", lambda: root)
    monkeypatch.setattr(health, "watcher_running", lambda: None)
    flag = _flag(root)
    lines = health.health_report()
    assert any(str(flag) in l and "will not start" in l for l in lines)
    flag.unlink()
    assert not any("stop request" in l for l in health.health_report())
