"""Watcher Control panel: the "pause while a recording program runs" settings.

WHY these tests: a GPU node in the behaviour room must stop pipeline work
while somebody records, and the operator configures and watches that from this
panel. A list that saves wrong, a key the Save button silently drops, or a
status line that says STOPPED while a watcher runs all fail silently -- the
recording stutters, or a second watcher gets started.

No Qt display is needed: the plain helpers are tested directly, and widget
methods are called unbound on a stand-in ``self`` (the pattern the other
widget tests in this suite use). Nothing here touches the real config file:
Path.home() is pointed at tmp_path.
"""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

cw = pytest.importorskip("mousereach.watcher.control_widget")
W = cw.WatcherControlWidget


# --- parse / format -----------------------------------------------------------

def test_parse_splits_on_commas_semicolons_and_lines_but_not_spaces():
    text = "recorder.exe, Other Recorder.exe; third.exe\nfourth.exe"
    assert cw.parse_program_list(text) == [
        "recorder.exe", "Other Recorder.exe", "third.exe", "fourth.exe"]


def test_parse_empty_means_no_programs():
    assert cw.parse_program_list("") == []
    assert cw.parse_program_list(None) == []
    assert cw.parse_program_list(" ,, ; ") == []


def test_parse_drops_case_duplicates_quotes_and_folders():
    text = '"recorder.exe", RECORDER.EXE, some folder\\recorder.exe, tools/other.exe'
    assert cw.parse_program_list(text) == ["recorder.exe", "other.exe"]


def test_format_round_trips_through_parse():
    names = ["recorder.exe", "Other Recorder.exe"]
    assert cw.format_program_list(names) == "recorder.exe, Other Recorder.exe"
    assert cw.parse_program_list(cw.format_program_list(names)) == names
    assert cw.format_program_list(None) == ""
    assert cw.format_program_list([]) == ""


# --- status text --------------------------------------------------------------

def test_pause_label_text():
    assert cw.pause_label_text(None) == ""
    assert cw.pause_label_text("recorder.exe is running") == "Paused: recorder.exe is running"


def test_headline_stopped_running_and_paused():
    assert cw.status_headline(False, None, None)[0] == "Status: STOPPED"
    assert cw.status_headline(True, None, None)[0] == "Status: RUNNING"
    txt, color = cw.status_headline(True, None, "recorder.exe is running")
    assert txt == "Status: RUNNING (PAUSED)" and color == "#c80"


def test_headline_counts_a_watcher_started_outside_the_panel():
    # The bug being fixed: the panel said STOPPED while a command-line watcher
    # was working, inviting the operator to start a second one.
    txt, _ = cw.status_headline(False, 4242, None)
    assert txt.startswith("Status: RUNNING") and "outside this panel" in txt
    txt, _ = cw.status_headline(False, 4242, "recorder.exe is running")
    assert txt.startswith("Status: RUNNING (PAUSED)")
    # The panel's own watcher is not described as outside.
    assert "outside" not in cw.status_headline(True, 4242, None)[0]


def test_headline_appends_last_error():
    txt, _ = cw.status_headline(False, None, None, run_error="boom")
    assert txt.endswith("last error: boom")


def test_all_new_operator_text_is_ascii():
    for s in (cw.PAUSE_PROGRAMS_HELP, cw.PAUSE_PROGRAMS_TOOLTIP,
              cw.PAUSE_REASON_TOOLTIP, cw._HAND_PAUSE_REASON, cw.SAVE_APPLIES_TEXT,
              cw.watching_label_text(["recorder.exe"], 120),
              cw.recording_check_text(["recorder.exe"], []),
              cw.run_once_refusal_text(cw._HAND_PAUSE_REASON)):
        s.encode("ascii")


def test_help_text_does_not_promise_more_than_the_watcher_does():
    # A crop already running is NOT stopped; an operator told "no pipeline
    # work" would start filming on top of it.
    assert "crop" in cw.PAUSE_PROGRAMS_HELP and "finishes first" in cw.PAUSE_PROGRAMS_HELP


def test_watching_label_shows_the_names_being_watched_for():
    assert cw.watching_label_text([], 120) == ""
    assert cw.watching_label_text(None, None) == ""
    txt = cw.watching_label_text(["recorder.exe"], 90)
    assert "recorder.exe" in txt and "90 s" in txt


def test_recording_check_text_flags_a_name_that_matches_nothing():
    assert "never pauses" in cw.recording_check_text([], None)
    assert "does not match" in cw.recording_check_text(["recoder.exe"], [])
    assert "Running right now: recorder.exe" in cw.recording_check_text(
        ["recorder.exe"], ["recorder.exe"])
    assert "stays paused" in cw.recording_check_text(["recorder.exe"], None, "boom")


def _once_self(reason):
    # _mode_select is None: if Run Once went on to start, it would fail on it.
    return SimpleNamespace(
        _is_running=lambda: False, _refuse_if_external_watcher=lambda: False,
        _current_pause_reason=lambda cfg: reason, _refresh=lambda: None,
        _mode_select=None)


def test_run_once_refuses_while_paused_by_hand_or_by_a_recording(monkeypatch):
    # The watcher's run_once refuses both; without this check the panel said
    # "Run Once started" and then nothing happened, with no word why.
    from mousereach import config as mr_config
    monkeypatch.setattr(mr_config.WatcherConfig, "load",
                        classmethod(lambda cls: SimpleNamespace()))
    shown = []
    monkeypatch.setattr(cw, "show_info", shown.append)
    W._run_once(_once_self(cw._HAND_PAUSE_REASON))
    assert "paused by hand" in shown[-1] and "Press Resume first" in shown[-1]
    W._run_once(_once_self("recorder.exe is running"))
    assert "recorder.exe is running" in shown[-1]
    assert "Close the recording program" in shown[-1]
    assert not any(m.startswith("Run Once started") for m in shown)


def test_resume_says_the_watcher_stays_paused_while_recording(tmp_path, monkeypatch):
    from mousereach import config as mr_config
    monkeypatch.setattr(mr_config.WatcherConfig, "load",
                        classmethod(lambda cls: SimpleNamespace()))
    shown = []
    monkeypatch.setattr(cw, "show_info", shown.append)
    monkeypatch.setattr(cw, "show_error", lambda m: shown.append("ERROR " + m))
    flag = tmp_path / "watcher_paused.flag"
    flag.write_text("x", encoding="utf-8")
    s = SimpleNamespace(_pause_file=lambda: flag, _refresh=lambda: None,
                        _recording_reason=lambda cfg: "recorder.exe is running")
    W._toggle_pause(s)
    assert not flag.exists()
    assert "stays paused" in shown[-1] and "recorder.exe is running" in shown[-1]

    flag.write_text("x", encoding="utf-8")
    s._recording_reason = lambda cfg: None
    W._toggle_pause(s)
    assert shown[-1] == "Watcher resumed."


# --- form <-> config ----------------------------------------------------------

class _Field:
    """Stand-in for any Qt input: remembers what was set, returns it back."""

    def __init__(self, value=None):
        self.value_ = value

    # setters
    def setText(self, v): self.value_ = v
    def setValue(self, v): self.value_ = v
    def setChecked(self, v): self.value_ = v
    def setCurrentText(self, v): self.value_ = v
    def setCurrentIndex(self, v): self.value_ = v
    def findData(self, v): return -1

    # getters
    def text(self): return self.value_ or ""
    def value(self): return self.value_
    def isChecked(self): return bool(self.value_)
    def currentText(self): return self.value_


def _form_self(programs="", grace=120):
    s = SimpleNamespace(
        _f_enabled=_Field(True), _f_mode=_Field("dlc_pc"), _f_poll=_Field(30),
        _f_stability=_Field(60), _f_retries=_Field(3), _f_maxpending=_Field(200),
        _f_gpu=_Field(0), _f_autoarchive=_Field(False), _f_alsoprocess=_Field(False),
        _f_dlccfg=_Field(""), _f_quarantine=_Field(""), _f_logdir=_Field(""),
        _f_dbpath=_Field(""), _f_staging=_Field(""), _mode_select=_Field(),
        _f_pause_programs=_Field(programs), _f_pause_grace=_Field(grace),
        _FORM_MANAGED_KEYS=W._FORM_MANAGED_KEYS,
    )
    s._form_to_dict = lambda: W._form_to_dict(s)
    return s


def test_form_writes_program_list_only_when_non_empty():
    d = W._form_to_dict(_form_self(programs="recorder.exe, b.exe", grace=45))
    assert d["pause_while_running"] == ["recorder.exe", "b.exe"]
    assert d["pause_resume_grace_seconds"] == 45

    d = W._form_to_dict(_form_self(programs="  "))
    assert "pause_while_running" not in d
    assert d["pause_resume_grace_seconds"] == 120


def test_both_new_keys_are_form_managed():
    # Managed = the form may CLEAR it. Without this, blanking the list would
    # keep the old list in the file and the PC would go on pausing.
    assert {"pause_while_running", "pause_resume_grace_seconds"} <= W._FORM_MANAGED_KEYS


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    shown = []
    monkeypatch.setattr(cw, "show_info", shown.append)
    monkeypatch.setattr(cw, "show_error", lambda m: shown.append("ERROR " + m))
    # Save checks the listed programs against the real process list; a fake
    # one here, so no test depends on what this machine is running.
    checked = []
    monkeypatch.setattr(cw, "check_programs_now",
                        lambda names: checked.append(list(names)) or ([], None))
    cfg = tmp_path / ".mousereach" / "config.json"
    cfg.parent.mkdir()
    return SimpleNamespace(cfg=cfg, shown=shown, checked=checked)


def test_save_merges_in_place_and_keeps_every_other_key(home):
    home.cfg.write_text(json.dumps({
        "processing_root": "some/processing/root",
        "backup": {"enabled": True},
        "watcher": {"work_priority": {"projects": ["PROJECT_A"]},
                    "repose_batch": 2, "mode": "processing_server"},
    }), encoding="utf-8")

    W._save_config(_form_self(programs="recorder.exe", grace=90))

    saved = json.loads(home.cfg.read_text(encoding="utf-8"))
    assert saved["processing_root"] == "some/processing/root"
    assert saved["backup"] == {"enabled": True}
    w = saved["watcher"]
    assert w["work_priority"] == {"projects": ["PROJECT_A"]}
    assert w["repose_batch"] == 2
    assert w["mode"] == "dlc_pc"                       # form-managed: form wins
    assert w["pause_while_running"] == ["recorder.exe"]
    assert w["pause_resume_grace_seconds"] == 90
    assert not any(m.startswith("ERROR") for m in home.shown)
    # Save says the list reaches a running watcher, and warns that no listed
    # program is open (the sign of a mistyped name).
    assert home.checked == [["recorder.exe"]]
    assert cw.SAVE_APPLIES_TEXT in home.shown[-1]
    assert "the name does not match" in home.shown[-1]


def test_blanking_the_list_on_save_turns_the_pause_off(home):
    home.cfg.write_text(json.dumps({
        "watcher": {"pause_while_running": ["recorder.exe"],
                    "pause_resume_grace_seconds": 30, "repose_batch": 2},
    }), encoding="utf-8")

    W._save_config(_form_self(programs=""))

    w = json.loads(home.cfg.read_text(encoding="utf-8"))["watcher"]
    assert "pause_while_running" not in w
    assert w["repose_batch"] == 2


def test_load_fills_the_new_fields(monkeypatch):
    from mousereach import config as mr_config

    class _Cfg:
        def __init__(self, d): self.d = d
        def to_dict(self): return dict(self.d)

    s = _form_self()
    monkeypatch.setattr(mr_config.WatcherConfig, "load", classmethod(
        lambda cls: _Cfg({"pause_while_running": ["recorder.exe", "b.exe"],
                          "pause_resume_grace_seconds": 30})))
    W._load_config_into_form(s)
    assert s._f_pause_programs.value_ == "recorder.exe, b.exe"
    assert s._f_pause_grace.value_ == 30

    # Nothing configured -> empty list, shipped grace.
    monkeypatch.setattr(mr_config.WatcherConfig, "load",
                        classmethod(lambda cls: _Cfg({})))
    W._load_config_into_form(s)
    assert s._f_pause_programs.value_ == ""
    assert s._f_pause_grace.value_ == cw.DEFAULT_RESUME_GRACE_SECONDS


# --- pause reason and the guard's lifetime ------------------------------------

class _FakeGuard:
    made = []

    def __init__(self, names, grace_seconds=120, **kw):
        self.names, self.grace = list(names), grace_seconds
        self.next_reason = None
        _FakeGuard.made.append(self)

    def reason(self):
        return self.next_reason


@pytest.fixture
def fake_guard_module(monkeypatch):
    _FakeGuard.made = []
    calls = []
    mod = types.ModuleType("mousereach.watcher.recording_guard")
    mod.RecordingGuard = _FakeGuard

    def pause_reason(processing_root=None, config=None, guard=None):
        calls.append((processing_root, config, guard))
        if processing_root is not None and (Path(processing_root) / "watcher_paused.flag").exists():
            return "paused by hand (watcher_paused.flag)"
        return guard.reason() if guard is not None else None

    mod.pause_reason = pause_reason
    monkeypatch.setitem(sys.modules, "mousereach.watcher.recording_guard", mod)
    return calls


def _reason_self(tmp_path):
    s = SimpleNamespace(_guard=None, _guard_key=None,
                        _pause_file=lambda: tmp_path / "watcher_paused.flag")
    s._recording_guard = lambda cfg: W._recording_guard(s, cfg)
    return s


def _cfg(names, grace=120):
    return SimpleNamespace(pause_while_running=names, pause_resume_grace_seconds=grace)


def test_no_programs_listed_means_no_guard_and_no_reason(tmp_path, fake_guard_module):
    s = _reason_self(tmp_path)
    assert W._current_pause_reason(s, _cfg([])) is None
    assert _FakeGuard.made == []


def test_guard_is_kept_across_refreshes_and_rebuilt_on_change(tmp_path, fake_guard_module):
    # Kept: the guard remembers when the program closed (the grace countdown).
    s = _reason_self(tmp_path)
    W._current_pause_reason(s, _cfg(["recorder.exe"]))
    W._current_pause_reason(s, _cfg(["RECORDER.EXE"]))   # same list, other capitals
    assert len(_FakeGuard.made) == 1
    W._current_pause_reason(s, _cfg(["recorder.exe"], grace=30))
    assert len(_FakeGuard.made) == 2 and _FakeGuard.made[-1].grace == 30
    W._current_pause_reason(s, _cfg([]))
    assert s._guard is None


def test_reason_shows_the_recording_program_and_the_hand_flag(tmp_path, fake_guard_module):
    s = _reason_self(tmp_path)
    cfg = _cfg(["recorder.exe"])
    W._current_pause_reason(s, cfg)
    s._guard.next_reason = "recorder.exe is running"
    reason = W._current_pause_reason(s, cfg)
    assert cw.pause_label_text(reason) == "Paused: recorder.exe is running"
    # The processing root handed to pause_reason is the flag's folder.
    assert fake_guard_module[-1][0] == tmp_path

    (tmp_path / "watcher_paused.flag").write_text("x", encoding="utf-8")
    assert W._current_pause_reason(s, cfg) == "paused by hand (watcher_paused.flag)"


def test_missing_guard_module_falls_back_safely(tmp_path, monkeypatch):
    # Programs listed but the check cannot run: say so (fail safe, recording
    # wins). Nothing listed: behave exactly as before this feature.
    monkeypatch.setitem(sys.modules, "mousereach.watcher.recording_guard", None)
    s = _reason_self(tmp_path)
    reason = W._current_pause_reason(s, _cfg(["recorder.exe"]))
    assert reason and reason.startswith("cannot check for recording programs")
    assert W._recording_reason(s, _cfg(["recorder.exe"])).startswith(
        "cannot check for recording programs")
    assert W._current_pause_reason(s, _cfg([])) is None
    (tmp_path / "watcher_paused.flag").write_text("x", encoding="utf-8")
    assert W._current_pause_reason(s, _cfg([])) == "paused by hand (watcher_paused.flag)"


def test_fallback_text_matches_the_real_guard_module(tmp_path):
    # The panel's own fallback must say exactly what the watcher says, or an
    # operator sees two different words for one pause.
    rg = pytest.importorskip("mousereach.watcher.recording_guard")
    assert cw._HAND_PAUSE_REASON == rg.HAND_PAUSE_REASON
    s = _reason_self(tmp_path)
    (tmp_path / "watcher_paused.flag").write_text("x", encoding="utf-8")
    assert W._current_pause_reason(s, _cfg(["recorder.exe"])) == rg.HAND_PAUSE_REASON
    assert isinstance(s._guard, rg.RecordingGuard)


# --- watcher started outside the panel ----------------------------------------

def _ext_self():
    return SimpleNamespace(_ext_pid=None, _ext_checked_at=float("-inf"))


def test_external_watcher_is_cached_and_never_this_process(monkeypatch):
    import os
    from mousereach.watcher import health
    calls = []
    monkeypatch.setattr(health, "watcher_running", lambda: calls.append(1) or 4242)
    s = _ext_self()
    assert W._external_watcher_pid(s) == 4242
    assert W._external_watcher_pid(s) == 4242
    assert len(calls) == 1                                   # cached

    monkeypatch.setattr(health, "watcher_running", lambda: os.getpid())
    s = _ext_self()
    assert W._external_watcher_pid(s) is None


def test_start_refuses_when_a_watcher_runs_outside_the_panel(monkeypatch):
    from mousereach.watcher import health
    shown, refreshed = [], []
    monkeypatch.setattr(cw, "show_info", shown.append)
    monkeypatch.setattr(health, "watcher_running", lambda: 4242)
    s = _ext_self()
    s._external_watcher_pid = lambda: W._external_watcher_pid(s)
    s._refresh = lambda: refreshed.append(1)
    s._is_running = lambda: False
    s._refuse_if_external_watcher = lambda: W._refuse_if_external_watcher(s)
    W._start(s)   # would raise on the missing attributes if it went on to start
    assert shown and "already running" in shown[0]
    assert refreshed
