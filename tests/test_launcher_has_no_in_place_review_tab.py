"""The main window no longer has the old "3 - Review Tool" tab.

WHY: that tab was the Ground Truth widget in review mode, whose save rewrote
the algorithm's segments, reaches and outcomes files in place, where the next
run erased the corrections and the watcher never read them. --reviews and
--step 2b/3b/4b used to open it; they now print a note and open only the tabs
that are always present (Review Queues among them), and a video given on the
command line goes to the shared video loader.

No window is opened: napari, the state manager and every tab are stood in.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

ALWAYS_LOADED = [
    "mousereach.watcher.control_widget",
    "mousereach.review.queue_launcher_widget",
    "mousereach.watcher.quarantine_widget",
    "mousereach.setup.setup_widget",
    "mousereach.pipeline.run_all_widget",
    "mousereach.dlc_compare.widget",
    "mousereach.fpfn_review.widget",
]


@pytest.fixture
def stood_in(monkeypatch, tmp_path):
    # launch() sets these for napari; recording them here puts each back as
    # it was when the test ends, so no later test sees them.
    for var in ("NAPARI_DISABLE_PLUGIN_AUTOLOAD", "NAPARI_DISABLE_PLUGINS",
                "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "NAPARI_SETTINGS"):
        monkeypatch.setenv(var, str(tmp_path / "napari_settings" / var))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "is_configured", staticmethod(lambda: True), raising=False)
    monkeypatch.setattr(Paths, "validate", staticmethod(lambda: []), raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path / "missing", raising=False)

    fake_napari = types.ModuleType("napari")
    fake_napari.Viewer = MagicMock(name="Viewer")
    fake_napari.run = MagicMock(name="run")
    monkeypatch.setitem(sys.modules, "napari", fake_napari)

    state = MagicMock(name="state")
    fake_state = types.ModuleType("mousereach.state")
    fake_state.MouseReachStateManager = MagicMock(return_value=state)
    fake_state.connect_widget_to_state = MagicMock()
    monkeypatch.setitem(sys.modules, "mousereach.state", fake_state)

    fake_index = types.ModuleType("mousereach.index")
    fake_index.PipelineIndex = MagicMock()
    monkeypatch.setitem(sys.modules, "mousereach.index", fake_index)

    gt_cls = MagicMock(name="GroundTruthWidget")
    fake_gt = types.ModuleType("mousereach.review.ground_truth_widget")
    fake_gt.GroundTruthWidget = gt_cls
    monkeypatch.setitem(sys.modules, "mousereach.review.ground_truth_widget", fake_gt)

    from mousereach import launcher
    step_tabs = [w[2] for w in launcher.WIDGETS]
    for name in ALWAYS_LOADED + step_tabs:
        monkeypatch.setitem(sys.modules, name, None)   # import fails -> tab skipped

    import qtpy.QtCore as QtCore
    import qtpy.QtWidgets as QtWidgets
    timer = MagicMock()
    timer.singleShot.side_effect = lambda ms, fn: fn()
    monkeypatch.setattr(QtCore, "QTimer", timer)
    monkeypatch.setattr(QtWidgets, "QScrollArea", MagicMock())
    return types.SimpleNamespace(state=state, gt_cls=gt_cls)


def _review_mode_calls(gt_cls):
    return [c for c in gt_cls.call_args_list if c.kwargs.get("review_mode")]


@pytest.mark.parametrize("steps", [["2b"], ["2b", "3b", "4b"]])
def test_review_steps_open_no_in_place_review_tab(stood_in, steps, capsys):
    from mousereach import launcher
    launcher.launch(steps=steps)
    assert _review_mode_calls(stood_in.gt_cls) == []
    assert "retired" in capsys.readouterr().out


def test_a_video_given_with_review_steps_goes_to_the_shared_loader(stood_in, tmp_path):
    from mousereach import launcher
    video = tmp_path / "20990101_ABC0904_P1.mp4"
    video.write_bytes(b"")
    launcher.launch(video_path=video, steps=["2b"])
    stood_in.state.load_video.assert_called_once()
    assert _review_mode_calls(stood_in.gt_cls) == []


def test_a_plain_start_prints_no_retirement_note(stood_in, capsys):
    """ALL_STEPS still lists the retired review steps; a plain start must not
    print the note meant for someone who asked for them."""
    from mousereach import launcher
    launcher.launch()
    assert "retired" not in capsys.readouterr().out
    assert _review_mode_calls(stood_in.gt_cls) == []
