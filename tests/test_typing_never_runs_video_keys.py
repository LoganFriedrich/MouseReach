"""Typing in a review panel never runs the video's keys, and the video's keys
are the review tool's.

Logan, 2026-10-24: "when I type numbers into text fields in the widgets ...
sometimes keybinds go off". A number box keeps what is typed into it, but the
keyboard was often on the control clicked last (the manual-entry choice, a
button, the outcome list), and napari's dock handed the digits to the video's
shortcuts -- which were napari's dot-editing keys (1 deleted dots, 2-5 changed
their mode), not the review tool's speed keys. See review/keyboard.py.
"""
import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QShowEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QComboBox, QPushButton, QRadioButton, QSpinBox

from mousereach.review.keyboard import is_typing_key

napari = pytest.importorskip("napari")


def test_what_counts_as_typing():
    assert is_typing_key(Qt.Key_2, Qt.NoModifier, "2")
    assert is_typing_key(Qt.Key_B, Qt.NoModifier, "b")
    assert is_typing_key(Qt.Key_Exclam, Qt.ShiftModifier, "!")
    assert is_typing_key(Qt.Key_Backspace, Qt.ControlModifier, "")   # napari: delete layer
    assert is_typing_key(Qt.Key_Delete, Qt.NoModifier, "")
    assert is_typing_key(Qt.Key_Return, Qt.NoModifier, "\r")
    assert not is_typing_key(Qt.Key_Space, Qt.NoModifier, " ")      # play stays a video key
    assert not is_typing_key(Qt.Key_Right, Qt.NoModifier, "")       # stepping too
    assert not is_typing_key(Qt.Key_Left, Qt.ShiftModifier, "")
    assert not is_typing_key(Qt.Key_R, Qt.ControlModifier, "\x12")


@pytest.fixture(scope="module")
def viewer():
    # One viewer whose layers are never removed (each test adds its own), and
    # which is not closed: removing a layer -- closing does -- makes napari
    # draw, and a machine without an OpenGL screen (the processing server's
    # remote session) cannot. The hidden window goes with the test process.
    yield napari.Viewer(show=False)


@pytest.fixture
def tool(viewer):
    """The real review widget, docked in napari, with a dots layer the way it
    adds one (selected, because it was added last)."""
    from mousereach.review.causal_review_widget import CausalReviewWidget
    w = CausalReviewWidget(viewer, triage_only=True)
    dock = viewer.window.add_dock_widget(w, area="right")
    w.showEvent(QShowEvent())                 # what docking and showing does
    dots = viewer.add_points(np.array([[0, 5, 5], [0, 6, 6]]), name="DLC Points")
    from mousereach.review.keyboard import review_keys_on_layer
    review_keys_on_layer(dots, w._review_keys)
    jumps = []
    w._jump_frames = jumps.append              # the arrow keys' handler
    # A button and a dropdown in the panel, the kind of control a reviewer
    # clicks just before typing a frame number.
    button, combo = QPushButton("Use current frame"), QComboBox()
    combo.addItems(["untouched", "retrieved"])
    w.layout().addWidget(button)
    w.layout().addWidget(combo)
    yield w, dots, button, combo, jumps
    viewer.window.remove_dock_widget(dock)


def _unchanged(w, dots):
    return (w.playback_speed == 1.0 and not w.is_playing
            and str(dots.mode) == "pan_zoom" and len(dots.data) == 2)


def test_a_number_typed_with_the_keyboard_on_a_button_runs_nothing(tool):
    w, dots, button, combo, _ = tool
    dots.selected_data = {0}
    for key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_6, Qt.Key_B, Qt.Key_Delete,
                Qt.Key_Backspace):
        QTest.keyClick(button, key)
        assert _unchanged(w, dots), key
    QTest.keyClick(button, Qt.Key_Backspace, Qt.ControlModifier)
    assert dots in napari.current_viewer().layers             # the dots layer is still there
    QTest.keyClick(combo, Qt.Key_Delete)
    assert _unchanged(w, dots)


def test_the_arrows_still_step_the_video_after_picking_an_outcome(tool):
    w, _, _, combo, jumps = tool
    QTest.keyClick(combo, Qt.Key_Right)
    assert jumps == [1]


def test_the_number_keys_on_the_video_set_the_speed_and_leave_the_dots_alone(tool, viewer):
    w, dots, *_ = tool
    viewer.layers.selection.active = dots
    keys = viewer.window._qt_viewer._key_map_handler
    dots.selected_data = {0, 1}
    for key, speed in [("2", 0.5), ("3", 1.0), ("4", 2.0), ("5", 4.0), ("1", 0.25)]:
        keys.press_key(key)
        assert w.playback_speed == speed, key
        assert str(dots.mode) == "pan_zoom" and len(dots.data) == 2, key
    for key in ("Delete", "Backspace", "A", "P"):
        keys.press_key(key)
        assert str(dots.mode) == "pan_zoom" and len(dots.data) == 2, key


def test_choosing_manual_entry_gives_start_the_keyboard(tool, monkeypatch):
    w, *_ = tool
    focused = []
    monkeypatch.setattr(QSpinBox, "setFocus", lambda self, *a: focused.append(self))
    w.n_frames = 5000
    w._q_widgets = {}
    # kept in a name: Qt deletes a panel nothing refers to
    picker = w._make_reach_picker({
        "segment_num": 1, "start_frame": 100, "end_frame": 900,
        "outcome": "triaged", "interaction_frame": None, "causal_reach": None,
        "all_reaches": [{"reach_id": 0, "start": 200, "end": 240}]})
    rp = w._q_widgets["_reach_picker"]
    rp["manual_button"].click()
    assert focused == [rp["start_spin"]]
    assert picker is not None


def test_the_resegmentation_tool_does_the_same(viewer, tmp_path, monkeypatch):
    from mousereach.review.fix_segmentation_widget import FixSegmentationWidget
    w = FixSegmentationWidget(viewer, tmp_path)
    dock = viewer.window.add_dock_widget(w, area="right")
    try:
        w.showEvent(QShowEvent())
        assert dock._mousereach_typing_guard is not None
        focused = []
        monkeypatch.setattr(QSpinBox, "setFocus", lambda self, *a: focused.append(self))
        w.gw_no.click()
        assert w.gw_spin.isEnabled() and focused == [w.gw_spin]
    finally:
        viewer.window.remove_dock_widget(dock)
