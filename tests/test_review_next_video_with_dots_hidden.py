"""Opening the next video with the tracking dots hidden does not crash.

Colin, 2026-10-29: every next video failed to load with "index 596438 is out of
bounds for axis 0 with size 564876". The review tool reuses one dots layer for
every video. napari remembers which dots are on the current frame and does not
re-take that list while the layer is hidden, so after the reviewer hid the dots
and moved to a video with fewer points, setting the new colours redrew from the
previous video's list -- which pointed past the end of the new points.
"""
import numpy as np
import pandas as pd
import pytest
from qtpy.QtGui import QShowEvent

napari = pytest.importorskip("napari")


@pytest.fixture(scope="module")
def viewer():
    # Not closed: see test_typing_never_runs_video_keys (no OpenGL screen here).
    yield napari.Viewer(show=False)


def _pose(n_frames, bodyparts):
    cols = {}
    for bp in bodyparts:
        cols[bp + "_x"] = np.full(n_frames, 10.0)
        cols[bp + "_y"] = np.full(n_frames, 20.0)
        cols[bp + "_likelihood"] = np.full(n_frames, 0.9)
    return pd.DataFrame(cols)


def test_next_video_with_fewer_points_loads_while_the_dots_are_hidden(viewer):
    from mousereach.review.causal_review_widget import CausalReviewWidget
    w = CausalReviewWidget(viewer, triage_only=True)
    viewer.window.add_dock_widget(w, area="right")
    w.showEvent(QShowEvent())
    w.scale_factor = 1.0

    w.dlc_df = _pose(1000, ["a", "b", "c", "d", "e", "f"])     # first video: 6000 dots
    w._add_dlc_points_layer()
    layer = w.points_layer
    viewer.dims.set_point(0, 999)          # on the last frame: the highest-numbered dots
    layer.visible = False                  # the reviewer hides the dots

    w.dlc_df = _pose(1000, ["a", "b", "c", "d", "e"])          # next video: 5000 dots
    w._add_dlc_points_layer()                                  # raised IndexError before

    assert w.points_layer is layer and len(layer.data) == 5000
    assert len(layer._view_data) == 5                          # the 5 dots of frame 999
    layer.visible = True                                       # and showing them again works
    assert len(layer._view_data) == 5
