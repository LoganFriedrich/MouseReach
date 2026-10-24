"""The review tool's pillar circle must not be pinned to a frame.

A circle drawn as (frame, y, x) changed the layer's frame extent on every
update; the viewer then moved the frame slider, which redrew the circle, and
while a new video's layers were swapped that loop nested until the tool died of
a stack overflow (2026-10-23). Drawn in (y, x), it cannot touch the frame axis.
"""
from types import SimpleNamespace

import pandas as pd

from mousereach.review.causal_review_widget import CausalReviewWidget


def test_pillar_circle_is_drawn_in_two_dimensions():
    layer = SimpleNamespace(data=None)
    dlc = pd.DataFrame([{"SABL_x": 100.0, "SABL_y": 400.0, "SABL_likelihood": 0.99,
                         "SABR_x": 200.0, "SABR_y": 400.0, "SABR_likelihood": 0.99}] * 10)
    stub = SimpleNamespace(viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(7, 0, 0))),
                           pillar_shapes_layer=layer, dlc_df=dlc, scale_factor=1.0)
    CausalReviewWidget._update_pillar_circle(stub)
    (circle,) = layer.data
    assert circle.shape[1] == 2                       # (y, x) only -- no frame column
    assert abs(circle[:, 1].mean() - 150.0) < 1.0     # centred between the two corners
    assert circle[:, 0].mean() < 400.0                # above the slot's bottom edge
