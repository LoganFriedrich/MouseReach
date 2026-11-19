"""Make the review windows usable on a small screen (e.g. 1366x768).

WHY THIS EXISTS: the review tools open in their own napari window with the
review panel docked on the right. On a 1366x768 display napari's own layer
panels took about 270 px on the left, the review panel asked for more width
than was left, and the video -- the thing being reviewed -- was squeezed to
about 200 px. The panel itself was then wider than the room it got, so its
text and table columns were cut off on the right.

The rules applied here, by every review window:

* napari's layer controls and layer list are hidden. A reviewer never uses
  them; the Window menu brings them back.
* The panel is never wider than about 40% of the screen, so the video always
  gets most of the width. Anything inside it that cannot fit wraps (labels)
  or scrolls (the panel sits in a scroll area), instead of being cut off.
* Optional sections fold away (``CollapsibleSection``), closed by default,
  so what a reviewer must answer comes first.

Nothing here changes what any control does; it only decides where things sit.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QLabel, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget,
)

# The review panel never takes more than this share of the screen's width.
PANEL_MAX_FRACTION = 0.40
# ...but is never capped below this, so a panel stays readable on a very
# small screen (it scrolls sideways instead).
PANEL_MIN_CAP_PX = 380


def screen_width(widget: QWidget = None) -> int:
    """Width in pixels of the screen the widget is on (primary screen if it is
    not shown yet). 0 when Qt cannot say."""
    try:
        scr = None
        if widget is not None and hasattr(widget, "screen"):
            scr = widget.screen()
        if scr is None:
            scr = QApplication.primaryScreen()
        return int(scr.availableGeometry().width()) if scr is not None else 0
    except Exception:
        return 0


def panel_width_cap(widget: QWidget = None) -> int:
    """The widest a review panel may be on this screen."""
    w = screen_width(widget)
    if w <= 0:
        return 0
    return max(PANEL_MIN_CAP_PX, int(w * PANEL_MAX_FRACTION))


def hide_layer_docks(viewer) -> None:
    """Hide napari's layer controls and layer list. A reviewer does not need
    them, and on a small screen they took a fifth of the width from the
    video. Window menu -> layer controls / layer list brings them back."""
    try:
        qv = getattr(viewer.window, "_qt_viewer", None) or viewer.window.qt_viewer
        qv.dockLayerControls.setVisible(False)
        qv.dockLayerList.setVisible(False)
    except Exception:
        pass


def wrap_all_labels(root: QWidget) -> None:
    """Turn on word wrap for every label under ``root``, so a long line wraps
    inside the panel instead of running off its right edge."""
    try:
        for lbl in root.findChildren(QLabel):
            lbl.setWordWrap(True)
    except Exception:
        pass


def fit_review_window(viewer, widget: QWidget, dock=None) -> None:
    """Apply the small-screen rules to one review window: hide napari's layer
    panels, cap the panel's width at about 40% of the screen, and size the
    dock to the panel's own preferred width within that cap.

    Call it right after ``viewer.window.add_dock_widget(widget, ...)``, with
    the dock that call returned."""
    hide_layer_docks(viewer)
    cap = panel_width_cap(widget)
    if cap > 0:
        widget.setMaximumWidth(cap)
    if dock is not None and cap > 0:
        try:
            want = min(cap, max(widget.sizeHint().width(), PANEL_MIN_CAP_PX))
            viewer.window._qt_window.resizeDocks([dock], [want], Qt.Horizontal)
        except Exception:
            pass


def scrollable_body(owner: QWidget) -> QVBoxLayout:
    """Give ``owner`` a vertical layout holding one scroll area, and return
    the layout of the widget inside it -- build the panel's contents there.
    Labels wrap to the panel's width; whatever still cannot fit (a row of
    buttons) scrolls sideways rather than being cut off."""
    outer = QVBoxLayout()
    outer.setContentsMargins(0, 0, 0, 0)
    owner.setLayout(outer)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setFrameShape(QScrollArea.NoFrame)
    inner = QWidget()
    body = QVBoxLayout()
    inner.setLayout(body)
    scroll.setWidget(inner)
    outer.addWidget(scroll, 1)
    owner._scroll_area = scroll
    owner._scroll_body = inner
    return body


class CollapsibleSection(QWidget):
    """A section with a clickable title that shows or hides its contents.

    Closed by default: used for the optional parts of a review panel, so the
    questions a reviewer must answer come first on a small screen. Clicking
    the title (the arrow) opens it; nothing inside changes behaviour while it
    is closed."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content = QWidget()
        self.content.setVisible(expanded)
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)
        self.setLayout(lay)
        self.toggle.toggled.connect(self.set_expanded)

    def set_expanded(self, on: bool) -> None:
        on = bool(on)
        if self.toggle.isChecked() != on:
            self.toggle.setChecked(on)      # re-enters through toggled
            return
        self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.content.setVisible(on)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()
