"""Review Queues launcher tab -- clear the Triage and Deep-Review folders.

"The GUI is god": an operator clears the pipeline's two human-review queues by
clicking, without a terminal. This tab shows how many videos are waiting in each
queue and opens the right review tool for each:

  * Triage queue (Paths.TRIAGE_REVIEW) -- per-element "which reach / what
    outcome" questions; opens the causal review tool in TRIAGED-ONLY mode.
  * Deep-Review queue (Paths.DEEP_REVIEW) -- segmentation-failed or escalated
    videos; opens the causal review tool in DEEP-REVIEW mode (full-segment walk +
    a CLEAR button that writes {stem}_deep_review_cleared.json so the watcher
    re-injects the video at the pipeline start).

Each queue opens in ITS OWN napari viewer window. The causal review widget owns
the viewer's single playhead, layers, and single-key shortcuts, so two of them
cannot share one viewer -- a separate window per queue avoids the collision.

ASCII-only console output (Windows cp1252). Qt text may use Unicode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
)
from qtpy.QtCore import QTimer

from napari.utils.notifications import show_info, show_error

logger = logging.getLogger(__name__)


class ReviewQueuesWidget(QWidget):
    """Launcher tab to open + clear the Triage / Deep-Review queues."""

    def __init__(self, napari_viewer=None):
        super().__init__()
        self.viewer = napari_viewer
        self._open_windows: List = []   # keep refs so viewers are not GC'd
        self._build_ui()
        QTimer.singleShot(150, self._refresh_counts)

    def _build_ui(self):
        root = QVBoxLayout(self)

        intro = QLabel(
            "Clear the pipeline's human-review queues. Each opens in its own "
            "window. Resolving a triage video (or clearing a deep-review video) "
            "sends it back through the pipeline automatically."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#888;")
        root.addWidget(intro)

        # --- Triage ---
        tri = QGroupBox("Triage queue  (per-element questions)")
        tl = QVBoxLayout(tri)
        self._triage_count = QLabel("(counting...)")
        self._triage_count.setStyleSheet("font-weight:bold;")
        tl.addWidget(self._triage_count)
        tbtn = QPushButton("Open Triage Review")
        tbtn.setStyleSheet("background:#16405a; color:white; font-weight:bold;")
        tbtn.clicked.connect(self._open_triage)
        tl.addWidget(tbtn)
        root.addWidget(tri)

        # --- Deep review ---
        deep = QGroupBox("Deep-Review queue  (seg-failed / escalated)")
        dl = QVBoxLayout(deep)
        self._deep_count = QLabel("(counting...)")
        self._deep_count.setStyleSheet("font-weight:bold;")
        dl.addWidget(self._deep_count)
        dbtn = QPushButton("Open Deep Review")
        dbtn.setStyleSheet("background:#3a5a16; color:white; font-weight:bold;")
        dbtn.clicked.connect(self._open_deep)
        dl.addWidget(dbtn)
        sbtn = QPushButton("Open Re-segmentation")
        sbtn.setStyleSheet("background:#5a3a16; color:white; font-weight:bold;")
        sbtn.setToolTip(
            "Fix segment boundaries by hand for deep-review videos whose "
            "problem is the segmentation (segmenter needs_human, "
            "reviewer-declared mislabels, triage escalations). Saving stamps "
            "the cuts as human-made so the pipeline keeps them on re-run."
        )
        sbtn.clicked.connect(self._open_reseg)
        dl.addWidget(sbtn)
        root.addWidget(deep)

        rbtn = QPushButton("Refresh counts")
        rbtn.clicked.connect(self._refresh_counts)
        root.addWidget(rbtn)
        root.addStretch()

    # ------------------------------------------------------------- counts
    def _count(self, queue_root: Optional[Path]) -> int:
        if not queue_root:
            return -1  # not configured (NAS root unset)
        root = Path(queue_root)
        if not root.exists():
            return 0   # configured but empty -- the folder is created on first routing
        try:
            from mousereach.review.causal_review_io import bundle_manifest_path
            n = 0
            for d in root.iterdir():
                if d.is_dir() and bundle_manifest_path(d).exists():
                    n += 1
            return n
        except Exception as e:
            logger.debug(f"queue count failed: {e}")
            return -1

    def _refresh_counts(self):
        from mousereach.config import Paths
        for lbl, root, name in (
            (self._triage_count, getattr(Paths, "TRIAGE_REVIEW", None), "Triage"),
            (self._deep_count, getattr(Paths, "DEEP_REVIEW", None), "Deep-Review"),
        ):
            n = self._count(root)
            if n < 0:
                lbl.setText(f"{name}: (queue not configured -- NAS root unset)")
            else:
                lbl.setText(f"{name}: {n} video(s) waiting")

    # -------------------------------------------------------------- open
    def _open_triage(self):
        from mousereach.config import Paths
        self._open_queue(getattr(Paths, "TRIAGE_REVIEW", None),
                         triage_only=True, deep_review=False, title="Triage Review")

    def _open_deep(self):
        from mousereach.config import Paths
        self._open_queue(getattr(Paths, "DEEP_REVIEW", None),
                         triage_only=False, deep_review=True, title="Deep Review")

    def _open_reseg(self):
        """Open the fix-segmentation tool over the deep-review queue in its own
        window -- manual boundary cuts for videos whose problem is the
        segmentation itself."""
        from mousereach.config import Paths
        queue_root = getattr(Paths, "DEEP_REVIEW", None)
        if not queue_root or not Path(queue_root).exists():
            show_error(f"Re-segmentation: queue folder not found ({queue_root}).")
            return
        try:
            import napari
            from mousereach.review.fix_segmentation_widget import FixSegmentationWidget
            v = napari.Viewer(title="Re-segmentation")
            w = FixSegmentationWidget(v, Path(queue_root))
            v.window.add_dock_widget(w, name="Re-segmentation", area="right")
            self._open_windows.append((v, w))
            show_info("Opened Re-segmentation.")
        except Exception as e:
            show_error(f"Could not open Re-segmentation: {e}")
            logger.exception("open re-segmentation failed")
        self._refresh_counts()

    def _open_queue(self, queue_root, triage_only, deep_review, title):
        if not queue_root or not Path(queue_root).exists():
            show_error(f"{title}: queue folder not found ({queue_root}).")
            return
        try:
            import napari
            from mousereach.review.causal_review_widget import CausalReviewWidget
            v = napari.Viewer(title=title)
            w = CausalReviewWidget(v, triage_only=triage_only, deep_review=deep_review)
            v.window.add_dock_widget(w, name=title, area="right")
            w.load_pending_queue(Path(queue_root))
            self._open_windows.append((v, w))
            show_info(f"Opened {title}.")
        except Exception as e:
            show_error(f"Could not open {title}: {e}")
            logger.exception("open review queue failed")
        self._refresh_counts()


def main():
    """Standalone launch of the Review Queues panel."""
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(ReviewQueuesWidget(viewer), name="Review Queues", area="right")
    napari.run()


if __name__ == "__main__":
    main()
