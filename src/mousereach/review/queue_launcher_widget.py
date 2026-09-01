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
    QMessageBox,
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

    QUICK_GUIDE = """
The pipeline holds videos it cannot finish on its own in two queues; this
tab is where a person clears them. Counts refresh on open and via the
Refresh button.

* Open Triage Review -- per-segment questions (which reach, what outcome,
  bench-sheet disagreements). Resolving every flagged segment releases the
  video automatically.
* Open Deep Review -- full-video walk for held-out videos; its
  "Clear -> re-enter pipeline" button is what releases a deep-review video.
* Open Re-segmentation -- fix WHERE the segment cuts are, guided
  segment-by-segment. Use for boundary/numbering problems. Saving now
  releases the video by itself IF it was routed here for segmentation;
  otherwise the status bar says why it stays.
* Release finished videos -- one button that removes every FINISHED
  video from the deep-review queue and sends it back through the
  pipeline (fully answered reviews + hand-fixed segmentations routed
  for segmentation). It never touches videos held for other reasons;
  those say so and release via Clear inside Deep Review.

Each tool opens in its own window and has its own ? guide.
"""

    def _build_ui(self):
        root = QVBoxLayout(self)

        head = QHBoxLayout()
        head.addStretch()
        try:
            from mousereach.review.help_button import attach_help
            attach_help(head, "Review Queues", self.QUICK_GUIDE, self)
        except Exception:
            pass
        root.addLayout(head)

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

        # Release panel. The reviewer works ONLY through this GUI, and the
        # button's label must state its effect outright -- it releases
        # finished videos out of this queue and back into the pipeline.
        self._release_info = QLabel("")
        self._release_info.setWordWrap(True)
        self._release_info.setStyleSheet("color:#888;")
        dl.addWidget(self._release_info)
        self._release_btn = QPushButton("Release finished videos -> back into pipeline")
        self._release_btn.setStyleSheet("background:#0f5a2e; color:white; font-weight:bold;")
        self._release_btn.setToolTip(
            "Removes every FINISHED video from this queue and sends it back "
            "through the pipeline: reviews with every segment answered, and "
            "hand-fixed segmentations on videos that were routed here FOR a "
            "segmentation problem. The watcher picks them up within about two "
            "minutes and re-runs from the reach stage on the existing pose; "
            "hand-set cuts are preserved. Videos held for other reasons (a QC "
            "hold, no routing reason recorded) are NOT touched by this button "
            "-- open Deep Review and use 'Clear -> re-enter pipeline' on those "
            "once their real concern is addressed.")
        self._release_btn.setEnabled(False)
        self._release_btn.clicked.connect(self._release_finished_clicked)
        dl.addWidget(self._release_btn)
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
        self._refresh_release_info()

    def _refresh_release_info(self):
        """Populate the release panel: how many videos this button would
        release, and how many it deliberately will not (with why)."""
        from mousereach.config import Paths
        queue = getattr(Paths, "DEEP_REVIEW", None)
        try:
            from mousereach.review.release_cli import classify_queue
            self._release_cls = classify_queue(queue)
        except Exception as e:
            self._release_info.setText(f"(release check failed: {e})")
            self._release_btn.setEnabled(False)
            return
        c = self._release_cls
        n = len(c["complete"]) + len(c["fixed_release"])
        held = len(c["fixed_held"])
        pw = len(c["partial_walk"])
        bits = []
        if n:
            bits.append(f"{n} finished and releasable "
                        f"({len(c['complete'])} fully answered, "
                        f"{len(c['fixed_release'])} segmentation fixed by hand)")
        else:
            bits.append("nothing finished to release")
        if held:
            bits.append(f"{held} held back -- routed for something a cut-fix "
                        f"does not address; release those with Clear inside "
                        f"Deep Review")
        if pw:
            bits.append(f"{pw} answered for every triaged segment -- open "
                        f"Deep Review to judge and Clear them")
        if c["already"]:
            bits.append(f"{len(c['already'])} already released, awaiting "
                        f"pickup (~2 min)")
        self._release_info.setText("; ".join(bits) + ".")
        if n:
            self._release_btn.setText(
                f"Release {n} finished video(s) -> back into pipeline")
            self._release_btn.setEnabled(True)
        else:
            self._release_btn.setText(
                "Release finished videos -> back into pipeline (none right now)")
            self._release_btn.setEnabled(False)

    def _release_finished_clicked(self):
        """Confirm, then write the release marker for every finished video.
        Never touches held-back or partial bundles."""
        from mousereach.config import Paths
        queue = getattr(Paths, "DEEP_REVIEW", None)
        c = getattr(self, "_release_cls", None)
        if not queue or not c:
            show_error("Release: queue not available.")
            return
        n = len(c["complete"]) + len(c["fixed_release"])
        if not n:
            show_info("Nothing finished to release.")
            return
        held = len(c["fixed_held"])
        msg = (f"Release {n} finished video(s) OUT of the deep-review queue "
               f"and back into the pipeline?\n\n"
               f"  {len(c['complete'])} with every segment answered\n"
               f"  {len(c['fixed_release'])} with segmentation fixed by hand "
               f"(routed here for segmentation)\n\n"
               f"They re-enter automatically within ~2 minutes; hand-set cuts "
               f"are preserved.")
        if held:
            msg += (f"\n\n{held} video(s) will NOT be touched (routed for "
                    f"something a cut-fix does not address) -- they stay in "
                    f"the queue until cleared inside Deep Review.")
        if QMessageBox.question(
                self, "Release finished videos?", msg,
                QMessageBox.Ok | QMessageBox.Cancel) != QMessageBox.Ok:
            return
        try:
            from mousereach.review.release_cli import release_finished
            done, failures = release_finished(Path(queue), c)
            if failures:
                show_error(f"Released {done}; {len(failures)} failed -- "
                           f"first: {failures[0]}")
            else:
                show_info(f"Released {done} video(s) back into the pipeline. "
                          f"The watcher picks them up within ~2 minutes.")
        except Exception as e:
            show_error(f"Release failed: {e}")
        self._refresh_counts()

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
