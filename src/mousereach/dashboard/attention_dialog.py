"""The "Resolve..." window: one row per item a person can act on -- a review
bundle that a route left without all its files, or a queue folder left for a
person -- with a button for each thing a person can do about it
(watcher/attention_actions.py). Every button asks before it acts, says what
it will delete if it deletes anything, and says afterwards what it did. WHY:
these holds must be resolvable from the screen. Opened from the Pipeline
Dashboard and from the Review Queues tab ('Folders left for a person').
"""
from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
    QHBoxLayout, QFrame,
)


class AttentionResolveDialog(QDialog):
    """Lists ``items`` (from attention.read_items) that have actions."""

    INTRO = (
        "Each row is a review bundle that a route left without all its files, or "
        "a queue folder left for a person. Every button asks before it acts. "
        "'Try to move the missing files in now' deletes a file at the route's old "
        "place only when the bundle already holds the same file byte for byte. "
        "'Nothing in it is needed: delete this folder' deletes that folder and "
        "everything in it. No other button deletes anything.")

    def __init__(self, items, parent=None, on_change=None):
        super().__init__(parent)
        self.setWindowTitle("Resolve what needs your attention")
        self.resize(820, 560)
        self._items = list(items)
        self._on_change = on_change
        outer = QVBoxLayout(self)
        intro = QLabel(self.INTRO)
        intro.setWordWrap(True)
        outer.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._body = QWidget()
        self._rows = QVBoxLayout(self._body)
        scroll.setWidget(self._body)
        outer.addWidget(scroll)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        outer.addWidget(close)
        self._fill()

    def _fill(self):
        from mousereach.watcher.attention_actions import actions_for
        while self._rows.count():
            w = self._rows.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        shown = 0
        for item in self._items:
            actions = actions_for(item)
            if not actions:
                continue
            shown += 1
            box = QFrame()
            box.setFrameShape(QFrame.StyledPanel)
            lay = QVBoxLayout(box)
            text = QLabel(item.get("says", ""))
            text.setWordWrap(True)
            lay.addWidget(text)
            buttons = QHBoxLayout()
            for key, label, explain in actions:
                if key.startswith("note:"):
                    # Not a button: why a file of this item has none.
                    note = QLabel(explain)
                    note.setWordWrap(True)
                    lay.addWidget(note)
                    continue
                b = QPushButton(label)
                b.setToolTip(explain)
                b.clicked.connect(lambda _=False, i=item, k=key, lab=label, e=explain:
                                  self._run(i, k, lab, e))
                buttons.addWidget(b)
            lay.addLayout(buttons)
            self._rows.addWidget(box)
        if not shown:
            self._rows.addWidget(QLabel("Nothing here needs a decision right now."))
        self._rows.addStretch(1)

    def _run(self, item, key, label, explain):
        from mousereach.watcher.attention_actions import ActionRefused, perform
        if QMessageBox.question(self, label, f"{explain}\n\nGo ahead?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            msg = perform(item, key)
        except ActionRefused as e:
            QMessageBox.warning(self, label, str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, label, f"It did not work: {e}")
            return
        QMessageBox.information(self, label, msg)
        if self._on_change is not None:
            self._items = list(self._on_change())
        self._fill()
