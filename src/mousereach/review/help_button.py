"""A "?" help button for every review tool's top-right corner.

Every napari review tool gets the same affordance: a small question-mark
button that opens a QUICK GUIDE dialog (what this tool is for, what to do,
what every control does), with a button inside that opens the full
documentation (docs/REVIEW_TOOLS.md) for the deeper dive. One implementation
here so the tools cannot drift apart in how help looks or where it lives.

Usage (inside a widget's _build_ui, on its top row/layout):

    from mousereach.review.help_button import attach_help
    attach_help(header_layout, tool_title="Re-segmentation",
                quick_guide=QUICK_GUIDE_TEXT)

``quick_guide`` is plain text (blank-line paragraphs; lines starting with
"* " render as bullets). ASCII only -- it may be printed to consoles too.
"""
from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, QToolButton,
)


def _docs_path() -> Path | None:
    """docs/REVIEW_TOOLS.md of the repo this package was imported from."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        cand = parent / "docs" / "REVIEW_TOOLS.md"
        if cand.is_file():
            return cand
    return None


class HelpDialog(QDialog):
    def __init__(self, tool_title: str, quick_guide: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("%s -- quick guide" % tool_title)
        self.resize(560, 520)
        lay = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        html_parts = []
        for para in quick_guide.strip().split("\n\n"):
            lines = para.splitlines()
            if all(l.lstrip().startswith("* ") for l in lines):
                items = "".join("<li>%s</li>" % l.lstrip()[2:] for l in lines)
                html_parts.append("<ul>%s</ul>" % items)
            else:
                html_parts.append("<p>%s</p>" % " ".join(lines))
        browser.setHtml("<h3>%s</h3>%s" % (tool_title, "".join(html_parts)))
        lay.addWidget(browser)

        row = QHBoxLayout()
        doc = _docs_path()
        deep = QPushButton("Open the full documentation (REVIEW_TOOLS.md)")
        deep.setToolTip(str(doc) if doc else "docs/REVIEW_TOOLS.md not found "
                        "next to this install")
        deep.setEnabled(doc is not None)
        if doc is not None:
            deep.clicked.connect(lambda: os.startfile(str(doc)))
        row.addWidget(deep)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)


def attach_help(header_layout, tool_title: str, quick_guide: str,
                parent=None) -> QToolButton:
    """Add the "?" button to the RIGHT end of ``header_layout`` and return it."""
    btn = QToolButton(parent)
    btn.setText("?")
    btn.setToolTip("Quick guide: what this tool is for and what every "
                   "control does. The full documentation opens from inside.")
    btn.setStyleSheet("QToolButton { font-weight: bold; border: 1px solid "
                      "#888; border-radius: 9px; min-width: 18px; "
                      "min-height: 18px; }")
    btn.clicked.connect(
        lambda: HelpDialog(tool_title, quick_guide, parent).exec_())
    header_layout.addWidget(btn)
    return btn
