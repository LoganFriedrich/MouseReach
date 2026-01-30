"""
Reusable Save Panel for Review Widgets
=====================================

Provides a clear, intuitive UI for the two save modes:
1. Save & Continue (primary) - Normal workflow, moves to next step
2. Save as Ground Truth (secondary) - For algorithm evaluation

Design principles:
- Primary action is obvious and prominent
- Secondary action is available but not distracting
- Clear explanations visible without hovering
- No jargon - plain language
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFrame, QMessageBox
)
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QFont


class SavePanel(QWidget):
    """
    Clear, intuitive save panel with two modes:

    1. "Done - Save & Continue" (green, prominent)
       - Saves your corrections
       - Marks video as reviewed
       - Ready for next pipeline step

    2. "Save as Reference Standard" (gray, collapsible)
       - Creates a "gold standard" file
       - Used to measure algorithm accuracy
       - For developers/researchers only
    """

    # Signals
    save_validated = Signal()  # User clicked primary save
    save_ground_truth = Signal()  # User clicked GT save

    def __init__(self, step_name: str = "review", parent=None):
        """
        Args:
            step_name: Name of the step (e.g., "boundaries", "reaches", "outcomes")
                      Used in button labels and messages
        """
        super().__init__(parent)
        self.step_name = step_name
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # === PRIMARY SAVE (Always visible, prominent) ===
        primary_group = QGroupBox("Save Your Work")
        primary_layout = QVBoxLayout()
        primary_group.setLayout(primary_layout)

        # Explanation
        primary_info = QLabel(
            "When you're done reviewing, save your corrections.\n"
            "The video will be marked as reviewed and ready for the next step."
        )
        primary_info.setWordWrap(True)
        primary_info.setStyleSheet("color: #aaa; font-size: 11px; padding-bottom: 5px;")
        primary_layout.addWidget(primary_info)

        # Primary save button - big, green, obvious
        self.save_btn = QPushButton("✓ Done - Save & Continue")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 12px 20px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        self.save_btn.clicked.connect(self._on_save_validated)
        self.save_btn.setEnabled(False)
        primary_layout.addWidget(self.save_btn)

        # Shortcut hint
        shortcut_label = QLabel("Keyboard: S")
        shortcut_label.setStyleSheet("color: #666; font-size: 10px;")
        shortcut_label.setAlignment(Qt.AlignCenter)
        primary_layout.addWidget(shortcut_label)

        layout.addWidget(primary_group)

        # === SECONDARY SAVE (Collapsible, less prominent) ===
        self.gt_group = QGroupBox("For Researchers (Advanced)")
        self.gt_group.setCheckable(True)
        self.gt_group.setChecked(False)  # Collapsed by default
        self.gt_group.setStyleSheet("""
            QGroupBox {
                color: #888;
                font-size: 11px;
            }
            QGroupBox::title {
                color: #888;
            }
        """)
        gt_layout = QVBoxLayout()
        self.gt_group.setLayout(gt_layout)

        # Container for collapsible content
        self.gt_content = QWidget()
        gt_content_layout = QVBoxLayout()
        gt_content_layout.setContentsMargins(0, 0, 0, 0)
        self.gt_content.setLayout(gt_content_layout)

        # Explanation
        gt_info = QLabel(
            "<b>What is a Reference Standard?</b><br>"
            "A carefully reviewed \"gold standard\" annotation used to measure "
            "how well the automatic algorithm performs.<br><br>"
            "<b>When to use:</b> Only if you're evaluating algorithm accuracy "
            "or training new models. Normal users don't need this."
        )
        gt_info.setWordWrap(True)
        gt_info.setStyleSheet("color: #888; font-size: 11px; padding: 5px;")
        gt_info.setTextFormat(Qt.RichText)
        gt_content_layout.addWidget(gt_info)

        # GT save button - subtle, gray
        self.save_gt_btn = QPushButton("Save as Reference Standard")
        self.save_gt_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: #ccc;
                font-size: 12px;
                padding: 8px 15px;
                border: 1px solid #666;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666;
                color: white;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #666;
            }
        """)
        self.save_gt_btn.clicked.connect(self._on_save_ground_truth)
        self.save_gt_btn.setEnabled(False)
        gt_content_layout.addWidget(self.save_gt_btn)

        gt_layout.addWidget(self.gt_content)

        # Connect checkbox to show/hide content
        self.gt_group.toggled.connect(self._on_gt_group_toggled)
        self._on_gt_group_toggled(False)  # Start collapsed

        layout.addWidget(self.gt_group)

        # === Status Label ===
        self.status_label = QLabel("Load a video to begin")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 5px;")
        layout.addWidget(self.status_label)

    def _on_gt_group_toggled(self, checked: bool):
        """Show/hide GT content when group is toggled."""
        self.gt_content.setVisible(checked)

    def _on_save_validated(self):
        """Handle primary save button click."""
        self.save_validated.emit()

    def _on_save_ground_truth(self):
        """Handle GT save with confirmation dialog."""
        reply = QMessageBox.question(
            self,
            "Save as Reference Standard?",
            "This creates a 'gold standard' file for algorithm evaluation.\n\n"
            "Are you sure you want to save as Reference Standard?\n\n"
            "(If you just want to save your corrections and continue, "
            "use 'Done - Save & Continue' instead.)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.save_ground_truth.emit()

    def set_enabled(self, enabled: bool):
        """Enable or disable both save buttons."""
        self.save_btn.setEnabled(enabled)
        self.save_gt_btn.setEnabled(enabled)

    def set_status(self, text: str, is_success: bool = True):
        """Set the status label text."""
        color = "#4CAF50" if is_success else "#f44336"
        self.status_label.setStyleSheet(f"color: {color}; padding: 5px;")
        self.status_label.setText(text)

    def clear_status(self):
        """Clear the status label."""
        self.status_label.setText("")


class SimpleSavePanel(QWidget):
    """
    Even simpler save panel for inline use.

    Just two clearly labeled buttons with inline explanations.
    No groupboxes or collapsible sections.

    Modes:
    - review_mode=False (default): Shows both "Save & Continue" and "Save as GT"
    - review_mode=True: Only shows "Save & Continue" (for Review Tool)
    """

    save_progress = Signal()  # Optional: save work-in-progress
    save_validated = Signal()
    save_ground_truth = Signal()

    def __init__(self, show_progress_save: bool = False, review_mode: bool = False, parent=None):
        """
        Args:
            show_progress_save: If True, show a "Save Progress" button for
                               intermediate saves (useful for longer review tasks)
            review_mode: If True, hide the Ground Truth save option.
                        Review Tool edits algo files directly, not GT files.
        """
        super().__init__(parent)
        self.show_progress_save = show_progress_save
        self.review_mode = review_mode
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.setLayout(layout)

        # === OPTIONAL: Save Progress (for longer tasks) ===
        if self.show_progress_save:
            progress_frame = QFrame()
            progress_frame.setStyleSheet("""
                QFrame {
                    background-color: #3a4a5a;
                    border-radius: 8px;
                    padding: 8px;
                }
            """)
            progress_layout = QVBoxLayout()
            progress_layout.setContentsMargins(10, 8, 10, 8)
            progress_frame.setLayout(progress_layout)

            progress_desc = QLabel("Not finished yet? Save and continue later")
            progress_desc.setStyleSheet("color: #8ac; font-size: 11px;")
            progress_layout.addWidget(progress_desc)

            self.save_progress_btn = QPushButton("Save Progress (Ctrl+S)")
            self.save_progress_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a6a8a;
                    color: white;
                    font-size: 12px;
                    padding: 8px;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover { background-color: #5a7a9a; }
                QPushButton:disabled { background-color: #555; color: #888; }
            """)
            self.save_progress_btn.clicked.connect(self.save_progress.emit)
            self.save_progress_btn.setEnabled(False)
            progress_layout.addWidget(self.save_progress_btn)

            layout.addWidget(progress_frame)
        else:
            self.save_progress_btn = None

        # === PRIMARY: Save & Continue ===
        primary_frame = QFrame()
        primary_frame.setStyleSheet("""
            QFrame {
                background-color: #2d4a2d;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        primary_layout = QVBoxLayout()
        primary_layout.setContentsMargins(10, 10, 10, 10)
        primary_frame.setLayout(primary_layout)

        primary_title = QLabel("<b>Finished Reviewing?</b>")
        primary_title.setStyleSheet("color: #8BC34A; font-size: 12px;")
        primary_layout.addWidget(primary_title)

        primary_desc = QLabel("Save corrections and mark as reviewed")
        primary_desc.setStyleSheet("color: #aaa; font-size: 11px;")
        primary_layout.addWidget(primary_desc)

        self.save_btn = QPushButton("✓ Done - Save & Continue (S)")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.save_btn.clicked.connect(self.save_validated.emit)
        self.save_btn.setEnabled(False)
        primary_layout.addWidget(self.save_btn)

        layout.addWidget(primary_frame)

        # === SECONDARY: Ground Truth (hidden in review_mode) ===
        if not self.review_mode:
            secondary_frame = QFrame()
            secondary_frame.setStyleSheet("""
                QFrame {
                    background-color: #3a3a3a;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
            secondary_layout = QVBoxLayout()
            secondary_layout.setContentsMargins(10, 10, 10, 10)
            secondary_frame.setLayout(secondary_layout)

            secondary_title = QLabel("<b>Algorithm Evaluation</b> (researchers only)")
            secondary_title.setStyleSheet("color: #888; font-size: 11px;")
            secondary_layout.addWidget(secondary_title)

            secondary_desc = QLabel("Create gold standard for measuring accuracy")
            secondary_desc.setStyleSheet("color: #666; font-size: 10px;")
            secondary_layout.addWidget(secondary_desc)

            self.save_gt_btn = QPushButton("Save as Reference Standard")
            self.save_gt_btn.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    color: #aaa;
                    font-size: 11px;
                    padding: 6px;
                    border: 1px solid #666;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #666; color: white; }
                QPushButton:disabled { background-color: #444; color: #555; }
            """)
            self.save_gt_btn.clicked.connect(self._on_gt_click)
            self.save_gt_btn.setEnabled(False)
            secondary_layout.addWidget(self.save_gt_btn)

            layout.addWidget(secondary_frame)
        else:
            # Review mode: no GT save option
            self.save_gt_btn = None

        # Status
        self.status_label = QLabel("Load a video to begin")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _on_gt_click(self):
        """Confirm before saving as ground truth."""
        reply = QMessageBox.question(
            self,
            "Save as Reference Standard?",
            "This creates a 'gold standard' file for algorithm evaluation.\n\n"
            "Normal users should use 'Save & Continue' instead.\n\n"
            "Continue with Reference Standard save?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_ground_truth.emit()

    def set_enabled(self, enabled: bool):
        if self.save_progress_btn is not None:
            self.save_progress_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        if self.save_gt_btn is not None:
            self.save_gt_btn.setEnabled(enabled)

    def set_status(self, text: str, is_success: bool = True):
        color = "#4CAF50" if is_success else "#f44336"
        self.status_label.setStyleSheet(f"color: {color}; padding: 5px;")
        self.status_label.setText(text)
