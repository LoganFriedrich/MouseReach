"""
Comparison Panel Widget - Reusable algo vs GT comparison UI component.

This panel can be added to any review widget to show:
- Algorithm output on the left (read-only)
- Ground truth or editable copy on the right
- Diff summary showing differences
- Save Validation and Save as GT buttons
"""

from pathlib import Path
from typing import Optional, List, Callable, Any
from dataclasses import dataclass
import json

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSplitter, QListWidget, QListWidgetItem,
    QAbstractItemView, QFrame
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QBrush


@dataclass
class ComparisonItem:
    """An item to display in the comparison panel."""
    index: int
    algo_value: Any
    gt_value: Any
    label: str  # Display label (e.g., "Boundary 1", "Segment 3")
    diff: Optional[Any] = None  # Computed difference


class ComparisonPanel(QWidget):
    """
    Reusable widget for comparing algorithm output vs ground truth.

    Usage:
        panel = ComparisonPanel()
        panel.set_items(comparison_items)
        panel.validation_saved.connect(on_validation_saved)
        panel.gt_saved.connect(on_gt_saved)
        layout.addWidget(panel)
    """

    # Signals
    validation_saved = Signal()  # Emitted when Save Validation clicked
    gt_saved = Signal()  # Emitted when Save as GT clicked
    item_selected = Signal(int)  # Emitted when an item is selected (index)
    item_edited = Signal(int, object)  # Emitted when GT value edited (index, new_value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: List[ComparisonItem] = []
        self._gt_exists = False
        self._editable = True

        self._build_ui()

    def _build_ui(self):
        """Build the comparison panel UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Header
        header = QLabel("<b>Algorithm vs Ground Truth</b>")
        header.setStyleSheet("font-size: 12px; padding: 5px;")
        layout.addWidget(header)

        # Splitter for side-by-side view
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Algorithm output
        algo_group = QGroupBox("Algorithm Output")
        algo_layout = QVBoxLayout()
        algo_group.setLayout(algo_layout)

        self._algo_list = QListWidget()
        self._algo_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._algo_list.setStyleSheet("QListWidget { background-color: #2a2a2a; }")
        self._algo_list.currentRowChanged.connect(self._on_algo_selection)
        algo_layout.addWidget(self._algo_list)

        splitter.addWidget(algo_group)

        # Right panel: Ground Truth
        gt_group = QGroupBox("Ground Truth")
        gt_layout = QVBoxLayout()
        gt_group.setLayout(gt_layout)

        self._gt_header = QLabel("(No GT file - showing algo copy)")
        self._gt_header.setStyleSheet("color: #888; font-size: 10px;")
        gt_layout.addWidget(self._gt_header)

        self._gt_list = QListWidget()
        self._gt_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._gt_list.currentRowChanged.connect(self._on_gt_selection)
        gt_layout.addWidget(self._gt_list)

        splitter.addWidget(gt_group)

        layout.addWidget(splitter)

        # Diff summary
        diff_frame = QFrame()
        diff_layout = QHBoxLayout()
        diff_layout.setContentsMargins(5, 5, 5, 5)
        diff_frame.setLayout(diff_layout)

        self._diff_label = QLabel("No comparison data")
        self._diff_label.setStyleSheet("font-weight: bold;")
        diff_layout.addWidget(self._diff_label)

        layout.addWidget(diff_frame)

        # Save buttons
        buttons_layout = QHBoxLayout()

        self._save_validation_btn = QPushButton("Save Validation")
        self._save_validation_btn.setToolTip(
            "Save corrections back to the algorithm output file.\n"
            "Fixes algo output for downstream pipeline steps."
        )
        self._save_validation_btn.clicked.connect(self._on_save_validation)
        buttons_layout.addWidget(self._save_validation_btn)

        buttons_layout.addStretch()

        self._save_gt_btn = QPushButton("Save as GT")
        self._save_gt_btn.setToolTip(
            "Save as ground truth file for algorithm evaluation.\n"
            "Records correct answers for testing/improving the algorithm."
        )
        self._save_gt_btn.setStyleSheet("background-color: #2d5016;")
        self._save_gt_btn.clicked.connect(self._on_save_gt)
        buttons_layout.addWidget(self._save_gt_btn)

        layout.addLayout(buttons_layout)

    def set_items(self, items: List[ComparisonItem], gt_exists: bool = False):
        """
        Set the comparison items to display.

        Args:
            items: List of ComparisonItem objects
            gt_exists: Whether a GT file exists (affects editability)
        """
        self._items = items
        self._gt_exists = gt_exists
        self._editable = not gt_exists

        # Update GT header
        if gt_exists:
            self._gt_header.setText("Ground Truth (from file)")
            self._gt_header.setStyleSheet("color: #4a9; font-size: 10px;")
        else:
            self._gt_header.setText("(No GT file - editing algo copy)")
            self._gt_header.setStyleSheet("color: #888; font-size: 10px;")

        self._update_lists()
        self._update_diff()

    def _update_lists(self):
        """Update both list widgets with current items."""
        self._algo_list.clear()
        self._gt_list.clear()

        for item in self._items:
            # Algo list
            algo_text = f"{item.label}: {self._format_value(item.algo_value)}"
            algo_item = QListWidgetItem(algo_text)
            algo_item.setData(Qt.UserRole, item.index)
            self._algo_list.addItem(algo_item)

            # GT list
            gt_text = f"{item.label}: {self._format_value(item.gt_value)}"
            gt_item = QListWidgetItem(gt_text)
            gt_item.setData(Qt.UserRole, item.index)

            # Highlight differences
            if item.algo_value != item.gt_value:
                if item.diff is not None:
                    # Show diff indicator - handle both numeric and string diffs
                    if isinstance(item.diff, (int, float)):
                        sign = "+" if item.diff > 0 else ""
                        gt_text = f"{item.label}: {self._format_value(item.gt_value)} ({sign}{item.diff})"
                    else:
                        # String diff (e.g., "MISMATCH" for outcomes)
                        gt_text = f"{item.label}: {self._format_value(item.gt_value)} (changed)"
                    gt_item.setText(gt_text)
                gt_item.setBackground(QBrush(QColor(80, 60, 30)))
            else:
                gt_item.setForeground(QBrush(QColor(100, 180, 100)))

            self._gt_list.addItem(gt_item)

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _update_diff(self):
        """Update the diff summary label."""
        if not self._items:
            self._diff_label.setText("No comparison data")
            self._diff_label.setStyleSheet("color: #888; font-weight: bold;")
            return

        total = len(self._items)
        differing = sum(1 for item in self._items if item.algo_value != item.gt_value)

        if differing == 0:
            self._diff_label.setText(f"All {total} items match")
            self._diff_label.setStyleSheet("color: #4a9; font-weight: bold;")
        else:
            pct = (total - differing) / total * 100
            self._diff_label.setText(f"{differing} of {total} differ ({pct:.0f}% match)")
            self._diff_label.setStyleSheet("color: #f80; font-weight: bold;")

    def _on_algo_selection(self, row: int):
        """Sync GT list when algo selection changes."""
        if row >= 0:
            self._gt_list.setCurrentRow(row)
            if row < len(self._items):
                self.item_selected.emit(self._items[row].index)

    def _on_gt_selection(self, row: int):
        """Sync algo list when GT selection changes."""
        if row >= 0:
            self._algo_list.setCurrentRow(row)
            if row < len(self._items):
                self.item_selected.emit(self._items[row].index)

    def _on_save_validation(self):
        """Handle Save Validation button click."""
        self.validation_saved.emit()

    def _on_save_gt(self):
        """Handle Save as GT button click."""
        self.gt_saved.emit()

    def update_gt_value(self, index: int, new_value: Any):
        """
        Update a GT value (called when user edits in the main widget).

        Args:
            index: Item index
            new_value: New value
        """
        for item in self._items:
            if item.index == index:
                item.gt_value = new_value
                if isinstance(item.algo_value, (int, float)) and isinstance(new_value, (int, float)):
                    item.diff = new_value - item.algo_value
                break
        self._update_lists()
        self._update_diff()

    def get_gt_values(self) -> dict:
        """Get all current GT values as a dict {index: value}."""
        return {item.index: item.gt_value for item in self._items}

    def select_item(self, index: int):
        """Select an item by its index."""
        for i, item in enumerate(self._items):
            if item.index == index:
                self._algo_list.setCurrentRow(i)
                self._gt_list.setCurrentRow(i)
                break


def create_boundary_comparison(
    algo_boundaries: List[int],
    gt_boundaries: Optional[List[int]] = None
) -> List[ComparisonItem]:
    """
    Create ComparisonItems for segmentation boundaries.

    Args:
        algo_boundaries: Algorithm-detected boundaries
        gt_boundaries: Ground truth boundaries (or None to use algo as base)

    Returns:
        List of ComparisonItem objects
    """
    if gt_boundaries is None:
        gt_boundaries = algo_boundaries.copy()

    items = []
    for i, (algo_b, gt_b) in enumerate(zip(algo_boundaries, gt_boundaries)):
        if i == 0:
            label = "B1 (→SA1)"
        elif i == len(algo_boundaries) - 1:
            label = f"B{i+1} (SA{i}→)"
        else:
            label = f"B{i+1} (SA{i}→{i+1})"

        items.append(ComparisonItem(
            index=i,
            algo_value=algo_b,
            gt_value=gt_b,
            label=label,
            diff=gt_b - algo_b if gt_b != algo_b else None
        ))

    return items


def create_reach_comparison(
    algo_reaches: List[dict],
    gt_reaches: Optional[List[dict]] = None,
    segment_num: int = 0
) -> List[ComparisonItem]:
    """
    Create ComparisonItems for reach detection.

    Args:
        algo_reaches: Algorithm-detected reaches for a segment
        gt_reaches: Ground truth reaches (or None to use algo as base)
        segment_num: Segment number for labeling

    Returns:
        List of ComparisonItem objects
    """
    if gt_reaches is None:
        gt_reaches = algo_reaches.copy()

    items = []
    max_reaches = max(len(algo_reaches), len(gt_reaches))

    for i in range(max_reaches):
        algo_r = algo_reaches[i] if i < len(algo_reaches) else None
        gt_r = gt_reaches[i] if i < len(gt_reaches) else None

        label = f"Seg{segment_num} R{i+1}"

        if algo_r and gt_r:
            algo_val = f"{algo_r.get('start_frame', '?')}-{algo_r.get('end_frame', '?')}"
            gt_val = f"{gt_r.get('start_frame', '?')}-{gt_r.get('end_frame', '?')}"
            diff = None
        elif algo_r:
            algo_val = f"{algo_r.get('start_frame', '?')}-{algo_r.get('end_frame', '?')}"
            gt_val = "(missing)"
            diff = None
        else:
            algo_val = "(missing)"
            gt_val = f"{gt_r.get('start_frame', '?')}-{gt_r.get('end_frame', '?')}"
            diff = None

        items.append(ComparisonItem(
            index=i,
            algo_value=algo_val,
            gt_value=gt_val,
            label=label,
            diff=diff
        ))

    return items


def create_outcome_comparison(
    algo_outcomes: List[dict],
    gt_outcomes: Optional[List[dict]] = None
) -> List[ComparisonItem]:
    """
    Create ComparisonItems for outcome classification.

    Args:
        algo_outcomes: Algorithm-detected outcomes
        gt_outcomes: Ground truth outcomes (or None to use algo as base)

    Returns:
        List of ComparisonItem objects
    """
    if gt_outcomes is None:
        gt_outcomes = algo_outcomes.copy()

    items = []
    for i, (algo_o, gt_o) in enumerate(zip(algo_outcomes, gt_outcomes)):
        label = f"Seg {algo_o.get('segment_num', i+1)}"

        algo_val = algo_o.get('outcome', 'unknown')
        gt_val = gt_o.get('outcome', 'unknown')

        items.append(ComparisonItem(
            index=i,
            algo_value=algo_val,
            gt_value=gt_val,
            label=label,
            diff="MISMATCH" if algo_val != gt_val else None
        ))

    return items
