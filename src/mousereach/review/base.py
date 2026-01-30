"""
Base classes for Algo vs GT review functionality.

Provides a mixin class that review widgets can use to add:
- Split view (algo output on left, GT/editable on right)
- Diff summary showing differences
- Separate "Save Validation" and "Save as GT" buttons
"""

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QFrame, QSplitter, QListWidget,
    QListWidgetItem, QAbstractItemView
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor


@dataclass
class DiffItem:
    """A single difference between algo output and GT."""
    index: int  # Which item (boundary index, reach id, segment num)
    algo_value: Any  # Value from algorithm
    gt_value: Any  # Value from ground truth (or edited value)
    diff: Any = None  # Computed difference (e.g., frame delta)
    description: str = ""  # Human-readable description


@dataclass
class DiffSummary:
    """Summary of all differences between algo and GT."""
    total_items: int = 0
    matching_items: int = 0
    differing_items: int = 0
    diffs: List[DiffItem] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_items == 0:
            return 1.0
        return self.matching_items / self.total_items

    def summary_text(self) -> str:
        if self.total_items == 0:
            return "No data to compare"
        if self.differing_items == 0:
            return f"All {self.total_items} items match"
        return f"{self.differing_items} of {self.total_items} items differ ({self.accuracy:.1%} match)"


class AlgoGTReviewMixin:
    """
    Mixin that adds Algo vs GT comparison functionality to review widgets.

    Usage:
        class MyReviewWidget(QWidget, AlgoGTReviewMixin):
            def __init__(self, viewer):
                QWidget.__init__(self)
                AlgoGTReviewMixin.__init__(self)
                # ... setup UI ...
                self._init_algo_gt_panels()  # Call after UI setup

            # Implement abstract methods:
            def _get_algo_output_path(self) -> Path: ...
            def _get_gt_path(self) -> Path: ...
            def _load_algo_data(self) -> dict: ...
            def _load_gt_data(self) -> Optional[dict]: ...
            def _compute_diff(self) -> DiffSummary: ...
            def _format_algo_item(self, index, value) -> str: ...
            def _format_gt_item(self, index, value) -> str: ...
    """

    # Signals for algo/GT events
    validation_saved = Signal(Path)
    gt_saved = Signal(Path)

    def __init__(self):
        # Data storage
        self.algo_data: Optional[Dict] = None  # From algorithm output
        self.gt_data: Optional[Dict] = None  # From ground truth file (if exists)
        self.working_data: Optional[Dict] = None  # Editable copy

        # UI elements (set by _init_algo_gt_panels)
        self._algo_list: Optional[QListWidget] = None
        self._gt_list: Optional[QListWidget] = None
        self._diff_label: Optional[QLabel] = None
        self._save_validation_btn: Optional[QPushButton] = None
        self._save_gt_btn: Optional[QPushButton] = None

    def _init_algo_gt_panels(self, parent_layout: QVBoxLayout) -> QWidget:
        """
        Initialize the split view panels for algo vs GT comparison.

        Call this after setting up the main UI layout.

        Args:
            parent_layout: The layout to add the panels to

        Returns:
            The container widget for the split view
        """
        # Container for split view
        container = QGroupBox("Algorithm vs Ground Truth")
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        # Splitter for side-by-side view
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Algorithm output (read-only)
        algo_panel = QGroupBox("Algorithm Output")
        algo_layout = QVBoxLayout()
        algo_panel.setLayout(algo_layout)

        self._algo_list = QListWidget()
        self._algo_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._algo_list.setStyleSheet("QListWidget { background-color: #2a2a2a; }")
        algo_layout.addWidget(self._algo_list)

        splitter.addWidget(algo_panel)

        # Right panel: Ground Truth (editable if no GT exists)
        gt_panel = QGroupBox("Ground Truth")
        gt_layout = QVBoxLayout()
        gt_panel.setLayout(gt_layout)

        self._gt_header = QLabel("(No GT file - editing algo copy)")
        self._gt_header.setStyleSheet("color: #888; font-size: 10px;")
        gt_layout.addWidget(self._gt_header)

        self._gt_list = QListWidget()
        self._gt_list.setSelectionMode(QAbstractItemView.SingleSelection)
        gt_layout.addWidget(self._gt_list)

        splitter.addWidget(gt_panel)

        container_layout.addWidget(splitter)

        # Diff summary
        diff_frame = QFrame()
        diff_layout = QHBoxLayout()
        diff_frame.setLayout(diff_layout)

        self._diff_label = QLabel("No comparison yet")
        self._diff_label.setStyleSheet("font-weight: bold;")
        diff_layout.addWidget(self._diff_label)

        container_layout.addWidget(diff_frame)

        # Save buttons
        buttons_layout = QHBoxLayout()

        self._save_validation_btn = QPushButton("Save Validation")
        self._save_validation_btn.setToolTip(
            "Save corrections back to the algorithm output file.\n"
            "Use this to fix the algo output for downstream pipeline steps."
        )
        self._save_validation_btn.clicked.connect(self._on_save_validation)
        buttons_layout.addWidget(self._save_validation_btn)

        buttons_layout.addStretch()

        self._save_gt_btn = QPushButton("Save as GT")
        self._save_gt_btn.setToolTip(
            "Save as a ground truth file for algorithm evaluation.\n"
            "Use this to record the correct answers for testing/improving the algorithm."
        )
        self._save_gt_btn.setStyleSheet("background-color: #2d5016;")
        self._save_gt_btn.clicked.connect(self._on_save_gt)
        buttons_layout.addWidget(self._save_gt_btn)

        container_layout.addLayout(buttons_layout)

        parent_layout.addWidget(container)

        # Connect selection sync
        self._algo_list.currentRowChanged.connect(self._on_algo_selection_changed)
        self._gt_list.currentRowChanged.connect(self._on_gt_selection_changed)

        return container

    def _load_algo_and_gt(self, video_path: Path):
        """Load algorithm output and ground truth (if exists)."""
        # Load algo data
        self.algo_data = self._load_algo_data()

        # Load GT data (may be None)
        self.gt_data = self._load_gt_data()

        # Set up working data
        if self.gt_data is not None:
            # GT exists - show read-only
            self.working_data = None
            self._gt_header.setText("Ground Truth (read-only)")
            self._gt_header.setStyleSheet("color: #4a9; font-size: 10px;")
            self._gt_list.setEnabled(False)
        else:
            # No GT - create editable copy from algo
            import copy
            self.working_data = copy.deepcopy(self.algo_data)
            self._gt_header.setText("(No GT file - editing algo copy)")
            self._gt_header.setStyleSheet("color: #888; font-size: 10px;")
            self._gt_list.setEnabled(True)

        # Update displays
        self._update_algo_list()
        self._update_gt_list()
        self._update_diff_display()

    def _update_algo_list(self):
        """Update the algorithm output list display."""
        if self._algo_list is None or self.algo_data is None:
            return

        self._algo_list.clear()
        items = self._get_algo_items()
        for idx, value in items:
            item_text = self._format_algo_item(idx, value)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, idx)
            self._algo_list.addItem(item)

    def _update_gt_list(self):
        """Update the GT/working list display."""
        if self._gt_list is None:
            return

        self._gt_list.clear()
        data = self.gt_data if self.gt_data else self.working_data
        if data is None:
            return

        items = self._get_gt_items()
        for idx, value in items:
            item_text = self._format_gt_item(idx, value)
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, idx)

            # Color code differences
            if self.algo_data:
                algo_value = self._get_algo_value_at(idx)
                if algo_value != value:
                    item.setBackground(QColor(80, 60, 30))  # Highlight diff

            self._gt_list.addItem(item)

    def _update_diff_display(self):
        """Update the diff summary label."""
        if self._diff_label is None:
            return

        diff = self._compute_diff()
        self._diff_label.setText(diff.summary_text())

        if diff.differing_items > 0:
            self._diff_label.setStyleSheet("color: #f80; font-weight: bold;")
        else:
            self._diff_label.setStyleSheet("color: #4a9; font-weight: bold;")

    def _on_algo_selection_changed(self, row: int):
        """Sync GT list selection when algo selection changes."""
        if self._gt_list and row >= 0:
            self._gt_list.setCurrentRow(row)

    def _on_gt_selection_changed(self, row: int):
        """Sync algo list selection when GT selection changes."""
        if self._algo_list and row >= 0:
            self._algo_list.setCurrentRow(row)

    def _on_save_validation(self):
        """Save corrections back to the algorithm output file."""
        if not hasattr(self, 'video_path') or not self.video_path:
            return

        data_to_save = self.working_data if self.working_data else self.algo_data
        if data_to_save is None:
            return

        output_path = self._get_algo_output_path()
        self._save_validation_to_file(data_to_save, output_path)

        # Update index
        try:
            from mousereach.index import PipelineIndex
            from mousereach.config import get_video_id
            video_id = get_video_id(self.video_path.name)
            index = PipelineIndex()
            index.load()
            index.record_validation_changed(video_id, self._get_step_name(), "validated")
            index.save()
        except Exception:
            pass

        self.validation_saved.emit(self.video_path)

    def _on_save_gt(self):
        """Save as ground truth file."""
        if not hasattr(self, 'video_path') or not self.video_path:
            return

        data_to_save = self.working_data if self.working_data else self.algo_data
        if data_to_save is None:
            return

        gt_path = self._get_gt_path()
        self._save_gt_to_file(data_to_save, gt_path)

        # Update GT data and refresh display
        self.gt_data = data_to_save
        self.working_data = None
        self._gt_header.setText("Ground Truth (read-only)")
        self._gt_header.setStyleSheet("color: #4a9; font-size: 10px;")
        self._gt_list.setEnabled(False)
        self._update_gt_list()

        # Update index
        try:
            from mousereach.index import PipelineIndex
            from mousereach.config import get_video_id
            video_id = get_video_id(self.video_path.name)
            index = PipelineIndex()
            index.load()
            index.record_gt_created(video_id, self._get_step_name())
            index.save()
        except Exception:
            pass

        self.gt_saved.emit(self.video_path)

    # === Abstract methods to be implemented by subclasses ===

    @abstractmethod
    def _get_algo_output_path(self) -> Path:
        """Return path to the algorithm output file (e.g., _segments.json)."""
        pass

    @abstractmethod
    def _get_gt_path(self) -> Path:
        """Return path to the ground truth file."""
        pass

    @abstractmethod
    def _load_algo_data(self) -> Dict:
        """Load and return algorithm output data."""
        pass

    @abstractmethod
    def _load_gt_data(self) -> Optional[Dict]:
        """Load and return ground truth data (None if doesn't exist)."""
        pass

    @abstractmethod
    def _get_algo_items(self) -> List[tuple]:
        """Return list of (index, value) tuples from algo data."""
        pass

    @abstractmethod
    def _get_gt_items(self) -> List[tuple]:
        """Return list of (index, value) tuples from GT/working data."""
        pass

    @abstractmethod
    def _get_algo_value_at(self, index: int) -> Any:
        """Get algo value at given index for comparison."""
        pass

    @abstractmethod
    def _compute_diff(self) -> DiffSummary:
        """Compute and return diff summary between algo and GT."""
        pass

    @abstractmethod
    def _format_algo_item(self, index: int, value: Any) -> str:
        """Format algo item for display."""
        pass

    @abstractmethod
    def _format_gt_item(self, index: int, value: Any) -> str:
        """Format GT item for display."""
        pass

    @abstractmethod
    def _save_validation_to_file(self, data: Dict, path: Path):
        """Save validation data to file."""
        pass

    @abstractmethod
    def _save_gt_to_file(self, data: Dict, path: Path):
        """Save ground truth data to file."""
        pass

    @abstractmethod
    def _get_step_name(self) -> str:
        """Return step name for index updates ('seg', 'reach', 'outcome')."""
        pass
