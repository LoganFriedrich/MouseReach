"""
Algorithm Performance Dashboard - Comprehensive napari widget.

Features:
- Performance tab: GT-verified metrics for all features
- History tab: Version comparison and trend graphs
- Exceptions tab: Error patterns with clickable examples
- Priorities tab: GT recommendations based on confidence analysis
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QFileDialog, QTabWidget, QHeaderView,
    QProgressBar, QSplitter, QFrame, QScrollArea, QGridLayout,
    QSizePolicy
)
from qtpy.QtCore import Qt, QTimer, Signal, QThread
from qtpy.QtGui import QFont, QColor, QPainter, QPen
import napari
from napari.utils.notifications import show_info, show_warning, show_error
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class TrendGraph(QWidget):
    """Simple trend line graph widget."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.data_points: List[Dict] = []  # [{date, value, version}]
        self.setMinimumHeight(120)
        self.setMinimumWidth(200)

    def set_data(self, points: List[Dict]):
        """Set data points for the graph."""
        self.data_points = points
        self.update()

    def paintEvent(self, event):
        """Draw the trend graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        if not self.data_points:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data")
            return

        # Margins
        margin_left = 50
        margin_right = 20
        margin_top = 25
        margin_bottom = 30

        plot_width = self.width() - margin_left - margin_right
        plot_height = self.height() - margin_top - margin_bottom

        if plot_width < 10 or plot_height < 10:
            return

        # Get value range
        values = [p.get('value', 0) for p in self.data_points]
        min_val = min(values) * 0.95
        max_val = max(values) * 1.05

        if max_val - min_val < 0.001:
            min_val -= 0.1
            max_val += 0.1

        # Draw title
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("", 9, QFont.Bold))
        painter.drawText(margin_left, 15, self.title)

        # Draw axes
        painter.setPen(QColor("#444444"))
        # Y axis
        painter.drawLine(margin_left, margin_top, margin_left, self.height() - margin_bottom)
        # X axis
        painter.drawLine(margin_left, self.height() - margin_bottom,
                        self.width() - margin_right, self.height() - margin_bottom)

        # Y axis labels
        painter.setPen(QColor("#888888"))
        painter.setFont(QFont("", 7))
        for i in range(5):
            y_val = min_val + (max_val - min_val) * (4 - i) / 4
            y_pos = margin_top + plot_height * i / 4
            painter.drawText(5, int(y_pos) + 4, f"{y_val:.0%}")

        # Plot line
        if len(self.data_points) > 1:
            pen = QPen(QColor("#00d4ff"), 2)
            painter.setPen(pen)

            points = []
            for i, point in enumerate(self.data_points):
                x = margin_left + (plot_width * i / (len(self.data_points) - 1))
                y = margin_top + plot_height * (1 - (point['value'] - min_val) / (max_val - min_val))
                points.append((x, y))

            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]),
                               int(points[i+1][0]), int(points[i+1][1]))

            # Draw points
            painter.setBrush(QColor("#00d4ff"))
            for x, y in points:
                painter.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)

        # X axis labels (dates)
        if self.data_points:
            painter.setPen(QColor("#888888"))
            first_date = self.data_points[0].get('date', '')[:10]
            last_date = self.data_points[-1].get('date', '')[:10]
            painter.drawText(margin_left, self.height() - 5, first_date)
            painter.drawText(self.width() - margin_right - 60, self.height() - 5, last_date)


class ProgressBarWidget(QWidget):
    """Custom progress bar for metrics display."""

    def __init__(self, label: str = "", value: float = 0, parent=None):
        super().__init__(parent)
        self.label = label
        self.value = value
        self.setMinimumHeight(25)

    def set_value(self, value: float, label: str = None):
        if label:
            self.label = label
        self.value = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#2a2a3e"))

        # Progress bar
        bar_width = int((self.width() - 100) * self.value)
        color = QColor("#00ff88") if self.value > 0.8 else QColor("#ffaa00") if self.value > 0.5 else QColor("#ff4444")
        painter.fillRect(100, 5, bar_width, self.height() - 10, color)

        # Label
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("", 9))
        painter.drawText(5, 17, self.label)

        # Value
        painter.drawText(self.width() - 45, 17, f"{self.value:.0%}")


class EvaluationWorker(QThread):
    """Background worker for running evaluations."""
    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, task: str):
        super().__init__()
        self.task = task

    def run(self):
        try:
            if self.task == "evaluate":
                from mousereach.eval.aggregate_eval import AggregateEvaluator
                evaluator = AggregateEvaluator()
                result = evaluator.evaluate_all(
                    progress_callback=lambda c, t, m: self.progress.emit(c, t, m)
                )
                self.finished.emit(result)

            elif self.task == "confidence":
                from mousereach.eval.confidence_analyzer import ConfidenceAnalyzer
                analyzer = ConfidenceAnalyzer()
                report = analyzer.analyze_all(
                    progress_callback=lambda c, t, m: self.progress.emit(c, t, m)
                )
                self.finished.emit(report)

            elif self.task == "patterns":
                from mousereach.eval.exception_patterns import detect_all_patterns
                detector = detect_all_patterns()
                self.finished.emit(detector)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class PerformanceViewerWidget(QWidget):
    """
    Comprehensive Algorithm Performance Dashboard.

    Tabs:
    1. Performance: GT-verified metrics for all features
    2. History: Version comparison and trend graphs
    3. Exceptions: Error patterns with clickable examples
    4. Priorities: GT recommendations
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Data cache
        self._eval_result = None
        self._confidence_report = None
        self._pattern_detector = None
        self._changelog = None
        self._patterns_list = []
        self._current_examples = []

        # Workers
        self._worker = None

        self._build_ui()
        QTimer.singleShot(100, self._initial_load)

    def _build_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("Algorithm Performance & Feedback")
        title.setFont(QFont("", 12, QFont.Bold))
        title.setStyleSheet("color: #00d4ff;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh All")
        self.refresh_btn.clicked.connect(self._refresh_all)
        header_layout.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self._export_report)
        header_layout.addWidget(self.export_btn)

        layout.addLayout(header_layout)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { padding: 8px 16px; }
            QTabBar::tab:selected { background: #2a2a3e; }
        """)

        # Create tabs
        self._create_performance_tab()
        self._create_history_tab()
        self._create_exceptions_tab()
        self._create_priorities_tab()

        layout.addWidget(self.tabs)

    def _create_performance_tab(self):
        """Create the Performance tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)

        # === SEGMENT BOUNDARIES ===
        seg_group = QGroupBox("SEGMENT BOUNDARIES")
        seg_group.setStyleSheet("QGroupBox { font-weight: bold; color: #aaa; }")
        seg_layout = QVBoxLayout(seg_group)

        self.seg_info_label = QLabel("GT files: 0 | Human-verified: 0")
        seg_layout.addWidget(self.seg_info_label)

        self.seg_bar = ProgressBarWidget("Accuracy", 0)
        seg_layout.addWidget(self.seg_bar)

        content_layout.addWidget(seg_group)

        # === REACHES ===
        reach_group = QGroupBox("REACHES")
        reach_group.setStyleSheet("QGroupBox { font-weight: bold; color: #aaa; }")
        reach_layout = QVBoxLayout(reach_group)

        self.reach_info_label = QLabel("GT files: 0 | Human-verified: 0 reaches")
        reach_layout.addWidget(self.reach_info_label)

        self.reach_detection_bar = ProgressBarWidget("Detection (Recall)", 0)
        reach_layout.addWidget(self.reach_detection_bar)

        self.reach_precision_bar = ProgressBarWidget("Precision", 0)
        reach_layout.addWidget(self.reach_precision_bar)

        # Timing breakdown
        timing_label = QLabel("Timing Accuracy:")
        timing_label.setStyleSheet("color: #888; margin-top: 5px;")
        reach_layout.addWidget(timing_label)

        self.timing_grid = QGridLayout()
        self.start_exact_label = QLabel("Start exact: --")
        self.start_1fr_label = QLabel("Start +/-1fr: --")
        self.start_2fr_label = QLabel("Start +/-2fr: --")
        self.end_exact_label = QLabel("End exact: --")
        self.end_1fr_label = QLabel("End +/-1fr: --")
        self.end_2fr_label = QLabel("End +/-2fr: --")

        self.timing_grid.addWidget(self.start_exact_label, 0, 0)
        self.timing_grid.addWidget(self.start_1fr_label, 0, 1)
        self.timing_grid.addWidget(self.start_2fr_label, 0, 2)
        self.timing_grid.addWidget(self.end_exact_label, 1, 0)
        self.timing_grid.addWidget(self.end_1fr_label, 1, 1)
        self.timing_grid.addWidget(self.end_2fr_label, 1, 2)

        reach_layout.addLayout(self.timing_grid)
        content_layout.addWidget(reach_group)

        # === OUTCOMES ===
        outcome_group = QGroupBox("OUTCOMES")
        outcome_group.setStyleSheet("QGroupBox { font-weight: bold; color: #aaa; }")
        outcome_layout = QVBoxLayout(outcome_group)

        self.outcome_info_label = QLabel("GT files: 0 | Human-verified: 0 segments")
        outcome_layout.addWidget(self.outcome_info_label)

        self.outcome_accuracy_bar = ProgressBarWidget("Classification", 0)
        outcome_layout.addWidget(self.outcome_accuracy_bar)

        # Confusion breakdown (text)
        self.confusion_text = QLabel("Confusion breakdown: --")
        self.confusion_text.setStyleSheet("color: #888;")
        self.confusion_text.setWordWrap(True)
        outcome_layout.addWidget(self.confusion_text)

        content_layout.addWidget(outcome_group)

        # === NON-GT CONFIDENCE ===
        conf_group = QGroupBox("NON-GT VIDEO CONFIDENCE")
        conf_group.setStyleSheet("QGroupBox { font-weight: bold; color: #aaa; }")
        conf_layout = QVBoxLayout(conf_group)

        self.conf_summary_label = QLabel("Loading...")
        conf_layout.addWidget(self.conf_summary_label)

        self.conf_distribution_label = QLabel("")
        self.conf_distribution_label.setWordWrap(True)
        conf_layout.addWidget(self.conf_distribution_label)

        content_layout.addWidget(conf_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        self.tabs.addTab(tab, "Performance")

    def _create_history_tab(self):
        """Create the History tab with trend graphs."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Version comparison selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Compare:"))

        self.version_combo_a = QComboBox()
        self.version_combo_a.setMinimumWidth(100)
        selector_layout.addWidget(self.version_combo_a)

        selector_layout.addWidget(QLabel("vs"))

        self.version_combo_b = QComboBox()
        self.version_combo_b.setMinimumWidth(100)
        selector_layout.addWidget(self.version_combo_b)

        compare_btn = QPushButton("Compare")
        compare_btn.clicked.connect(self._compare_versions)
        selector_layout.addWidget(compare_btn)

        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        # Trend graphs
        graphs_layout = QVBoxLayout()

        self.recall_graph = TrendGraph("Reach Recall Over Time")
        graphs_layout.addWidget(self.recall_graph)

        self.precision_graph = TrendGraph("Reach Precision Over Time")
        graphs_layout.addWidget(self.precision_graph)

        self.outcome_graph = TrendGraph("Outcome Accuracy Over Time")
        graphs_layout.addWidget(self.outcome_graph)

        layout.addLayout(graphs_layout)

        # Changelog text
        changelog_group = QGroupBox("Change History")
        changelog_layout = QVBoxLayout(changelog_group)

        self.changelog_text = QTextEdit()
        self.changelog_text.setReadOnly(True)
        self.changelog_text.setFont(QFont("Consolas", 9))
        self.changelog_text.setMaximumHeight(200)
        changelog_layout.addWidget(self.changelog_text)

        layout.addWidget(changelog_group)

        self.tabs.addTab(tab, "History")

    def _create_exceptions_tab(self):
        """Create the Exceptions tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Pattern list
        self.pattern_table = QTableWidget()
        self.pattern_table.setColumnCount(5)
        self.pattern_table.setHorizontalHeaderLabels([
            "Severity", "Pattern", "Count", "Videos", "Description"
        ])
        self.pattern_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pattern_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pattern_table.itemSelectionChanged.connect(self._on_pattern_selected)
        layout.addWidget(self.pattern_table)

        # Pattern details
        details_group = QGroupBox("Pattern Details")
        details_layout = QVBoxLayout(details_group)

        self.pattern_detail_text = QTextEdit()
        self.pattern_detail_text.setReadOnly(True)
        self.pattern_detail_text.setFont(QFont("Consolas", 9))
        self.pattern_detail_text.setMaximumHeight(150)
        details_layout.addWidget(self.pattern_detail_text)

        # Examples table
        self.examples_table = QTableWidget()
        self.examples_table.setColumnCount(4)
        self.examples_table.setHorizontalHeaderLabels([
            "Video", "Segment", "GT Value", "Algo Value"
        ])
        self.examples_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.examples_table.setMaximumHeight(150)
        self.examples_table.doubleClicked.connect(self._open_example)
        details_layout.addWidget(self.examples_table)

        layout.addWidget(details_group)

        self.tabs.addTab(tab, "Exceptions")

    def _create_priorities_tab(self):
        """Create the Priorities tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Priority categories
        priority_text = QLabel(
            "Based on error patterns, missing GT, and low-confidence detections:"
        )
        priority_text.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(priority_text)

        # High priority
        high_group = QGroupBox("HIGH PRIORITY")
        high_group.setStyleSheet("QGroupBox { color: #ff6666; font-weight: bold; }")
        high_layout = QVBoxLayout(high_group)
        self.high_priority_list = QTextEdit()
        self.high_priority_list.setReadOnly(True)
        self.high_priority_list.setMaximumHeight(120)
        high_layout.addWidget(self.high_priority_list)
        layout.addWidget(high_group)

        # Medium priority
        med_group = QGroupBox("MEDIUM PRIORITY")
        med_group.setStyleSheet("QGroupBox { color: #ffaa00; font-weight: bold; }")
        med_layout = QVBoxLayout(med_group)
        self.med_priority_list = QTextEdit()
        self.med_priority_list.setReadOnly(True)
        self.med_priority_list.setMaximumHeight(120)
        med_layout.addWidget(self.med_priority_list)
        layout.addWidget(med_group)

        # Strengths
        strengths_group = QGroupBox("CURRENT STRENGTHS")
        strengths_group.setStyleSheet("QGroupBox { color: #00ff88; font-weight: bold; }")
        strengths_layout = QVBoxLayout(strengths_group)
        self.strengths_list = QTextEdit()
        self.strengths_list.setReadOnly(True)
        self.strengths_list.setMaximumHeight(100)
        strengths_layout.addWidget(self.strengths_list)
        layout.addWidget(strengths_group)

        layout.addStretch()

        self.tabs.addTab(tab, "Priorities")

    def _initial_load(self):
        """Load initial data."""
        self._load_changelog()
        self._refresh_all()

    def _load_changelog(self):
        """Load changelog data."""
        try:
            from .changelog import get_changelog
            self._changelog = get_changelog()
            self._update_history_tab()
        except Exception as e:
            print(f"Error loading changelog: {e}")

    def _refresh_all(self):
        """Refresh all data."""
        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start evaluation worker
        self._worker = EvaluationWorker("evaluate")
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_eval_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current, total, message):
        """Handle progress updates."""
        percent = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{message} ({current}/{total})")

    def _on_eval_finished(self, result):
        """Handle evaluation completion."""
        self._eval_result = result
        self._update_performance_tab()

        # Start confidence analysis
        self._worker = EvaluationWorker("confidence")
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_confidence_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_confidence_finished(self, report):
        """Handle confidence analysis completion."""
        self._confidence_report = report
        self._update_confidence_section()

        # Start pattern detection
        self._worker = EvaluationWorker("patterns")
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_patterns_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_patterns_finished(self, detector):
        """Handle pattern detection completion."""
        self._pattern_detector = detector
        self._update_exceptions_tab()
        self._update_priorities_tab()

        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)

    def _on_error(self, message):
        """Handle worker errors."""
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)
        show_error(f"Error: {message}")
        print(f"Worker error: {message}")

    def _update_performance_tab(self):
        """Update performance tab with evaluation results."""
        if not self._eval_result:
            return

        result = self._eval_result

        # Segments
        if result.segments:
            s = result.segments
            self.seg_info_label.setText(
                f"GT files: {s.n_gt_files} | Human-verified: {s.n_human_verified}"
            )
            self.seg_bar.set_value(s.f1, "Accuracy")
        else:
            self.seg_info_label.setText("GT files: 0 - HIGH PRIORITY")
            self.seg_bar.set_value(0, "No data")

        # Reaches
        if result.reaches:
            r = result.reaches
            self.reach_info_label.setText(
                f"GT files: {r.n_gt_files} | Human-verified: {r.n_human_verified} reaches"
            )
            self.reach_detection_bar.set_value(r.recall, "Detection (Recall)")
            self.reach_precision_bar.set_value(r.precision, "Precision")

            # Timing breakdown
            if r.start_timing:
                self.start_exact_label.setText(f"Start exact: {r.start_timing.exact:.0%}")
                self.start_1fr_label.setText(f"+/-1fr: {r.start_timing.within_1:.0%}")
                self.start_2fr_label.setText(f"+/-2fr: {r.start_timing.within_2:.0%}")
            if r.end_timing:
                self.end_exact_label.setText(f"End exact: {r.end_timing.exact:.0%}")
                self.end_1fr_label.setText(f"+/-1fr: {r.end_timing.within_1:.0%}")
                self.end_2fr_label.setText(f"+/-2fr: {r.end_timing.within_2:.0%}")
        else:
            self.reach_info_label.setText("GT files: 0")
            self.reach_detection_bar.set_value(0, "No data")
            self.reach_precision_bar.set_value(0, "No data")

        # Outcomes
        if result.outcomes:
            o = result.outcomes
            self.outcome_info_label.setText(
                f"GT files: {o.n_gt_files} | Human-verified: {o.n_human_verified} segments"
            )
            self.outcome_accuracy_bar.set_value(o.f1, "Classification")

            # Confusion breakdown
            if o.error_categories:
                top_errors = sorted(o.error_categories.items(), key=lambda x: -x[1])[:3]
                error_text = ", ".join([f"{k}: {v}" for k, v in top_errors if v > 0])
                self.confusion_text.setText(f"Top errors: {error_text}" if error_text else "No errors detected")
            else:
                self.confusion_text.setText("--")
        else:
            self.outcome_info_label.setText("GT files: 0")
            self.outcome_accuracy_bar.set_value(0, "No data")

    def _update_confidence_section(self):
        """Update confidence section in performance tab."""
        if not self._confidence_report:
            return

        r = self._confidence_report

        if r.total_videos > 0:
            gt_pct = r.videos_with_gt * 100 // r.total_videos
        else:
            gt_pct = 0

        self.conf_summary_label.setText(
            f"Total videos: {r.total_videos} | With GT: {r.videos_with_gt} ({gt_pct}%) | Without GT: {r.videos_without_gt}"
        )

        self.conf_distribution_label.setText(
            f"Confidence distribution (non-GT): "
            f"High (>85%): {r.high_confidence} | "
            f"Medium (65-85%): {r.medium_confidence} | "
            f"Low (<65%): {r.low_confidence}"
        )

    def _update_history_tab(self):
        """Update history tab with changelog and trends."""
        if not self._changelog:
            return

        # Update version combos
        versions = self._changelog.get_all_versions()
        self.version_combo_a.clear()
        self.version_combo_b.clear()
        self.version_combo_a.addItems(versions)
        self.version_combo_b.addItems(versions)

        # Get trend data
        recall_data = self._changelog.get_trend_data('recall')
        precision_data = self._changelog.get_trend_data('precision')
        outcome_data = self._changelog.get_trend_data('accuracy')

        self.recall_graph.set_data(recall_data)
        self.precision_graph.set_data(precision_data)
        self.outcome_graph.set_data(outcome_data)

        # Changelog text
        report = self._changelog.generate_report(n_entries=10)
        self.changelog_text.setPlainText(report)

    def _update_exceptions_tab(self):
        """Update exceptions tab with detected patterns."""
        if not self._pattern_detector:
            return

        all_patterns = self._pattern_detector.get_priority_patterns(min_count=1)

        self.pattern_table.setRowCount(len(all_patterns))

        for i, pattern in enumerate(all_patterns):
            severity_item = QTableWidgetItem(pattern.severity.upper())
            if pattern.severity == "critical":
                severity_item.setBackground(QColor("#661111"))
            elif pattern.severity == "high":
                severity_item.setBackground(QColor("#663311"))
            elif pattern.severity == "medium":
                severity_item.setBackground(QColor("#554411"))

            self.pattern_table.setItem(i, 0, severity_item)
            self.pattern_table.setItem(i, 1, QTableWidgetItem(pattern.name))
            self.pattern_table.setItem(i, 2, QTableWidgetItem(str(pattern.count)))
            self.pattern_table.setItem(i, 3, QTableWidgetItem(str(len(pattern.affected_videos))))
            self.pattern_table.setItem(i, 4, QTableWidgetItem(pattern.description))

        # Store patterns for detail view
        self._patterns_list = all_patterns

    def _on_pattern_selected(self):
        """Handle pattern selection."""
        row = self.pattern_table.currentRow()
        if row < 0 or row >= len(self._patterns_list):
            return

        pattern = self._patterns_list[row]

        # Update detail text
        detail = f"Pattern: {pattern.name}\n\n"
        detail += f"WHY THIS HAPPENS:\n{pattern.explanation}\n\n"
        detail += f"POTENTIAL FIX:\n{pattern.potential_fix}\n"
        self.pattern_detail_text.setPlainText(detail)

        # Update examples table
        self.examples_table.setRowCount(len(pattern.examples))
        for i, ex in enumerate(pattern.examples):
            self.examples_table.setItem(i, 0, QTableWidgetItem(ex.video_id))
            self.examples_table.setItem(i, 1, QTableWidgetItem(str(ex.segment_num or "--")))
            self.examples_table.setItem(i, 2, QTableWidgetItem(str(ex.gt_value or "--")))
            self.examples_table.setItem(i, 3, QTableWidgetItem(str(ex.algo_value or "--")))

        # Store examples for opening
        self._current_examples = pattern.examples

    def _open_example(self, index):
        """Open selected example in GT tool."""
        row = index.row()
        if row < 0 or not hasattr(self, '_current_examples') or row >= len(self._current_examples):
            return

        example = self._current_examples[row]
        show_info(f"Open GT tool for {example.video_id}, segment {example.segment_num}")
        # Future enhancement: Actually open the GT tool widget and navigate to this video/segment
        # Tracked as potential UI improvement for cross-widget navigation

    def _update_priorities_tab(self):
        """Update priorities tab."""
        high_items = []
        med_items = []
        strengths = []

        # Check for missing GT
        if self._eval_result:
            if not self._eval_result.segments or self._eval_result.segments.n_gt_files == 0:
                high_items.append("Segment boundaries - NO GT exists yet")

        # Low confidence videos
        if self._confidence_report and self._confidence_report.priority_videos:
            n_low = len(self._confidence_report.priority_videos)
            high_items.append(f"Low-confidence videos ({n_low} flagged) - need GT")

        # From patterns
        if self._pattern_detector:
            priority_patterns = self._pattern_detector.get_priority_patterns(min_count=5)
            for pattern in priority_patterns:
                if pattern.severity in ["critical", "high"]:
                    high_items.append(f"{pattern.name} ({pattern.count} cases)")
                elif pattern.severity == "medium":
                    med_items.append(f"{pattern.name} ({pattern.count} cases)")

        # Determine strengths
        if self._eval_result:
            if self._eval_result.reaches and self._eval_result.reaches.recall > 0.9:
                strengths.append(f"Reach detection ({self._eval_result.reaches.recall:.0%} recall)")
            if self._eval_result.outcomes and self._eval_result.outcomes.f1 > 0.9:
                strengths.append(f"Outcome classification ({self._eval_result.outcomes.f1:.0%} accuracy)")

        # Update UI
        self.high_priority_list.setPlainText(
            "\n".join([f"- {item}" for item in high_items]) if high_items else "No high-priority items"
        )
        self.med_priority_list.setPlainText(
            "\n".join([f"- {item}" for item in med_items]) if med_items else "No medium-priority items"
        )
        self.strengths_list.setPlainText(
            "\n".join([f"- {item}" for item in strengths]) if strengths else "Run evaluation to identify strengths"
        )

    def _compare_versions(self):
        """Compare two versions."""
        if not self._changelog:
            return

        version_a = self.version_combo_a.currentText()
        version_b = self.version_combo_b.currentText()

        if not version_a or not version_b:
            show_warning("Select two versions to compare")
            return

        comparison = self._changelog.get_version_comparison(version_a, version_b)

        if not comparison:
            show_warning("No comparison data available for these versions")
            return

        # Show comparison in a message
        lines = [f"Comparison: {version_a} vs {version_b}\n"]
        for metric, data in comparison.items():
            delta = data['delta']
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {metric}: {data[version_a]:.3f} -> {data[version_b]:.3f} ({sign}{data['delta_percent']:.1f}%)")

        show_info("\n".join(lines))

    def _export_report(self):
        """Export comprehensive report."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", f"performance_report_{datetime.now().strftime('%Y%m%d')}.md",
            "Markdown (*.md)"
        )

        if not path:
            return

        lines = []
        lines.append("# Algorithm Performance Report")
        lines.append(f"Generated: {datetime.now().isoformat()}\n")

        # Performance summary
        if self._eval_result:
            try:
                from mousereach.eval.aggregate_eval import AggregateEvaluator
                evaluator = AggregateEvaluator()
                evaluator._last_result = self._eval_result
                lines.append(evaluator.generate_report())
                lines.append("\n")
            except Exception as e:
                lines.append(f"Error generating performance summary: {e}\n")

        # Changelog
        if self._changelog:
            lines.append(self._changelog.generate_report())
            lines.append("\n")

        # Exception patterns
        if self._pattern_detector:
            lines.append(self._pattern_detector.generate_report())
            lines.append("\n")

        # Confidence
        if self._confidence_report:
            try:
                from mousereach.eval.confidence_analyzer import ConfidenceAnalyzer
                analyzer = ConfidenceAnalyzer()
                lines.append(analyzer.generate_report())
            except Exception as e:
                lines.append(f"Error generating confidence report: {e}\n")

        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        show_info(f"Report saved to {path}")
