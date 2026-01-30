"""
Validation analysis - compare algorithm results with manual/ground truth scoring.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    """Metrics comparing algorithm to ground truth."""

    video_name: str
    n_segments: int

    # Outcome accuracy
    outcome_accuracy: float  # % segments with correct outcome
    outcome_confusion: Dict[str, Dict[str, int]]  # Confusion matrix

    # Reach linkage accuracy (for segments with outcomes)
    reach_linkage_accuracy: Optional[float] = None  # % with correct causal reach

    # Per-outcome accuracies
    retrieved_precision: Optional[float] = None
    retrieved_recall: Optional[float] = None
    displaced_precision: Optional[float] = None
    displaced_recall: Optional[float] = None
    untouched_precision: Optional[float] = None
    untouched_recall: Optional[float] = None


class ValidationAnalyzer:
    """Compare algorithm outputs with manual scoring and ground truth."""

    def __init__(self, base_dir: Path):
        """
        Initialize validator.

        Args:
            base_dir: Directory containing both algorithm and ground truth files
        """
        self.base_dir = Path(base_dir)

    def load_ground_truth(self, video_name: str) -> Optional[Dict]:
        """
        Load ground truth file for a video.

        Args:
            video_name: Video name without extension

        Returns:
            Ground truth data dict or None if not found
        """
        gt_path = self.base_dir / f"{video_name}_outcome_ground_truth.json"

        if not gt_path.exists():
            # Try legacy naming
            gt_path = self.base_dir / f"{video_name}_reach_outcomes_ground_truth.json"

        if not gt_path.exists():
            gt_path = self.base_dir / f"{video_name}_pellet_outcomes_ground_truth.json"

        if not gt_path.exists():
            return None

        with open(gt_path) as f:
            return json.load(f)

    def load_algorithm_output(self, video_name: str) -> Optional[Dict]:
        """
        Load algorithm output for a video.

        Args:
            video_name: Video name without extension

        Returns:
            Algorithm output dict or None if not found
        """
        algo_path = self.base_dir / f"{video_name}_pellet_outcomes.json"

        if not algo_path.exists():
            return None

        with open(algo_path) as f:
            return json.load(f)

    def validate_video(self, video_name: str) -> Optional[ValidationMetrics]:
        """
        Validate algorithm output against ground truth for one video.

        Args:
            video_name: Video name without extension

        Returns:
            ValidationMetrics or None if ground truth not available
        """
        # Load both files
        ground_truth = self.load_ground_truth(video_name)
        algorithm = self.load_algorithm_output(video_name)

        if ground_truth is None or algorithm is None:
            return None

        # Compare segment by segment
        gt_segments = ground_truth.get('segments', [])
        algo_segments = algorithm.get('segments', [])

        if len(gt_segments) != len(algo_segments):
            print(f"Warning: {video_name} has different segment counts (GT: {len(gt_segments)}, Algo: {len(algo_segments)})")
            return None

        # Outcome comparison
        n_correct_outcomes = 0
        confusion = defaultdict(lambda: defaultdict(int))

        # Per-outcome counts for precision/recall
        outcomes = ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched']
        true_positives = {o: 0 for o in outcomes}
        false_positives = {o: 0 for o in outcomes}
        false_negatives = {o: 0 for o in outcomes}

        # Reach linkage comparison
        n_linkage_comparable = 0
        n_correct_linkage = 0

        for gt_seg, algo_seg in zip(gt_segments, algo_segments):
            gt_outcome = gt_seg.get('outcome')
            algo_outcome = algo_seg.get('outcome')

            # Normalize outcome names
            gt_outcome = self._normalize_outcome(gt_outcome)
            algo_outcome = self._normalize_outcome(algo_outcome)

            # Confusion matrix
            confusion[gt_outcome][algo_outcome] += 1

            # Outcome accuracy
            if gt_outcome == algo_outcome:
                n_correct_outcomes += 1
                true_positives[gt_outcome] += 1
            else:
                false_positives[algo_outcome] += 1
                false_negatives[gt_outcome] += 1

            # Reach linkage comparison (only for segments with outcomes)
            if gt_outcome not in ['untouched', 'uncertain', 'no_pellet']:
                gt_reach_id = gt_seg.get('causal_reach_id')
                algo_reach_id = algo_seg.get('causal_reach_id')

                if gt_reach_id is not None and algo_reach_id is not None:
                    n_linkage_comparable += 1
                    if gt_reach_id == algo_reach_id:
                        n_correct_linkage += 1

        # Compute metrics
        outcome_accuracy = n_correct_outcomes / len(gt_segments) if gt_segments else 0
        reach_linkage_accuracy = n_correct_linkage / n_linkage_comparable if n_linkage_comparable > 0 else None

        # Precision/recall for retrieved outcomes
        retrieved_precision = self._precision(true_positives['retrieved'], false_positives['retrieved'])
        retrieved_recall = self._recall(true_positives['retrieved'], false_negatives['retrieved'])

        displaced_precision = self._precision(
            true_positives['displaced_sa'] + true_positives['displaced_outside'],
            false_positives['displaced_sa'] + false_positives['displaced_outside']
        )
        displaced_recall = self._recall(
            true_positives['displaced_sa'] + true_positives['displaced_outside'],
            false_negatives['displaced_sa'] + false_negatives['displaced_outside']
        )

        untouched_precision = self._precision(true_positives['untouched'], false_positives['untouched'])
        untouched_recall = self._recall(true_positives['untouched'], false_negatives['untouched'])

        return ValidationMetrics(
            video_name=video_name,
            n_segments=len(gt_segments),
            outcome_accuracy=outcome_accuracy,
            outcome_confusion=dict(confusion),
            reach_linkage_accuracy=reach_linkage_accuracy,
            retrieved_precision=retrieved_precision,
            retrieved_recall=retrieved_recall,
            displaced_precision=displaced_precision,
            displaced_recall=displaced_recall,
            untouched_precision=untouched_precision,
            untouched_recall=untouched_recall
        )

    def validate_all(self) -> List[ValidationMetrics]:
        """
        Validate all videos in directory that have ground truth.

        Returns:
            List of ValidationMetrics for all validated videos
        """
        results = []

        # Find all ground truth files (new and legacy naming)
        gt_files = list(self.base_dir.glob('*_outcome_ground_truth.json'))
        gt_files.extend(self.base_dir.glob('*_reach_outcomes_ground_truth.json'))
        gt_files.extend(self.base_dir.glob('*_pellet_outcomes_ground_truth.json'))

        for gt_file in gt_files:
            # Extract video name
            video_name = gt_file.stem.replace('_outcome_ground_truth', '')
            video_name = video_name.replace('_reach_outcomes_ground_truth', '')
            video_name = video_name.replace('_pellet_outcomes_ground_truth', '')

            metrics = self.validate_video(video_name)
            if metrics:
                results.append(metrics)

        return results

    def export_validation_summary(self, output_path: Path):
        """
        Export validation summary for all videos to CSV.

        Args:
            output_path: Output CSV path
        """
        results = self.validate_all()

        if not results:
            print("No ground truth files found for validation")
            return

        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'video_name', 'n_segments', 'outcome_accuracy',
                'reach_linkage_accuracy',
                'retrieved_precision', 'retrieved_recall',
                'displaced_precision', 'displaced_recall',
                'untouched_precision', 'untouched_recall'
            ])

            # Data rows
            for metrics in results:
                writer.writerow([
                    metrics.video_name,
                    metrics.n_segments,
                    f"{metrics.outcome_accuracy:.3f}",
                    f"{metrics.reach_linkage_accuracy:.3f}" if metrics.reach_linkage_accuracy else "",
                    f"{metrics.retrieved_precision:.3f}" if metrics.retrieved_precision else "",
                    f"{metrics.retrieved_recall:.3f}" if metrics.retrieved_recall else "",
                    f"{metrics.displaced_precision:.3f}" if metrics.displaced_precision else "",
                    f"{metrics.displaced_recall:.3f}" if metrics.displaced_recall else "",
                    f"{metrics.untouched_precision:.3f}" if metrics.untouched_precision else "",
                    f"{metrics.untouched_recall:.3f}" if metrics.untouched_recall else ""
                ])

        # Print summary
        mean_accuracy = np.mean([m.outcome_accuracy for m in results])
        print(f"\nValidation Summary:")
        print(f"  {len(results)} videos validated")
        print(f"  Mean outcome accuracy: {mean_accuracy:.1%}")
        print(f"  Exported to {output_path}")

    def _normalize_outcome(self, outcome: str) -> str:
        """Normalize outcome string for comparison."""
        if outcome is None:
            return 'uncertain'

        outcome = outcome.lower().strip()

        # Map variations to standard names
        if 'retrieved' in outcome or 'success' in outcome:
            return 'retrieved'
        elif 'displaced' in outcome:
            if 'outside' in outcome:
                return 'displaced_outside'
            return 'displaced_sa'
        elif 'untouched' in outcome or 'missed' in outcome:
            return 'untouched'
        elif 'no_pellet' in outcome:
            return 'no_pellet'
        else:
            return 'uncertain'

    def _precision(self, true_pos: int, false_pos: int) -> Optional[float]:
        """Compute precision."""
        total = true_pos + false_pos
        return true_pos / total if total > 0 else None

    def _recall(self, true_pos: int, false_neg: int) -> Optional[float]:
        """Compute recall."""
        total = true_pos + false_neg
        return true_pos / total if total > 0 else None


# Import for confusion matrix
from collections import defaultdict


class ManualScoreLoader:
    """Load manual pellet scores from Excel files (SharePoint format)."""

    def __init__(self, excel_path: Path, sheet_name: str = '3b_Manual_Tray'):
        """
        Initialize loader for manual score Excel file.

        Args:
            excel_path: Path to Excel file (e.g., Connectome_01_Animal_Tracking.xlsx)
            sheet_name: Sheet name containing manual scores (default: '3b_Manual_Tray')
        """
        self.excel_path = Path(excel_path)
        self.sheet_name = sheet_name
        self.df = None

        if self.excel_path.exists():
            self.df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)

    def get_manual_scores(self, video_name: str) -> Optional[List[str]]:
        """
        Get manual scores for a video.

        Manual score encoding (from SharePoint Excel):
            2 = retrieved
            1 = displaced
            0 = untouched
            NaN/blank = no score

        Args:
            video_name: Video name (e.g., '20250624_CNT0115_P2')

        Returns:
            List of 20 outcome strings or None if not found
        """
        if self.df is None:
            return None

        # Try to find row matching video name
        possible_cols = ['Video', 'video_name', 'Video Name', 'File', 'filename']

        row = None
        for col in possible_cols:
            if col in self.df.columns:
                matches = self.df[self.df[col].str.contains(video_name, na=False, case=False)]
                if len(matches) > 0:
                    row = matches.iloc[0]
                    break

        if row is None:
            return None

        # Extract scores for pellets 1-20
        outcomes = []

        for pellet_num in range(1, 21):
            # Try different column naming conventions
            score = None
            for col_name in [str(pellet_num), f'P{pellet_num}', f'Pellet {pellet_num}']:
                if col_name in row.index:
                    score = row[col_name]
                    break

            # Convert score to outcome string
            if pd.isna(score):
                outcomes.append('uncertain')
            elif score == 2 or score == '2':
                outcomes.append('retrieved')
            elif score == 1 or score == '1':
                outcomes.append('displaced_sa')
            elif score == 0 or score == '0':
                outcomes.append('untouched')
            else:
                outcomes.append('uncertain')

        return outcomes
