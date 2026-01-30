"""Collect evaluation results from unified GT files into structured data."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


BOUNDARY_TOLERANCE = 5
REACH_TOLERANCE = 10

OUTCOME_CLASSES = [
    "retrieved", "displaced_sa", "displaced_outside",
    "untouched", "no_pellet", "uncertain"
]


@dataclass
class VideoSegResult:
    video_name: str
    n_gt: int = 0
    n_algo: int = 0
    n_matched: int = 0
    recall: float = 0.0
    boundary_errors: List[int] = field(default_factory=list)


@dataclass
class VideoReachResult:
    video_name: str
    has_reach_gt: bool = False
    n_gt: int = 0
    n_algo: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    start_errors: List[int] = field(default_factory=list)
    end_errors: List[int] = field(default_factory=list)


@dataclass
class VideoOutcomeResult:
    video_name: str
    n_gt: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    misclassifications: List[Dict] = field(default_factory=list)


@dataclass
class CorpusResults:
    """All results across the entire GT corpus."""
    seg_results: List[VideoSegResult] = field(default_factory=list)
    reach_results: List[VideoReachResult] = field(default_factory=list)
    outcome_results: List[VideoOutcomeResult] = field(default_factory=list)
    skipped_reach: List[str] = field(default_factory=list)
    all_boundary_errors: List[int] = field(default_factory=list)
    all_start_errors: List[int] = field(default_factory=list)
    all_end_errors: List[int] = field(default_factory=list)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)


def collect_all(processing_dir: Path) -> CorpusResults:
    """Iterate over all unified GT files, compare against algo output."""
    processing_dir = Path(processing_dir)
    gt_files = sorted(processing_dir.glob("*_unified_ground_truth.json"))

    corpus = CorpusResults()
    # Initialize confusion matrix with zeros
    for gt_cls in OUTCOME_CLASSES:
        corpus.confusion_matrix[gt_cls] = {algo_cls: 0 for algo_cls in OUTCOME_CLASSES}

    for gt_file in gt_files:
        with open(gt_file) as f:
            gt_data = json.load(f)

        video_name = gt_data.get("video_name", gt_file.stem.replace("_unified_ground_truth", ""))

        # Segmentation
        seg_result = _collect_seg(gt_data, processing_dir, video_name)
        if seg_result:
            corpus.seg_results.append(seg_result)
            corpus.all_boundary_errors.extend(seg_result.boundary_errors)

        # Reaches
        reach_result = _collect_reaches(gt_data, processing_dir, video_name)
        if reach_result.has_reach_gt:
            corpus.reach_results.append(reach_result)
            corpus.all_start_errors.extend(reach_result.start_errors)
            corpus.all_end_errors.extend(reach_result.end_errors)
        else:
            corpus.skipped_reach.append(video_name)

        # Outcomes
        outcome_result, outcome_pairs = _collect_outcomes(gt_data, processing_dir, video_name)
        if outcome_result:
            corpus.outcome_results.append(outcome_result)
            # Update confusion matrix with ALL comparisons (correct + incorrect)
            for gt_cls, algo_cls in outcome_pairs:
                if gt_cls in corpus.confusion_matrix and algo_cls in corpus.confusion_matrix.get(gt_cls, {}):
                    corpus.confusion_matrix[gt_cls][algo_cls] += 1

    return corpus


def _collect_seg(gt_data: dict, processing_dir: Path, video_name: str) -> Optional[VideoSegResult]:
    """Compare GT boundaries against algo boundaries."""
    gt_boundaries = gt_data.get("segmentation", {}).get("boundaries", [])
    if not gt_boundaries:
        return None

    gt_frames = [b["frame"] for b in gt_boundaries]

    seg_file = processing_dir / f"{video_name}_segments.json"
    if not seg_file.exists():
        return None

    with open(seg_file) as f:
        seg_data = json.load(f)
    algo_frames = seg_data.get("boundaries", [])

    result = VideoSegResult(video_name=video_name, n_gt=len(gt_frames), n_algo=len(algo_frames))

    algo_used = set()
    for gt_frame in gt_frames:
        best_match = None
        best_error = None
        for j, algo_frame in enumerate(algo_frames):
            if j in algo_used:
                continue
            error = algo_frame - gt_frame
            if abs(error) <= BOUNDARY_TOLERANCE:
                if best_match is None or abs(error) < abs(best_error):
                    best_match = j
                    best_error = error
        if best_match is not None:
            algo_used.add(best_match)
            result.n_matched += 1
            result.boundary_errors.append(best_error)

    result.recall = result.n_matched / result.n_gt if result.n_gt > 0 else 0.0
    return result


def _collect_reaches(gt_data: dict, processing_dir: Path, video_name: str) -> VideoReachResult:
    """Compare GT reaches against algo reaches."""
    gt_reaches = gt_data.get("reaches", {}).get("reaches", [])

    result = VideoReachResult(video_name=video_name)

    if not gt_reaches:
        result.has_reach_gt = False
        return result

    result.has_reach_gt = True
    result.n_gt = len(gt_reaches)

    # Load algo reaches
    reaches_file = processing_dir / f"{video_name}_reaches.json"
    if not reaches_file.exists():
        result.fn = result.n_gt
        return result

    with open(reaches_file) as f:
        algo_data = json.load(f)

    algo_reaches = []
    for seg in algo_data.get("segments", []):
        algo_reaches.extend(seg.get("reaches", []))
    result.n_algo = len(algo_reaches)

    # Match GT to algo within tolerance
    gt_matched = set()
    algo_matched = set()

    for i, gt_r in enumerate(gt_reaches):
        gt_start = gt_r["start_frame"]
        gt_end = gt_r["end_frame"]

        for j, algo_r in enumerate(algo_reaches):
            if j in algo_matched:
                continue
            algo_start = algo_r["start_frame"]
            algo_end = algo_r["end_frame"]

            if abs(gt_start - algo_start) <= REACH_TOLERANCE and abs(gt_end - algo_end) <= REACH_TOLERANCE:
                gt_matched.add(i)
                algo_matched.add(j)
                result.start_errors.append(algo_start - gt_start)
                result.end_errors.append(algo_end - gt_end)
                break

    result.tp = len(gt_matched)
    result.fn = result.n_gt - result.tp
    result.fp = result.n_algo - result.tp
    result.precision = result.tp / (result.tp + result.fp) if (result.tp + result.fp) > 0 else 0.0
    result.recall = result.tp / (result.tp + result.fn) if (result.tp + result.fn) > 0 else 0.0
    result.f1 = (2 * result.precision * result.recall / (result.precision + result.recall)
                 if (result.precision + result.recall) > 0 else 0.0)
    return result


def _collect_outcomes(gt_data: dict, processing_dir: Path, video_name: str):
    """Compare GT outcomes against algo outcomes.

    Returns:
        Tuple of (VideoOutcomeResult or None, list of (gt_class, algo_class) pairs)
    """
    gt_outcomes = gt_data.get("outcomes", {}).get("segments", [])
    if not gt_outcomes:
        return None, []

    outcome_file = processing_dir / f"{video_name}_pellet_outcomes.json"
    if not outcome_file.exists():
        return None, []

    with open(outcome_file) as f:
        outcome_data = json.load(f)
    algo_outcomes = outcome_data.get("segments", [])

    # Index algo outcomes by segment_num
    algo_by_seg = {}
    for ao in algo_outcomes:
        seg = ao.get("segment_num") or ao.get("segment_index")
        if seg is not None:
            algo_by_seg[seg] = ao

    result = VideoOutcomeResult(video_name=video_name, n_gt=len(gt_outcomes))
    pairs = []  # (gt_class, algo_class) for confusion matrix

    for gt_o in gt_outcomes:
        gt_seg = gt_o.get("segment_num") or gt_o.get("segment_index")
        gt_class = gt_o.get("outcome")

        algo_o = algo_by_seg.get(gt_seg)
        if algo_o is None:
            continue

        algo_class = algo_o.get("outcome")
        pairs.append((gt_class, algo_class))

        if gt_class == algo_class:
            result.n_correct += 1
        else:
            result.misclassifications.append({
                "segment": gt_seg,
                "gt": gt_class,
                "algo": algo_class,
            })

    result.accuracy = result.n_correct / result.n_gt if result.n_gt > 0 else 0.0
    return result, pairs
