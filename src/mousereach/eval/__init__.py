"""
MouseReach Algorithm Evaluation Toolkit
==================================

Tools for comparing algorithm outputs against ground truth files,
identifying error patterns, and generating actionable recommendations.

Usage:
    from mousereach.eval import SegmentationEvaluator, ReachEvaluator, OutcomeEvaluator

    # Evaluate segmentation
    evaluator = SegmentationEvaluator(gt_dir, algo_dir)
    results = evaluator.evaluate_all()
    print(evaluator.generate_report())

    # Aggregate evaluation across all GT files
    from mousereach.eval import AggregateEvaluator
    agg = AggregateEvaluator()
    result = agg.evaluate_all()
    print(agg.generate_report())

    # Detect exception patterns
    from mousereach.eval import detect_all_patterns
    detector = detect_all_patterns()
    print(detector.generate_report())

    # Confidence analysis for non-GT videos
    from mousereach.eval import ConfidenceAnalyzer
    analyzer = ConfidenceAnalyzer()
    report = analyzer.analyze_all()

CLI:
    mousereach-eval --seg dev_SampleData/
    mousereach-eval --reach dev_SampleData/
    mousereach-eval --outcome dev_SampleData/
    mousereach-eval --all
"""

from .base import BaseEvaluator, EvalResult, ErrorCategory
from .seg_evaluator import SegmentationEvaluator, SegEvalResult
from .reach_evaluator import ReachEvaluator, ReachEvalResult
from .outcome_evaluator import OutcomeEvaluator, OutcomeEvalResult
from .aggregate_eval import (
    AggregateEvaluator,
    AggregateResult,
    FeatureMetrics,
    TimingBreakdown,
    HumanVerificationDetector,
    evaluate_all_gt,
    get_performance_summary,
)
from .exception_patterns import (
    ExceptionPatternDetector,
    ExceptionPattern,
    ErrorExample,
    detect_all_patterns,
)
from .confidence_analyzer import (
    ConfidenceAnalyzer,
    ConfidenceReport,
    VideoConfidence,
    get_priority_videos,
)
from .collect_results import collect_all, CorpusResults
from .plot_results import generate_all_plots
from .version_simulator import (
    evaluate_all_versions,
    print_version_comparison,
    generate_version_report,
    VERSION_FILTERS,
    VersionFilter,
    ALGORITHMIC_VERSIONS,
    AlgorithmicVersion,
)

__all__ = [
    # Base
    "BaseEvaluator",
    "EvalResult",
    "ErrorCategory",
    # Segmentation
    "SegmentationEvaluator",
    "SegEvalResult",
    # Reach
    "ReachEvaluator",
    "ReachEvalResult",
    # Outcome
    "OutcomeEvaluator",
    "OutcomeEvalResult",
    # Aggregate
    "AggregateEvaluator",
    "AggregateResult",
    "FeatureMetrics",
    "TimingBreakdown",
    "HumanVerificationDetector",
    "evaluate_all_gt",
    "get_performance_summary",
    # Exception patterns
    "ExceptionPatternDetector",
    "ExceptionPattern",
    "ErrorExample",
    "detect_all_patterns",
    # Confidence
    "ConfidenceAnalyzer",
    "ConfidenceReport",
    "VideoConfidence",
    "get_priority_videos",
    # Version simulation
    "evaluate_all_versions",
    "print_version_comparison",
    "generate_version_report",
    "VERSION_FILTERS",
    "VersionFilter",
    "ALGORITHMIC_VERSIONS",
    "AlgorithmicVersion",
    # Report generation
    "collect_all",
    "CorpusResults",
    "generate_all_plots",
]
