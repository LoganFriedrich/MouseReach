"""Auto-update the Current Performance section of ALGORITHM_REFERENCE.md."""

import numpy as np
from datetime import datetime
from pathlib import Path

START_MARKER = "<!-- AUTO-GENERATED PERFORMANCE SECTION START -->"
END_MARKER = "<!-- AUTO-GENERATED PERFORMANCE SECTION END -->"


def generate_performance_section(results) -> str:
    """Generate the full performance section markdown from CorpusResults."""
    from .collect_results import OUTCOME_CLASSES

    today = datetime.now().strftime("%Y-%m-%d")
    n_seg = len(results.seg_results)
    n_reach = len(results.reach_results)
    n_skip = len(results.skipped_reach)
    n_out = len(results.outcome_results)
    n_boundaries = sum(r.n_gt for r in results.seg_results)
    n_matched_reaches = len(results.all_start_errors)
    n_outcomes = sum(r.n_gt for r in results.outcome_results)

    lines = []
    lines.append(START_MARKER)
    lines.append("## Current Performance")
    lines.append("")
    lines.append(
        f"Evaluated {today} against {n_seg} ground truth videos with human-verified "
        f"annotations. Run `python -m mousereach.eval.report_cli` to regenerate these "
        f"numbers and produce plots."
    )

    # --- Aggregate ---
    lines.append("")
    lines.append("### Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value | Corpus Size |")
    lines.append("|--------|-------|-------------|")

    if results.seg_results:
        mean_seg = np.mean([r.recall * 100 for r in results.seg_results])
        lines.append(f"| Segmentation boundary recall | {mean_seg:.1f}% | {n_seg} videos ({n_boundaries} boundaries) |")
    if results.reach_results:
        mean_p = np.mean([r.precision * 100 for r in results.reach_results])
        mean_r = np.mean([r.recall * 100 for r in results.reach_results])
        mean_f1 = np.mean([r.f1 * 100 for r in results.reach_results])
        lines.append(f"| Reach detection precision | {mean_p:.1f}% | {n_reach} videos ({n_matched_reaches:,} matched reaches) |")
        lines.append(f"| Reach detection recall | {mean_r:.1f}% | {n_reach} videos |")
        lines.append(f"| Reach detection F1 | {mean_f1:.1f}% | {n_reach} videos |")
    if results.outcome_results:
        mean_acc = np.mean([r.accuracy * 100 for r in results.outcome_results])
        lines.append(f"| Outcome classification accuracy | {mean_acc:.1f}% | {n_out} videos ({n_outcomes} outcomes) |")

    if results.skipped_reach:
        names = ", ".join(results.skipped_reach)
        lines.append("")
        lines.append(f"{n_skip} videos were excluded from reach evaluation because they lack reach ground truth annotations ({names}).")

    # --- Segmentation per-video ---
    lines.append("")
    lines.append("### Segmentation: Per-Video Boundary Recall")
    lines.append("")
    perfect = [r for r in results.seg_results if r.recall >= 1.0]
    imperfect = sorted([r for r in results.seg_results if r.recall < 1.0], key=lambda r: r.recall)
    lines.append(f"{len(perfect)} of {n_seg} videos achieved 100% boundary detection.")
    if imperfect:
        lines.append(f" The remaining {len(imperfect)} videos:")
        lines.append("")
        lines.append("| Video | Recall | Matched/Total |")
        lines.append("|-------|--------|---------------|")
        for r in imperfect:
            lines.append(f"| {r.video_name} | {r.recall*100:.1f}% | {r.n_matched}/{r.n_gt} |")
        lines.append(f"| All others ({len(perfect)} videos) | 100% | {perfect[0].n_gt}/{perfect[0].n_gt} |" if perfect else "")

    # Timing
    if results.all_boundary_errors:
        errs = results.all_boundary_errors
        exact = sum(1 for e in errs if e == 0)
        w1 = sum(1 for e in errs if abs(e) <= 1)
        w2 = sum(1 for e in errs if abs(e) <= 2)
        mean_e = np.mean(errs)
        std_e = np.std(errs)
        lines.append("")
        lines.append(f"**Boundary timing accuracy** ({len(errs)} matched boundaries):")
        lines.append("")
        lines.append("| Tolerance | Count | Percentage |")
        lines.append("|-----------|-------|------------|")
        lines.append(f"| Exact match (0 frames) | {exact} | {exact/len(errs)*100:.1f}% |")
        lines.append(f"| Within 1 frame | {w1} | {w1/len(errs)*100:.1f}% |")
        lines.append(f"| Within 2 frames | {w2} | {w2/len(errs)*100:.1f}% |")
        lines.append(f"| Mean error | {mean_e:+.2f} frames | {'Slightly early' if mean_e < 0 else 'Slightly late' if mean_e > 0 else 'Centered'} |")
        lines.append(f"| Standard deviation | {std_e:.2f} frames | |")
        if mean_e < 0:
            lines.append("")
            lines.append("The negative mean error indicates the algorithm tends to detect boundaries slightly before the human-marked frame — consistent with detecting the onset of tray motion rather than the moment the pellet reaches final position.")

    # --- Reach per-video ---
    lines.append("")
    lines.append("### Reach Detection: Per-Video Precision/Recall")
    lines.append("")
    lines.append("Performance varies substantially across videos. Some videos achieve perfect detection; others have high false positive rates.")
    lines.append("")
    lines.append("| Video | Precision | Recall | F1 | True Positives | False Positives | False Negatives |")
    lines.append("|-------|-----------|--------|-----|----------------|-----------------|-----------------|")
    for r in sorted(results.reach_results, key=lambda x: -x.f1):
        lines.append(f"| {r.video_name} | {r.precision*100:.0f}% | {r.recall*100:.0f}% | {r.f1*100:.0f}% | {r.tp} | {r.fp} | {r.fn} |")

    low_prec = [r for r in results.reach_results if r.precision < 0.5]
    if low_prec:
        names = ", ".join(r.video_name for r in low_prec)
        lines.append("")
        lines.append(f"**{len(low_prec)} problematic videos** ({names}) have precision below 50%, meaning the algorithm detects more false reaches than real ones. These videos likely have challenging DLC tracking conditions (hand flicker, frequent paw-at-slit without reaching).")

    # Timing
    if results.all_start_errors and results.all_end_errors:
        se = results.all_start_errors
        ee = results.all_end_errors
        se_exact = sum(1 for e in se if e == 0)
        se_w1 = sum(1 for e in se if abs(e) <= 1)
        se_w2 = sum(1 for e in se if abs(e) <= 2)
        ee_exact = sum(1 for e in ee if e == 0)
        ee_w1 = sum(1 for e in ee if abs(e) <= 1)
        ee_w2 = sum(1 for e in ee if abs(e) <= 2)
        lines.append("")
        lines.append(f"**Reach timing accuracy** ({len(se):,} matched reaches):")
        lines.append("")
        lines.append("| Metric | Start Frame | End Frame |")
        lines.append("|--------|-------------|-----------|")
        lines.append(f"| Exact match (0 frames) | {se_exact:,} ({se_exact/len(se)*100:.1f}%) | {ee_exact:,} ({ee_exact/len(ee)*100:.1f}%) |")
        lines.append(f"| Within 1 frame | {se_w1:,} ({se_w1/len(se)*100:.1f}%) | {ee_w1:,} ({ee_w1/len(ee)*100:.1f}%) |")
        lines.append(f"| Within 2 frames | {se_w2:,} ({se_w2/len(se)*100:.1f}%) | {ee_w2:,} ({ee_w2/len(ee)*100:.1f}%) |")
        lines.append(f"| Mean error | {np.mean(se):+.2f} frames | {np.mean(ee):+.2f} frames |")

        lines.append("")
        lines.append(f"**Start frames are highly accurate** — {se_exact/len(se)*100:.1f}% exact match. This is because reach start is well-defined (hand first appears above confidence threshold).")
        lines.append("")
        end_mean = np.mean(ee)
        lines.append(f"**End frames have a systematic {'early' if end_mean < 0 else 'late'} bias** — mean error of {end_mean:+.2f} frames, with only {ee_exact/len(ee)*100:.1f}% exact matches. The algorithm ends reaches ~{abs(round(end_mean))} frame{'s' if abs(round(end_mean)) != 1 else ''} before the human-marked end, consistent with DLC confidence dipping before the hand fully retracts. This is the most significant measurement concern because kinematic features (velocity, trajectory) are computed over the detected reach window.")

    # --- Outcome per-video ---
    lines.append("")
    lines.append("### Outcome Classification: Per-Video Accuracy")
    lines.append("")
    perfect_out = [r for r in results.outcome_results if r.accuracy >= 1.0]
    imperfect_out = sorted([r for r in results.outcome_results if r.accuracy < 1.0], key=lambda r: r.accuracy)
    lines.append(f"{len(perfect_out)} of {n_out} videos achieved 100% accuracy.")
    if imperfect_out:
        lines.append(f" The remaining {len(imperfect_out)} videos:")
        lines.append("")
        lines.append("| Video | Accuracy | Errors | Error Details |")
        lines.append("|-------|----------|--------|---------------|")
        for r in imperfect_out:
            n_err = len(r.misclassifications)
            details = "; ".join(
                f"{m['gt']}->{m['algo']} (seg {m['segment']})"
                for m in r.misclassifications[:5]
            )
            if n_err > 5:
                details += f"; +{n_err - 5} more"
            lines.append(f"| {r.video_name} | {r.accuracy*100:.0f}% | {n_err} | {details} |")

    # Confusion matrix
    cm = results.confusion_matrix
    present = [c for c in OUTCOME_CLASSES if sum(cm.get(c, {}).values()) > 0 or
               sum(cm.get(o, {}).get(c, 0) for o in OUTCOME_CLASSES) > 0]
    if present:
        total = sum(sum(cm.get(gt, {}).values()) for gt in present)
        lines.append("")
        lines.append(f"**Confusion matrix** ({total} total classifications):")
        lines.append("")
        header = "| GT \\\\ Algorithm | " + " | ".join(present) + " |"
        sep = "|" + "|".join(["---"] * (len(present) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        for gt_cls in present:
            cells = []
            for algo_cls in present:
                val = cm.get(gt_cls, {}).get(algo_cls, 0)
                cells.append(f"**{val}**" if gt_cls == algo_cls else str(val))
            lines.append(f"| **{gt_cls}** | " + " | ".join(cells) + " |")

    # --- Weakness summary ---
    lines.append("")
    lines.append("### Summary of Weaknesses")
    lines.append("")
    lines.append("| Issue | Severity | Affected Videos | Impact |")
    lines.append("|-------|----------|----------------|--------|")

    if low_prec:
        lines.append(f"| Reach false positives (precision < 50%) | High | {len(low_prec)} of {n_reach} | Corrupts kinematic analysis with non-reach data |")
    if results.all_end_errors:
        end_mean = np.mean(results.all_end_errors)
        if abs(end_mean) >= 0.5:
            lines.append(f"| Reach end frame {abs(end_mean):.1f}-frame {'early' if end_mean < 0 else 'late'} bias | Medium | All | Systematic bias in kinematic measurements |")

    # Count dominant misclassification patterns
    error_patterns = {}
    for r in results.outcome_results:
        for m in r.misclassifications:
            key = f"{m['gt']} -> {m['algo']}"
            error_patterns[key] = error_patterns.get(key, 0) + 1
    for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
        if count >= 3:
            n_affected = sum(1 for r in results.outcome_results
                           if any(f"{m['gt']} -> {m['algo']}" == pattern for m in r.misclassifications))
            lines.append(f"| {pattern} misclassification | Medium | {n_affected} of {n_out} | {count} incorrect classifications |")

    lines.append(END_MARKER)
    return "\n".join(lines)


def update_file(doc_path: Path, results) -> bool:
    """Replace the performance section in the doc with fresh data.

    Returns True if the file was updated, False if markers not found.
    """
    content = doc_path.read_text(encoding="utf-8")

    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)
    if start_idx == -1 or end_idx == -1:
        return False

    end_idx += len(END_MARKER)
    new_section = generate_performance_section(results)
    new_content = content[:start_idx] + new_section + content[end_idx:]
    doc_path.write_text(new_content, encoding="utf-8")
    return True


def main():
    """CLI entry point: update ALGORITHM_REFERENCE.md with current eval data."""
    from mousereach.config import require_processing_root
    from .collect_results import collect_all

    processing_dir = require_processing_root() / "Processing"
    project_root = require_processing_root().parent  # MouseReach/

    # Find the doc
    doc_path = project_root / "ALGORITHM_REFERENCE.md"
    if not doc_path.exists():
        # Try one level up
        doc_path = project_root.parent / "ALGORITHM_REFERENCE.md"
    if not doc_path.exists():
        print(f"Error: ALGORITHM_REFERENCE.md not found")
        return

    print(f"Collecting evaluation results from {processing_dir}...")
    results = collect_all(processing_dir)

    n_seg = len(results.seg_results)
    n_reach = len(results.reach_results)
    n_out = len(results.outcome_results)
    print(f"Corpus: {n_seg} seg, {n_reach} reach, {n_out} outcome")

    print(f"Updating {doc_path}...")
    success = update_file(doc_path, results)
    if success:
        print("Performance section updated successfully.")
    else:
        print("Error: Could not find AUTO-GENERATED markers in the document.")
        print("Expected markers:")
        print(f"  {START_MARKER}")
        print(f"  {END_MARKER}")


if __name__ == "__main__":
    main()
