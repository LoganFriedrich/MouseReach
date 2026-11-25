# Version Simulator Usage Guide

> **Historical tool -- read this first.** The versions this tool
> "simulates" (v0.5.0 to v3.4.0) are filters applied to reach files that already exist.
> None of them is the reach detector that runs today, which is the v8 trained classifier
> (`mousereach.reach.v8`, version 8.1.0; see `docs/ALGO_2_REACH_DETECTION.md`). The
> simulator replays only each version's duration and extent filters; the entry and exit
> methods listed for a version (nose engagement, retraction ...) are descriptions and are
> not replayed. Versions that filter on extent (v2.0.0, v2.5.0, v3.0.0, v3.3.0) stop with
> an error on reach files written by the current detector, which leaves extent empty (null).
>
> **WARNING -- run it only on a COPY of your reach files.** While it runs, it deletes each
> `{video}_reaches.json` in the reach folder you give it, puts a filtered copy in its
> place, and writes the original back from memory when that video is done. If it is
> stopped part-way (window closed, crash, power cut), the original is left filtered or
> missing. First copy the `*_reaches.json` and `*_reach_ground_truth.json` files into a
> scratch folder and point the simulator at that folder -- never at `Processing` or any
> other live pipeline folder. The examples below that say `Processing Processing` were
> written before this warning: put your scratch folder in place of `Processing`.
>
> What each simulated version keeps, as defined in `ALGORITHMIC_VERSIONS` in
> `src/mousereach/eval/version_simulator.py` (that definition is authoritative where the
> text below disagrees): v0.5.0 every reach; v1.0.0 duration >= 3 frames; v1.5.0
> duration >= 3; v2.0.0 duration >= 3 and extent > 0; v2.5.0 duration >= 3 and
> extent > 0; v3.0.0 duration >= 10 and extent >= 5 px; v3.3.0 duration >= 2 and
> extent >= 0; v3.4.0 duration >= 2. No version requires an apex.

## Overview

The version simulator simulates how OLD algorithm versions would have performed on the current ground truth data. This is useful for:
- Understanding the impact of algorithm changes
- Validating that fixes actually improved performance
- Documenting algorithm evolution

## Quick Start

**Before you start:** the simulator temporarily replaces each `{video}_reaches.json` in the reach folder with a filtered copy and writes the original back afterwards. Run it on a COPY of the reach and ground-truth files in a scratch folder, never on a live pipeline folder (see the warning at the top).

### Command Line

```bash
# Run all version simulations
cd /path/to/MouseReach
python -m mousereach.eval.version_simulator Processing Processing

# Generate detailed report
python -m mousereach.eval.version_simulator Processing Processing --output version_report.txt

# Evaluate specific versions only
python -m mousereach.eval.version_simulator Processing Processing --versions v3.3.0 v3.4.0
```

### Python API

```python
from mousereach.eval import evaluate_all_versions, print_version_comparison

# Run all version simulations
results = evaluate_all_versions(
    reaches_dir="Processing",
    gt_dir="Processing"
)

# Print comparison table
print_version_comparison(results)

# Access specific metrics
v3_3_metrics = results["v3.3.0"]
print(f"v3.3.0 Recall: {v3_3_metrics['recall']:.1%}")
print(f"v3.3.0 detected {v3_3_metrics['n_reaches_detected']} reaches")

# Generate report
from mousereach.eval import generate_version_report
report = generate_version_report("Processing", "Processing", "report.txt")
```

## Version History

(The simulator also defines v0.5.0 (no filter), v1.5.0 (adds nose
engagement; filter duration >= 3) and v2.5.0 (duration >= 3, extent > 0, end at
disappearance or retraction). Only the duration and extent filters are replayed.)

### v1.0.0 - Basic reach detection
- Basic threshold: duration >= 3 frames, no extent filter
- No version requires an apex
- **Issue**: High false negative rate (missed short reaches)

### v2.0.0 - Added extent filtering
- duration >= 3 frames and extent > 0 (positive only)
- Nose engagement was introduced earlier, in v1.5.0
- **Issue**: Still missed reaches that didn't cross slit

### v3.0.0 - Stricter filtering
- Required extent >= 5 pixels
- Added duration filter (min 10 frames)
- **Issue**: Reduced false positives but increased false negatives

### v3.3.0 - The problematic version ⚠️
- Filter: `reaches = [r for r in reaches if r.max_extent_pixels >= 0]`
- **BUG**: This dropped ~85% of valid reaches
- **Root cause**: Hand positions were measured relative to BOXR_x reference
- Negative extent values are VALID (hand approaches but doesn't cross slit)

### v3.4.0 - Fixed (simulated)
- duration >= 2 frames, no extent filter
- Superseded by the v8 detector, which does not compute extent (max_extent_pixels is null)

## Understanding the v3.3.0 Bug

The v3.3.0 bug was caused by a misunderstanding of what "extent" means:

```python
# v3.3.0 code (BAD):
reaches = [r for r in reaches if r.max_extent_pixels >= 0]
```

**Why this was wrong**:
1. Extent is measured as `hand_x - BOXR_x`
2. BOXR_x is the RIGHT edge of the slit opening
3. If hand approaches but doesn't cross BOXR_x, extent is negative
4. But these are VALID reaches! The mouse is attempting to reach

**Ground truth data showed**:
- Many human-annotated reaches have -2 to -15px extent
- These are scientifically meaningful "approach" reaches
- Mouse clearly attempting to reach, even if paw doesn't cross reference line

**The fix (v3.4.0)**:
- Remove the filter entirely
- Preserve extent value in each Reach object
- Let downstream analysis decide what extent threshold to use

## Example Output

The comparison prints a heading, then one block per version, then a summary
(numbers depend on your data):

```
REACH DETECTION ALGORITHM EVOLUTION
...
v3.4.0 <marker>
  Philosophy: ...
  Precision:  x%  |  Recall:  y%  |  F1: z
  Detected: n / m GT reaches
...
KEY INSIGHTS:
  Best Precision: ...
  Best Recall: ...
  LESSON LEARNED: ...
```

## Custom Analysis

You can also apply custom filters to simulate hypothetical versions:

```python
from mousereach.eval.version_simulator import VersionFilter, apply_version_filter

# Define a custom version
custom_filter = VersionFilter(
    version="v2.5.0-custom",
    philosophy="Custom: extent >= 2, duration >= 5",
    entry_method="likelihood",
    exit_method="disappear",
    min_extent_pixels=2.0,
    min_duration_frames=5
)

# Load current reaches
import json
with open("video_reaches.json") as f:
    data = json.load(f)

# Apply custom filter
for segment in data["segments"]:
    reaches = segment["reaches"]
    filtered = apply_version_filter(reaches, custom_filter)
    print(f"Segment {segment['segment_num']}: {len(reaches)} -> {len(filtered)} reaches")
```

## Key Functions

### `evaluate_all_versions(reaches_dir, gt_dir, tolerance=10)`
Runs all version simulations and returns metrics dict.

### `print_version_comparison(results)`
Pretty-prints comparison table showing precision/recall/F1 for each version.

### `generate_version_report(reaches_dir, gt_dir, output_path=None)`
Generates comprehensive report with version history and findings.

### `apply_version_filter(reaches, version)`
Applies historical version filters to reach list. Import it from `mousereach.eval.version_simulator`; `mousereach.eval` does not export it.

## How It Works

1. **Loads current algorithm output** (`*_reaches.json` files)
2. **Applies old version's filters** to simulate what it would have detected:
   - v1.0.0: Drops reaches with duration < 3
   - v2.0.0: Drops reaches with duration < 3 or extent <= 0
   - v3.0.0: Drops reaches with extent < 5 or duration < 10
   - v3.3.0: Drops reaches with duration < 2 or extent < 0 (the bug!)
   - v3.4.0: Drops reaches with duration < 2 (no extent filter)
   - The extent filters need a number in `max_extent_pixels`; reach files from the current detector have null extent and make those versions stop with a TypeError.
3. **Evaluates against ground truth** using standard ReachEvaluator
4. **Returns metrics** (precision, recall, F1, etc.)

## Testing

Run the test script:
```bash
cd <MouseReach folder>\src\mousereach\eval
python test_version_simulator.py
```

The v1.0.0 test currently fails, because it expects an extent > 5 filter that the code's v1.0.0 does not have; the run stops there. When every test passes, the output ends with:
```
Testing version_simulator module...
============================================================
[OK] All version filters defined
[OK] v1.0.0 filter works
[OK] v2.0.0 filter works
[OK] v3.0.0 filter works
[OK] v3.3.0 filter (bug) works as expected
[OK] v3.4.0 filter (current) works
[OK] Bug impact analysis complete

[OK] All tests passed!
```

## Troubleshooting

### "No ground truth files found"
- Check that your GT directory contains `*_reach_ground_truth.json` files
- Verify paths are correct

### "No successful evaluations"
Printed as `<version>: ERROR - No successful evaluations`. No `*_reaches.json` had a video id matching a `*_reach_ground_truth.json`, or every comparison failed.
- Check that your reaches directory contains `*_reaches.json` files
- Verify video IDs match between GT and reaches files

### "Import error"
Make sure you're in the correct directory:
```bash
cd /path/to/MouseReach
python -m mousereach.eval.version_simulator ...
```
