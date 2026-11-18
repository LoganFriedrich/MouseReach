# Version Simulator (historical tool) - Summary

> **Historical tool -- read this first (Update 2026-11-18).** The versions this tool
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

## What Was Created

A comprehensive version simulation module at `src/mousereach/eval/version_simulator.py` that simulates how OLD algorithm versions would perform on current ground truth data.

## Files Created

1. **`src/mousereach/eval/version_simulator.py`**
   - Main module with simulation logic
   - Eight simulated versions: v0.5.0, v1.0.0, v1.5.0, v2.0.0, v2.5.0, v3.0.0, v3.3.0, v3.4.0
   - Evaluation functions
   - Report generation
   - CLI entry point

2. **`src/mousereach/eval/USAGE_VERSION_SIMULATOR.md`**
   - Complete usage guide
   - Examples for CLI and Python API
   - Detailed version history
   - Troubleshooting tips

3. **`src/mousereach/eval/test_version_simulator.py`**
   - Unit tests for version filters
   - Bug impact analysis
   - Extent distribution demonstration

4. **`VERSION_SIMULATOR_README.md`** (this file)
   - Overview of what was created
   - Quick start guide

## Algorithm Versions Implemented

### v1.0.0 - Basic reach detection
- duration >= 3 frames; no extent filter

### v2.0.0 - Added extent filtering
- duration >= 3 frames and extent > 0 (no apex requirement)

### v3.0.0 - Stricter filtering
- extent >= 5 pixels
- duration >= 10 frames

### v3.3.0 - The problematic version ⚠️
- Filter: `extent >= 0`
- **BUG**: drops every reach with negative extent

### v3.4.0 - Fixed (simulated)
- duration >= 2 frames, no extent filter (simulated; the current detector is reach.v8 8.1.0)

## Quick Start

### Command Line

```bash
cd /path/to/MouseReach

# Run all version simulations
python -m mousereach.eval.version_simulator Processing Processing

# Generate detailed report
python -m mousereach.eval.version_simulator Processing Processing --output report.txt
```

### Python API

```python
from mousereach.eval import evaluate_all_versions, print_version_comparison

# Run simulations
results = evaluate_all_versions(
    reaches_dir="Processing",
    gt_dir="Processing"
)

# Print comparison table
print_version_comparison(results)

# Access metrics
print(f"v3.3.0 recall: {results['v3.3.0']['recall']:.1%}")
print(f"v3.4.0 recall: {results['v3.4.0']['recall']:.1%}")
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

## The v3.3.0 Bug Explained

```python
# v3.3.0 code (BAD):
reaches = [r for r in reaches if r.max_extent_pixels >= 0]
```

**Why this was wrong:**
- Extent = `hand_x - BOXR_x` (hand position relative to slit reference) under the pre-v8 detectors; the current detector does not compute extent (null)
- Negative extent = hand approaches but doesn't cross slit line
- But these are VALID reaches! Mouse is clearly attempting
- Human GT showed many valid reaches have -2 to -15px extent

**Impact:**
- All "approach" reaches were filtered out

**The fix (v3.4.0):**
- Remove filter entirely
- Preserve extent values for downstream analysis
- Let researchers filter by extent threshold as needed

## Testing

Run the test suite:
```bash
cd /path/to/MouseReach\src\mousereach\eval
python test_version_simulator.py
```

Expected output (Update 2026-11-18: the v1.0.0 test currently FAILS, because it expects an
extent > 5 filter that the code's v1.0.0 does not have; the run stops there):
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

## Integration with Existing Code

The module is fully integrated with the existing evaluation framework:
- Uses `ReachEvaluator` for standard metrics
- Compatible with existing `*_reaches.json` format
- Works with existing ground truth files
- Exported in `mousereach.eval.__init__.py`

## Use Cases

1. **Validate algorithm improvements**
   - Show v3.4.0 is better than v3.3.0
   - Quantify impact of bug fix

2. **Document algorithm evolution**
   - Track precision/recall changes over time
   - Understand tradeoffs in different versions

3. **Research analysis**
   - Compare different extent thresholds
   - Understand what "reach" definition works best

4. **Quality assurance**
   - Verify fixes don't introduce regressions
   - Test impact of proposed changes

## Future Extensions

The module is designed to be extensible:

```python
# Define custom version
from mousereach.eval.version_simulator import VersionFilter, apply_version_filter

custom = VersionFilter(
    version="v2.5.0-custom",
    philosophy="Custom: extent >= 2, duration >= 5",
    entry_method="likelihood",
    exit_method="disappear",
    min_extent_pixels=2.0,
    min_duration_frames=5
)

# Apply to data
filtered = apply_version_filter(reaches, custom)
```

## Contact

For questions or issues with the version simulator:
- Check `USAGE_VERSION_SIMULATOR.md` for detailed examples
- Run `test_version_simulator.py` to verify installation
- Review code comments in `version_simulator.py`
