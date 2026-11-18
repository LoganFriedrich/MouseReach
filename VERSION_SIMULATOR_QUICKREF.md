# Version Simulator - Quick Reference Card

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

## TL;DR

Simulate how OLD (pre-v8) reach filter versions would score against ground-truth reach files.

## One-Line Usage

```bash
python -m mousereach.eval.version_simulator Processing Processing
```

## Key Files

| File | Purpose |
|------|---------|
| `src/mousereach/eval/version_simulator.py` | Main module |
| `src/mousereach/eval/test_version_simulator.py` | Test suite |
| `src/mousereach/eval/USAGE_VERSION_SIMULATOR.md` | Detailed usage guide |
| `VERSION_SIMULATOR_README.md` | Overview and examples |

## The Bug (v3.3.0)

```python
# BAD CODE:
reaches = [r for r in reaches if r.max_extent_pixels >= 0]
```

**Impact**: The simulated v3.3.0 filter (extent >= 0) drops every reach with negative extent.

**Why**: Negative extent values are VALID (hand approaches but doesn't cross slit)

## The Fix (v3.4.0)

```python
# GOOD CODE:
# No extent filtering - preserve all reaches
```

**Result**: reaches with negative extent are kept.

## Version Summary

| Version | Filter |
|---------|--------|
| v0.5.0 | none |
| v1.0.0 | duration >= 3 |
| v1.5.0 | duration >= 3 |
| v2.0.0 | duration >= 3, extent > 0 |
| v2.5.0 | duration >= 3, extent > 0 |
| v3.0.0 | extent >= 5, duration >= 10 |
| v3.3.0 | duration >= 2, extent >= 0 (the bug) |
| v3.4.0 | duration >= 2 |

## Quick Start

### CLI
```bash
# Run all simulations
python -m mousereach.eval.version_simulator Processing Processing

# Generate report (runs ALL versions at the default tolerance; --versions and --tolerance are ignored with --output)
python -m mousereach.eval.version_simulator Processing Processing --output report.txt

# Specific versions
python -m mousereach.eval.version_simulator Processing Processing --versions v3.3.0 v3.4.0
```

### Python
```python
from mousereach.eval import evaluate_all_versions, print_version_comparison

# Run simulations
results = evaluate_all_versions("Processing", "Processing")

# Print table
print_version_comparison(results)

# Check specific version
print(f"v3.3.0 recall: {results['v3.3.0']['recall']:.1%}")
print(f"v3.4.0 recall: {results['v3.4.0']['recall']:.1%}")
```

## Expected Output

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

## Testing

```bash
cd /path/to/MouseReach\src\mousereach\eval
python test_version_simulator.py
```

## Custom Version

```python
from mousereach.eval.version_simulator import VersionFilter, apply_version_filter

custom = VersionFilter(
    version="custom",
    philosophy="Custom threshold",
    entry_method="likelihood",
    exit_method="disappear",
    min_extent_pixels=3.0,
    min_duration_frames=5
)

filtered = apply_version_filter(reaches, custom)
```

## What It Does

1. Loads current `*_reaches.json` files
2. Applies old version filters to simulate what they would detect
3. Evaluates against `*_reach_ground_truth.json` files
4. Returns precision, recall, F1 for each version

## Why It Matters

- **Validates fixes**: Compares v3.4.0 with v3.3.0
- **Documents evolution**: Shows how algorithm improved over time
- **Informs research**: Helps choose appropriate extent thresholds

## More Info

- Full usage guide: `src/mousereach/eval/USAGE_VERSION_SIMULATOR.md`
- Overview: `VERSION_SIMULATOR_README.md`
