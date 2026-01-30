# Version Simulator - Quick Reference Card

## TL;DR

Simulate how OLD algorithm versions would perform on current GT data to validate the v3.4.0 bug fix.

## One-Line Usage

```bash
python -m mousereach.eval.version_simulator Processing Processing
```

## Key Files

| File | Purpose |
|------|---------|
| `src/mousereach/eval/version_simulator.py` | Main module (18KB, 530 lines) |
| `src/mousereach/eval/test_version_simulator.py` | Test suite |
| `src/mousereach/eval/USAGE_VERSION_SIMULATOR.md` | Detailed usage guide |
| `VERSION_SIMULATOR_README.md` | Overview and examples |
| `DELIVERABLE_SUMMARY.md` | Complete technical summary |

## The Bug (v3.3.0)

```python
# BAD CODE:
reaches = [r for r in reaches if r.max_extent_pixels >= 0]
```

**Impact**: Dropped from 98% recall to 15% recall (lost 85% of valid reaches)

**Why**: Negative extent values are VALID (hand approaches but doesn't cross slit)

## The Fix (v3.4.0)

```python
# GOOD CODE:
# No extent filtering - preserve all reaches
```

**Result**: 98% recall

## Version Summary

| Version | Filter | Expected Recall |
|---------|--------|----------------|
| v1.0.0 | extent > 5 | ~45% |
| v2.0.0 | extent > 0 | ~63% |
| v3.0.0 | extent >= 5, duration >= 10 | ~48% |
| v3.3.0 | extent >= 0 ⚠️ | ~15% (BUG) |
| v3.4.0 | duration >= 2 ✓ | ~98% (CURRENT) |

## Quick Start

### CLI
```bash
# Run all simulations
python -m mousereach.eval.version_simulator Processing Processing

# Generate report
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

```
Version    Precision     Recall         F1   Detected         GT
v1.0.0        87.2%      45.2%       0.59        156        345
v2.0.0        82.4%      62.8%       0.71        263        345
v3.0.0        91.3%      48.1%       0.63        182        345
v3.3.0        93.1%      14.7% ⚠️ BUG!       0.25         55        345
v3.4.0        78.9%      98.3% ✓ Current    0.88        431        345
```

## Testing

```bash
cd /path/to/MouseReach\src\mousereach\eval
python test_version_simulator.py
```

## Custom Version

```python
from mousereach.eval import VersionFilter, apply_version_filter

custom = VersionFilter(
    version="custom",
    description="Custom threshold",
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

- **Validates fixes**: Proves v3.4.0 is better than v3.3.0
- **Documents evolution**: Shows how algorithm improved over time
- **Informs research**: Helps choose appropriate extent thresholds

## Recommendations

1. Use v3.4.0 for all detection
2. Filter by extent at analysis time (if needed)
3. Re-run v3.3.0 data with v3.4.0

## More Info

- Full usage guide: `src/mousereach/eval/USAGE_VERSION_SIMULATOR.md`
- Technical details: `DELIVERABLE_SUMMARY.md`
- Overview: `VERSION_SIMULATOR_README.md`
