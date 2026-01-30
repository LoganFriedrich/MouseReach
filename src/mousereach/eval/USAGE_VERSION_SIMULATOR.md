# Version Simulator Usage Guide

## Overview

The version simulator simulates how OLD algorithm versions would have performed on the current ground truth data. This is useful for:
- Understanding the impact of algorithm changes
- Validating that fixes actually improved performance
- Documenting algorithm evolution

## Quick Start

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

### v1.0.0 - Basic reach detection
- Only detected reaches with extent > 5 pixels
- No apex detection
- Simple threshold-based start/end
- **Issue**: High false negative rate (missed short reaches)

### v2.0.0 - Added extent filtering
- Required extent > 0 (positive only)
- Added apex detection
- Better start/end logic with nose engagement
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
- Human GT analysis (2026-01) revealed many valid reaches have -2 to -15px extent

### v3.4.0 - Current (fixed) ✓
- Removed the extent filter entirely
- Best recall (finds all reaches including short approaches)
- Extent value preserved in Reach object for downstream filtering if needed

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

```
================================================================================
REACH DETECTION VERSION COMPARISON
================================================================================

Version    Precision     Recall         F1   Detected         GT
--------------------------------------------------------------------------------
v1.0.0        87.2%      45.2%       0.59        156        345
v2.0.0        82.4%      62.8%       0.71        263        345
v3.0.0        91.3%      48.1%       0.63        182        345
v3.3.0        93.1%      14.7% ⚠️ BUG!       0.25         55        345
v3.4.0        78.9%      98.3% ✓ Current    0.88        431        345
--------------------------------------------------------------------------------

IMPACT OF v3.3.0 BUG:
  - Recall dropped from 98.3% to 14.7%
  - Lost 85% of valid reaches
  - v3.4.0 fix improved recall by 568%
```

## Custom Analysis

You can also apply custom filters to simulate hypothetical versions:

```python
from mousereach.eval.version_simulator import VersionFilter, apply_version_filter

# Define a custom version
custom_filter = VersionFilter(
    version="v2.5.0",
    description="Custom: extent >= 2, duration >= 5",
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

## Recommendations

1. **Use v3.4.0 (current)** for all detection
2. **Filter by extent at analysis time** if needed:
   ```python
   # Load reaches
   reaches = load_reaches("video_reaches.json")

   # Filter by extent for specific analysis
   extended_reaches = [r for r in reaches if r.max_extent_pixels >= 5]
   approach_reaches = [r for r in reaches if -5 <= r.max_extent_pixels < 5]
   ```
3. **Document your extent threshold** in your analysis scripts
4. **Re-run detection with v3.4.0** for any data processed with v3.3.0

## Troubleshooting

### "No ground truth files found"
- Check that your GT directory contains `*_reach_ground_truth.json` files
- Verify paths are correct

### "No algorithm output files found"
- Check that your reaches directory contains `*_reaches.json` files
- Verify video IDs match between GT and reaches files

### "Import error"
Make sure you're in the correct directory:
```bash
cd /path/to/MouseReach
python -m mousereach.eval.version_simulator ...
```
