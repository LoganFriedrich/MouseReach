# Step 2: Reach Detection

## Status: NOT YET IMPLEMENTED

This step will detect individual reaches within each pellet segment.

## Planned Functionality

- Detect reach initiation (paw lifts off from resting position)
- Detect reach termination (paw returns or contacts pellet)
- Classify reach type (grasp attempt, touch, miss)
- Track reach trajectory

## Input

- Video file
- DLC tracking data
- Validated segment boundaries (from Step 1)

## Output

- `*_reaches.json` - List of detected reaches with timing and features

## Prerequisites

- Step 1 must be complete (files in Seg_Validated/)
