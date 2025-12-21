# MouseReach

Behavioral analysis pipeline for mouse skilled reaching tasks.

## ASPA2 - Automated Skilled Pellet Assessment v2

A complete pipeline for analyzing mouse reaching behavior from video.

### Quick Start

```bash
# Activate your DLC environment
conda activate DLC-env

# Run batch segmentation
cd ASPA2
python scripts/util_batch_segment.py

# Validate segments
python tools/boundary_annotator_v2.py
```

### Documentation

- [Pipeline Specification](ASPA2/ASPA2_PIPELINE_SPEC.md) - Full pipeline design
- [Development Summary](ASPA2/2025-12-19_ASPA2_Summary.md) - Current progress

### Structure

```
ASPA2/
├── aspa2_core/          # Core algorithms
├── tools/               # Interactive validation tools
├── scripts/             # Batch processing utilities
├── tests/ground_truth/  # Validated test data
└── docs/                # Documentation
```

### Requirements

- Python 3.8+
- DeepLabCut
- PyQt5
- pandas, numpy, scipy
- opencv-python

### Current Version

- Segmenter: v2.1.0
- Algorithm: sabl_centered_crossing_v2
- Accuracy: 99.2% (125/126 boundaries on ground truth)

---

*Blackmore Lab - Marquette University*
