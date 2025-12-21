# aspa2_core

Core algorithms for ASPA2.

## Files
- `segmenter_robust.py` - Boundary detection algorithm (v2.1.0)
- `__init__.py` - Package init

## Usage
```python
from aspa2_core.segmenter_robust import segment_video_robust
boundaries, diagnostics = segment_video_robust("path/to/dlc_file.h5")
```
