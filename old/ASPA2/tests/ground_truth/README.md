# ground_truth

Human-validated boundary annotations for testing algorithm accuracy.

## File Format
```json
{
  "video_name": "20250806_CNT0311_P2",
  "video_file": "20250806_CNT0311_P2.mp4",
  "total_frames": 48689,
  "fps": 60.0,
  "boundaries": [3278, 5117, ...],  // 21 boundary frames
  "n_boundaries": 21
}
```

## Current Ground Truth (as of 2025-12-19)
- 20250820_CNT0104_P2 (normal)
- 20251029_CNT0408_P1 (stuck tray early)
- 20251031_CNT0413_P2 (normal)
- 20251031_CNT0415_P1 (normal)
- 20250806_CNT0311_P2 (late start + stuck)
- 20250806_CNT0312_P2 (late start + stuck)
