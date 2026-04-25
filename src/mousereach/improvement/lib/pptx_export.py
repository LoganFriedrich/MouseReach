"""
PowerPoint export helpers for improvement process figures.

TODO: Implement PPTX generation for comparison slides showing
algorithm improvement across snapshots. Will use python-pptx
following the pattern established in generate_protocol_pptx.py
(see Databases/figures/Connectome_Grant/diagrams/).

Planned helpers:
  - create_comparison_slide(before_snapshot, after_snapshot) -> pptx slide
  - export_snapshot_summary(snapshot_dir) -> pptx slide with diagram + metrics
  - render_to_png(pptx_path) -> PNG via COM automation (Windows only)
"""
