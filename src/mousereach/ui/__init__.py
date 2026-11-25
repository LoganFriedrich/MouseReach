"""
MouseReach UI Utilities
==================

Common UI components and styling utilities for napari widgets.
"""

# Only names that ui/utils.py actually defines. This package once also
# imported ``apply_widget_style``, which was never defined anywhere, so
# ``import mousereach.ui`` raised ImportError and took down every widget that
# imports the package (found by importing all the modules in the
# tree).
from mousereach.ui.utils import (
    HelpDialog,
    create_header_with_help,
    create_help_button,
    style_dev_button,
)

__all__ = [
    "HelpDialog",
    "create_header_with_help",
    "create_help_button",
    "style_dev_button",
]
