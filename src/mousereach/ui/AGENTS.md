<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# ui

## Purpose
Shared UI components for napari-based MouseReach widgets, providing help dialog popups with markdown-to-HTML rendering, consistent button styling, header widgets with integrated help buttons, and standardized help text templates for all pipeline steps (video preparation, DLC processing, segmentation, reach/outcome detection, feature extraction).

## Key Files
| File | Description |
|------|-------------|
| `utils.py` | UI component library - HelpDialog (markdown popup), help button creators, header widgets, dev button styling, and extensive STEP*_HELP templates for pipeline documentation |
| `__init__.py` | Exports HelpDialog and apply_widget_style (note: apply_widget_style referenced in __init__ but not defined in utils.py) |

## For AI Agents

### Working In This Directory
- All components use qtpy for Qt abstraction (supports PyQt5/PySide2/PySide6)
- HelpDialog converts simple markdown to HTML: headers (#, ##, ###), bold (**text**), lists (- item)
- Help text templates define user-facing documentation for each pipeline step (STEP0_HELP through STEP5_HELP, plus PIPELINE_HELP)
- `create_help_button()` returns 24x24 circular "?" button with hover effects
- `style_dev_button()` creates dashed-border gray buttons for development-only features
- Help templates explain: what the step does, normal workflow, keyboard shortcuts, when to use, ground truth vs validation

### CLI Commands
N/A - This is a library module imported by napari widgets, not a standalone CLI tool

## Dependencies

### Internal
None - this is a shared utility module used by other mousereach UI components

### External
- `qtpy.QtWidgets` - Qt widget classes (QPushButton, QDialog, QVBoxLayout, etc.)
- `qtpy.QtCore` - Qt core functionality (Qt alignment constants)
- `qtpy.QtGui` - Qt GUI classes (QFont)
- `pathlib` - Path handling (imported but not used in current code)

<!-- MANUAL: -->
