# MouseReach Documentation Guide

**For Developers: How to maintain documentation for MouseReach**

---

## Overview

Every MouseReach tool should have **three levels of documentation**:

1. **Quick Reference** - In-code docstrings and `--help` output
2. **Tool README** - Step-specific README.md in each Step folder
3. **User Guide** - Comprehensive USER_GUIDE.md at project root

When you modify any tool, **update all three levels**.

---

## Documentation Hierarchy

```
MouseReach/
├── README.md                # Project overview, quick start, CLI reference
├── USER_GUIDE.md            # Full user workflow guide
├── INSTALL.md               # Installation instructions
├── DOCUMENTATION_GUIDE.md   # This file (developer reference)
├── AGENTS.md                # AI-navigable codebase index
├── CLAUDE.md                # AI development context
│
└── src/mousereach/          # Module-level documentation (AGENTS.md files)
    ├── AGENTS.md            # Main package index
    ├── video_prep/AGENTS.md     # Step 0 - Video cropping
    ├── dlc/AGENTS.md            # Step 1 - DeepLabCut integration
    ├── segmentation/AGENTS.md   # Step 2 - Trial boundaries
    ├── reach/AGENTS.md          # Step 3 - Reach detection
    ├── outcomes/AGENTS.md       # Step 4 - Pellet outcomes
    ├── kinematics/AGENTS.md     # Step 5 - Grasp features
    └── export/AGENTS.md         # Step 6 - Export to Excel
```

---

## Level 1: Quick Reference (In-Code)

### Script Headers

Every Python script should have a docstring header:

```python
#!/usr/bin/env python3
"""
Script Name
===========

One-line description of what this does.

Features:
- Feature 1
- Feature 2

Usage:
    python script.py arg1 arg2
    python script.py --help

Examples:
    python script.py video.mp4
    python script.py --batch /path/to/folder/
"""
```

### CLI Help

All CLI commands must support `--help`:

```python
parser = argparse.ArgumentParser(
    description="What this command does",
    epilog="Example: mousereach-command input.mp4 -o output/"
)
```

### Function Docstrings

Public functions need docstrings:

```python
def process_video(path: Path, threshold: float = 0.5) -> dict:
    """
    Process a video and return results.

    Args:
        path: Path to video file
        threshold: Detection threshold (0-1)

    Returns:
        Dictionary with keys: 'frames', 'detections', 'confidence'

    Raises:
        FileNotFoundError: If video doesn't exist
        ValueError: If threshold out of range
    """
```

---

## Level 2: Tool README (Step-Specific)

Each `Step*/README.md` should contain:

### Required Sections

```markdown
# Step N: Tool Name

Brief description of what this step does in the pipeline.

## Quick Start

\`\`\`bash
# Most common usage
mousereach-command input.mp4
\`\`\`

## Input Requirements

- What files are needed
- Expected format/naming conventions
- Dependencies on previous steps

## Output

- What files are generated
- File format description
- Where outputs are saved

## CLI Commands

| Command | Description |
|---------|-------------|
| `mousereach-xxx` | Main processing |
| `mousereach-xxx-review` | Manual review |

## Keyboard Shortcuts (for GUI tools)

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| N/P | Next/Previous |

## Algorithm Overview

Brief technical description of how it works.

## Troubleshooting

Common issues and solutions.
```

---

## Level 3: User Guide (Comprehensive)

The root `USER_GUIDE.md` should:

1. **Start with Quick Start** - Get users running in 30 seconds
2. **Explain the full pipeline** - Overview diagram
3. **Detail each step** - With screenshots if possible
4. **Show real workflows** - "Option A: GUI", "Option B: CLI"
5. **Troubleshooting section** - Common issues

### User Guide Sections

```markdown
# MouseReach User Guide

## Quick Start (v2.x)
## Overview
## Prerequisites
## Step 0: Video Preparation
## Step 1: DLC Processing
## Step 2: Segmentation
## Step 3: Reach Detection
## Step 4: Pellet Outcomes
## Step 5: Grasp Kinematics
## Step 6: Export
## Typical Workflow Summary
## Troubleshooting
## Getting Help
```

---

## When to Update Documentation

### New Feature Added
- [ ] Update relevant script docstring
- [ ] Update Step README.md
- [ ] Update USER_GUIDE.md
- [ ] Update MouseReach_README.md (if CLI changed)
- [ ] Update INSTALL.md (if new dependencies)

### Bug Fixed
- [ ] Add to Troubleshooting if common issue
- [ ] Update any incorrect documentation

### API Changed
- [ ] Update all affected docstrings
- [ ] Update CLI help text
- [ ] Update README examples
- [ ] Update USER_GUIDE examples

### New Step Added
- [ ] Create Step*/README.md
- [ ] Add to USER_GUIDE.md
- [ ] Add to MouseReach_README.md
- [ ] Update pipeline diagrams

---

## Documentation Standards

### Formatting

- Use **bold** for emphasis
- Use `code` for commands, filenames, parameters
- Use tables for reference info
- Use code blocks for examples

### Version Info

Every major doc file should end with:

```markdown
---

*MouseReach vX.Y.Z | Updated YYYY-MM-DD*
```

### Keep In Sync

These values must match across all docs:
- Version numbers
- CLI command names
- Keyboard shortcuts
- File naming conventions

---

## Checklist: Documentation Review

Before any release:

- [ ] All `--help` outputs are accurate
- [ ] All README.md files reflect current functionality
- [ ] USER_GUIDE.md Quick Start works
- [ ] INSTALL.md instructions work on fresh system
- [ ] All keyboard shortcuts documented
- [ ] Pipeline diagram matches actual steps
- [ ] No references to removed features
- [ ] Version numbers updated

---

## Key Files to Keep Updated

| File | Purpose | Update When |
|------|---------|-------------|
| `MouseReach_README.md` | Overview, CLI reference | Any CLI change |
| `USER_GUIDE.md` | Full workflow guide | Any workflow change |
| `INSTALL.md` | Installation | New dependencies |
| `launch_all.py` | Launcher header | Launcher changes |
| `mousereach_state.py` | State manager header | State changes |
| `Step*/README.md` | Step-specific docs | That step changes |

---

*MouseReach Documentation Guide | Updated 2025-01-05*
