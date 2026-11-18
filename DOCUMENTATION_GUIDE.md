# MouseReach Documentation Guide

**For Developers: How to maintain documentation for MouseReach**

---

## Overview

Every MouseReach tool should have **three levels of documentation**:

1. **Quick Reference** - In-code docstrings and `--help` output
2. **Tool docs** - the stage and tool documents in docs/ (listed in docs/DOC_COVERAGE.json)
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
├── docs/                    # How each stage and tool behaves (see docs/DOC_COVERAGE.json)
│
└── src/mousereach/          # Module-level documentation (docstrings)
    ├── video_prep/          # Step 0 - Video cropping
    ├── dlc/                 # Step 1 - DeepLabCut integration
    ├── segmentation/        # Step 2 - Trial boundaries
    ├── reach/               # Step 3 - Reach detection
    ├── outcomes/            # Step 4 - Pellet outcomes
    ├── kinematics/          # Step 5 - Grasp features
    └── export/              # Step 6 - Export to Excel
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

docs/CLI_REFERENCE.md is generated from every command's `--help` by
`python -m mousereach.docs.generate_cli_reference` (rerun it after changing any
help text; it overwrites the file). `mousereach-doc-check --install-hook` installs
the commit hook that flags code changes whose documents were not touched.

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

Each stage document in docs/ should contain:

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
- [ ] Update the stage document in docs/
- [ ] Update USER_GUIDE.md
- [ ] Update README.md (if CLI changed)
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
- [ ] Create the stage document in docs/
- [ ] Add to USER_GUIDE.md
- [ ] Add to README.md
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
| `README.md` | Overview, CLI reference | Any CLI change |
| `USER_GUIDE.md` | Full workflow guide | Any workflow change |
| `INSTALL.md` | Installation | New dependencies |
| `src/mousereach/launcher.py` | Launcher header | Launcher changes |
| `src/mousereach/state.py` | State manager header | State changes |

---

*MouseReach Documentation Guide | Updated 2026-11-18*
