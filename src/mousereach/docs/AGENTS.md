<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-16 | Updated: 2026-01-16 -->

# docs

## Purpose
Algorithm documentation extraction module that parses structured docstrings from MouseReach algorithm source files (segmentation, reach detection, outcome classification, feature extraction) and generates human-readable documentation in markdown or JSON format. Enables automated generation of algorithm documentation from code comments.

## Key Files
| File | Description |
|------|-------------|
| `extractor.py` | Core extraction logic - parses structured docstrings with sections (ALGORITHM SUMMARY, SCIENTIFIC DESCRIPTION, DETECTION RULES, KEY PARAMETERS, etc.) from algorithm source files |
| `cli.py` | Command-line interface for mousereach-docs - supports filtering by algorithm, markdown/JSON output, and file export |
| `__init__.py` | Module exports AlgorithmDocExtractor class |

## For AI Agents

### Working In This Directory
- The extractor expects algorithm files to contain structured docstrings with ALL CAPS section headers followed by `===` underlines
- Recognized sections: ALGORITHM SUMMARY, SCIENTIFIC DESCRIPTION, INPUT REQUIREMENTS, DETECTION RULES, KEY PARAMETERS, OUTPUT FORMAT, VALIDATION HISTORY, KNOWN LIMITATIONS, REFERENCES
- Algorithm file locations are hardcoded in `ALGORITHM_FILES` dict: segmentation, reach_detection, outcome_classification, feature_extraction
- Uses AST parsing to extract module-level docstrings and regex to find VERSION constants
- If no structured sections found, falls back to using entire docstring as summary

### CLI Commands
```bash
mousereach-docs                    # Show all algorithms (markdown)
mousereach-docs --algo reach_detection   # Show specific algorithm
mousereach-docs --output docs.md   # Save to file
mousereach-docs --format json      # Output as JSON
```

## Dependencies

### Internal
- `mousereach.segmentation.core.segmenter_robust` - Segmentation algorithm source
- `mousereach.reach.core.reach_detector` - Reach detection algorithm source
- `mousereach.outcomes.core.pellet_outcome` - Outcome classification algorithm source
- `mousereach.kinematics.core.feature_extractor` - Feature extraction algorithm source

### External
- `ast` - Python AST parsing for docstring extraction
- `re` - Regex for section parsing and VERSION extraction
- `pathlib` - Path handling
- `dataclasses` - AlgorithmDoc data structure
- `json` - JSON output format

<!-- MANUAL: -->
