"""
Algorithm documentation extractor.

Parses structured docstrings from algorithm source files and
generates readable documentation.

Expected docstring format:
    ALGORITHM SUMMARY
    =================
    Brief description.

    SCIENTIFIC DESCRIPTION
    ======================
    Detailed explanation.

    DETECTION RULES
    ===============
    1. Rule one
    2. Rule two

    KEY PARAMETERS
    ==============
    | Parameter | Value | Rationale |
    |-----------|-------|-----------|

    VALIDATION HISTORY
    ==================
    - v1.0: Initial release
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AlgorithmDoc:
    """Extracted documentation for one algorithm."""
    name: str
    version: str = ""
    source_file: str = ""
    summary: str = ""
    scientific_description: str = ""
    input_requirements: str = ""
    detection_rules: str = ""
    key_parameters: str = ""
    output_format: str = ""
    validation_history: str = ""
    known_limitations: str = ""
    references: str = ""
    raw_docstring: str = ""

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = []
        lines.append(f"## {self.name}" + (f" (v{self.version})" if self.version else ""))
        lines.append("")

        if self.source_file:
            lines.append(f"*Source: `{self.source_file}`*")
            lines.append("")

        if self.summary:
            lines.append("### Summary")
            lines.append(self.summary)
            lines.append("")

        if self.scientific_description:
            lines.append("### How It Works")
            lines.append(self.scientific_description)
            lines.append("")

        if self.input_requirements:
            lines.append("### Input Requirements")
            lines.append(self.input_requirements)
            lines.append("")

        if self.detection_rules:
            lines.append("### Detection Rules")
            lines.append(self.detection_rules)
            lines.append("")

        if self.key_parameters:
            lines.append("### Key Parameters")
            lines.append(self.key_parameters)
            lines.append("")

        if self.output_format:
            lines.append("### Output Format")
            lines.append(self.output_format)
            lines.append("")

        if self.validation_history:
            lines.append("### Validation History")
            lines.append(self.validation_history)
            lines.append("")

        if self.known_limitations:
            lines.append("### Known Limitations")
            lines.append(self.known_limitations)
            lines.append("")

        if self.references:
            lines.append("### References")
            lines.append(self.references)
            lines.append("")

        return "\n".join(lines)


class AlgorithmDocExtractor:
    """Extracts structured documentation from algorithm source files."""

    # Map of algorithm names to source files (relative to mousereach package)
    ALGORITHM_FILES = {
        "segmentation": "segmentation/core/segmenter_robust.py",
        "reach_detection": "reach/core/reach_detector.py",
        "outcome_classification": "outcomes/core/pellet_outcome.py",
        "feature_extraction": "kinematics/core/feature_extractor.py",
    }

    # Sections to extract from docstrings
    SECTIONS = [
        "ALGORITHM SUMMARY",
        "SCIENTIFIC DESCRIPTION",
        "INPUT REQUIREMENTS",
        "DETECTION RULES",
        "KEY PARAMETERS",
        "OUTPUT FORMAT",
        "VALIDATION HISTORY",
        "KNOWN LIMITATIONS",
        "REFERENCES",
    ]

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize extractor.

        Args:
            base_path: Path to mousereach package. Auto-detected if not provided.
        """
        if base_path is None:
            # Auto-detect from this file's location
            base_path = Path(__file__).parent.parent
        self.base_path = Path(base_path)

    def extract_all(self) -> Dict[str, AlgorithmDoc]:
        """Extract documentation from all algorithm files."""
        docs = {}
        for name, rel_path in self.ALGORITHM_FILES.items():
            doc = self.extract(name)
            if doc:
                docs[name] = doc
        return docs

    def extract(self, algorithm: str) -> Optional[AlgorithmDoc]:
        """
        Extract documentation for one algorithm.

        Args:
            algorithm: Algorithm name (e.g., "segmentation", "reach_detection")

        Returns:
            AlgorithmDoc or None if not found
        """
        rel_path = self.ALGORITHM_FILES.get(algorithm)
        if not rel_path:
            return None

        file_path = self.base_path / rel_path
        if not file_path.exists():
            return None

        # Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get module docstring
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree) or ""
        except SyntaxError:
            docstring = ""

        # Also try to get VERSION constant
        version = ""
        version_match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            version = version_match.group(1)

        # Parse sections
        doc = AlgorithmDoc(
            name=algorithm.replace("_", " ").title(),
            version=version,
            source_file=rel_path,
            raw_docstring=docstring
        )

        # Extract each section
        sections = self._parse_sections(docstring)

        doc.summary = sections.get("ALGORITHM SUMMARY", "")
        doc.scientific_description = sections.get("SCIENTIFIC DESCRIPTION", "")
        doc.input_requirements = sections.get("INPUT REQUIREMENTS", "")
        doc.detection_rules = sections.get("DETECTION RULES", "")
        doc.key_parameters = sections.get("KEY PARAMETERS", "")
        doc.output_format = sections.get("OUTPUT FORMAT", "")
        doc.validation_history = sections.get("VALIDATION HISTORY", "")
        doc.known_limitations = sections.get("KNOWN LIMITATIONS", "")
        doc.references = sections.get("REFERENCES", "")

        # If no structured sections, use the whole docstring as summary
        if not any([doc.summary, doc.scientific_description, doc.detection_rules]):
            doc.summary = docstring.strip()

        return doc

    def _parse_sections(self, docstring: str) -> Dict[str, str]:
        """Parse docstring into sections."""
        sections = {}

        if not docstring:
            return sections

        # Split by section headers (ALL CAPS followed by ===)
        pattern = r'([A-Z][A-Z\s]+)\n[=]+\n'
        parts = re.split(pattern, docstring)

        # First part is preamble (before any sections)
        i = 1
        while i < len(parts) - 1:
            header = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            # Find where content ends (at next section or end)
            sections[header] = content
            i += 2

        return sections

    def generate_report(self, format: str = "markdown") -> str:
        """
        Generate formatted documentation for all algorithms.

        Args:
            format: "markdown" or "json"

        Returns:
            Formatted string
        """
        docs = self.extract_all()

        if format == "json":
            import json
            return json.dumps(
                {name: {
                    "name": doc.name,
                    "version": doc.version,
                    "summary": doc.summary,
                    "scientific_description": doc.scientific_description,
                    "detection_rules": doc.detection_rules,
                    "key_parameters": doc.key_parameters,
                } for name, doc in docs.items()},
                indent=2
            )

        # Markdown format
        lines = []
        lines.append("# MouseReach Algorithm Documentation")
        lines.append("")
        lines.append("*Auto-generated from source code docstrings.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        for name, doc in docs.items():
            lines.append(doc.to_markdown())
            lines.append("---")
            lines.append("")

        return "\n".join(lines)
