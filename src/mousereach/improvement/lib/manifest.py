"""
Manifest dataclass for improvement snapshots.

Each snapshot directory contains a manifest.json describing what version
of the algorithm was captured, when, by whom, and with what results.

Usage:
    from mousereach.improvement.lib.manifest import Manifest

    m = Manifest(
        version_id="v2.1.3",
        tag="phantom-first post-validation",
        timestamp="2026-04-23T20:00:56-05:00",
        code_hash="8e43976",
        description="Replaced narrow B1-miss correction with principled two-step post-validation.",
    )
    m.to_json("manifest.json")

    m2 = Manifest.from_json("manifest.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Manifest:
    """Metadata for a single improvement snapshot.

    Fields
    ------
    version_id : str
        Semantic version of the algorithm (e.g. "v2.1.3").
    tag : str
        Human-readable label for this snapshot (e.g. "phantom-first post-validation").
    timestamp : str
        ISO-8601 timestamp of the commit or snapshot creation.
    code_hash : str
        Short git commit hash that produced this version.
    pipeline_versions : dict
        Mapping of pipeline component names to their versions at snapshot time
        (e.g. {"segmenter": "2.1.3", "reach_detector": "7.0.0"}).
    inputs : list of str
        Descriptions of input data used for evaluation
        (e.g. ["47-GT-video corpus", "49 P-tray videos"]).
    metrics_summary : dict
        Key performance numbers
        (e.g. {"videos_21_21_at_5f": "33/47", "boundaries_at_5f": "959/987"}).
    artifacts : list of str
        Relative paths to key files in this snapshot directory
        (e.g. ["vault/logic_diagram.md", "figures/logic_diagram.png"]).
    description : str
        Free-text description of what this version does and why it matters.
    """
    version_id: str
    tag: str
    timestamp: str = ""
    code_hash: str = ""
    pipeline_versions: Dict[str, str] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        return asdict(self)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Write manifest to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Manifest:
        """Construct a Manifest from a plain dict."""
        return cls(
            version_id=d.get("version_id", ""),
            tag=d.get("tag", ""),
            timestamp=d.get("timestamp", ""),
            code_hash=d.get("code_hash", ""),
            pipeline_versions=d.get("pipeline_versions", {}),
            inputs=d.get("inputs", []),
            metrics_summary=d.get("metrics_summary", {}),
            artifacts=d.get("artifacts", []),
            description=d.get("description", ""),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Manifest:
        """Read a manifest from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"Manifest(version_id={self.version_id!r}, tag={self.tag!r}, "
            f"code_hash={self.code_hash!r})"
        )
