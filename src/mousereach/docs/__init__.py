"""
MouseReach Algorithm Documentation Extraction Module

Extracts structured documentation from algorithm source files.

Usage:
    from mousereach.docs import AlgorithmDocExtractor
    extractor = AlgorithmDocExtractor()
    docs = extractor.extract_all()

CLI:
    mousereach-docs                  # Print all algorithm docs
    mousereach-docs --algo reach     # Print reach detection docs
    mousereach-docs --output docs.md # Save to file
"""

from .extractor import AlgorithmDocExtractor

__all__ = ["AlgorithmDocExtractor"]
