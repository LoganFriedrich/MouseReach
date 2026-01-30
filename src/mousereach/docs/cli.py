"""
CLI for algorithm documentation extraction.

Command: mousereach-docs
"""

import argparse
from pathlib import Path

from .extractor import AlgorithmDocExtractor


def main():
    """Extract and display algorithm documentation."""
    parser = argparse.ArgumentParser(
        description="Extract algorithm documentation from source code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mousereach-docs                    Show all algorithms
  mousereach-docs --algo reach       Show reach detection only
  mousereach-docs --output docs.md   Save to file
  mousereach-docs --format json      Output as JSON
        """
    )
    parser.add_argument(
        "--algo", "-a",
        choices=["segmentation", "reach_detection", "outcome_classification",
                 "feature_extraction", "all"],
        default="all",
        help="Algorithm to show (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: print to stdout)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )

    args = parser.parse_args()

    extractor = AlgorithmDocExtractor()

    if args.algo == "all":
        output = extractor.generate_report(format=args.format)
    else:
        doc = extractor.extract(args.algo)
        if doc:
            if args.format == "json":
                import json
                output = json.dumps({
                    "name": doc.name,
                    "version": doc.version,
                    "summary": doc.summary,
                    "scientific_description": doc.scientific_description,
                    "detection_rules": doc.detection_rules,
                    "key_parameters": doc.key_parameters,
                }, indent=2)
            else:
                output = doc.to_markdown()
        else:
            print(f"No documentation found for {args.algo}")
            return

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Documentation written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
