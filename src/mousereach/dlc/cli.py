#!/usr/bin/env python3
"""CLI entry points for MouseReach DLC"""

import argparse
from pathlib import Path


def main_batch():
    """Run DLC batch processing"""
    from mousereach.dlc.core import run_dlc_batch, find_videos_for_dlc
    
    parser = argparse.ArgumentParser(description="Run DLC on videos")
    parser.add_argument('-i', '--input', type=Path, required=True, help="Input directory")
    parser.add_argument('-c', '--config', type=Path, required=True, help="DLC config.yaml or project folder")
    parser.add_argument('-o', '--output', type=Path, help="Output directory")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device (default: 0)")
    parser.add_argument('--cpu', action='store_true', help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    # Auto-append config.yaml if directory passed
    config_path = args.config
    if config_path.is_dir():
        config_path = config_path / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return
    
    videos = find_videos_for_dlc(args.input)
    print(f"Found {len(videos)} videos to process")
    
    if not videos:
        print("No videos to process.")
        return
    
    # Use None for CPU mode
    gpu = None if args.cpu else args.gpu
    
    results = run_dlc_batch(videos, config_path, args.output, gpu)
    
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\nComplete: {success}/{len(results)} succeeded")


def main_quality():
    """Check DLC quality"""
    from mousereach.dlc.core import check_dlc_quality
    
    parser = argparse.ArgumentParser(description="Check DLC output quality")
    parser.add_argument('h5_files', type=Path, nargs='+', help="DLC .h5 files")
    parser.add_argument('-o', '--output', type=Path, help="Output directory for reports")
    
    args = parser.parse_args()
    
    for h5_path in args.h5_files:
        report = check_dlc_quality(h5_path)
        print(f"\n{report.video_name}: {report.overall_quality.upper()}")
        print(f"  Frames: {report.total_frames}, Mean likelihood: {report.mean_likelihood:.2f}")
        if report.issues:
            for issue in report.issues:
                print(f"  ⚠ {issue}")
        
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            report.save(args.output / f"{report.video_name}_dlc_quality.json")


if __name__ == "__main__":
    main_quality()
