#!/usr/bin/env python3
"""
review_tool.py - Human review and correction tool for reach detection

Allows viewing detected reaches and making corrections.
Corrections are saved back to the reaches JSON in a format usable by Step 4.

Usage:
    python revise_or_review/review_tool.py --video video.mp4 --reaches video_reaches.json
    python revise_or_review/review_tool.py --dir /path/to/Processing/

[GUI IMPLEMENTATION PLACEHOLDER]

For now, this provides a CLI interface for reviewing and editing.
A Napari-based GUI can be added later.
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys


from mousereach.reach.core import ReachDetector, get_username


def load_reaches(path: Path) -> dict:
    """Load reaches from JSON"""
    with open(path) as f:
        return json.load(f)


def save_reaches(data: dict, path: Path):
    """Save reaches to JSON"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def print_segment_summary(data: dict, segment_num: int):
    """Print summary of a segment's reaches"""
    for seg in data['segments']:
        if seg['segment_num'] == segment_num:
            print(f"\nSegment {segment_num}: {seg['n_reaches']} reaches")
            print(f"  Frames: {seg['start_frame']} - {seg['end_frame']}")
            print(f"  Reaches:")
            for r in seg['reaches']:
                print(f"    #{r['reach_num']}: frames {r['start_frame']}-{r['end_frame']} (apex: {r['apex_frame']}, extent: {r['max_extent_ruler']:.2f})")
            return
    print(f"Segment {segment_num} not found")


def add_reach(data: dict, segment_num: int, start: int, apex: int, end: int) -> bool:
    """Add a reach to a segment"""
    for seg in data['segments']:
        if seg['segment_num'] == segment_num:
            # Find next reach number
            existing_nums = [r['reach_num'] for r in seg['reaches']]
            next_num = max(existing_nums) + 1 if existing_nums else 1
            
            # Find max global reach_id
            max_id = max(r['reach_id'] for s in data['segments'] for r in s['reaches']) if any(s['reaches'] for s in data['segments']) else 0
            
            new_reach = {
                'reach_id': max_id + 1,
                'reach_num': next_num,
                'start_frame': start,
                'apex_frame': apex,
                'end_frame': end,
                'duration_frames': end - start,
                'max_extent_pixels': 0,  # Can't compute without DLC data
                'max_extent_ruler': 0,
                'manually_added': True
            }
            
            seg['reaches'].append(new_reach)
            seg['n_reaches'] = len(seg['reaches'])
            
            # Update corrections count
            data['corrections_made'] = data.get('corrections_made', 0) + 1
            
            return True
    return False


def remove_reach(data: dict, segment_num: int, reach_num: int) -> bool:
    """Remove a reach from a segment"""
    for seg in data['segments']:
        if seg['segment_num'] == segment_num:
            original_len = len(seg['reaches'])
            seg['reaches'] = [r for r in seg['reaches'] if r['reach_num'] != reach_num]
            
            if len(seg['reaches']) < original_len:
                seg['n_reaches'] = len(seg['reaches'])
                data['corrections_made'] = data.get('corrections_made', 0) + 1
                return True
    return False


def mark_validated(data: dict, notes: str = ""):
    """Mark the file as validated"""
    data['validated'] = True
    data['validated_by'] = get_username()
    data['validated_at'] = datetime.now().isoformat()
    if notes:
        data['validation_notes'] = notes


def save_validation_record(data: dict, reaches_path: Path, original_data: dict = None):
    """Save a separate validation record (audit trail)"""
    video_name = data.get('video_name', reaches_path.stem.replace('_reaches', ''))
    validation_path = reaches_path.parent / f"{video_name}_reaches_validation.json"
    
    # Count changes if we have original data
    changes_made = 0
    if original_data:
        changes_made = data.get('corrections_made', 0)
    
    validation_record = {
        'video_name': video_name,
        'validated_by': get_username(),
        'validated_at': datetime.now().isoformat(),
        'changes_made': changes_made,
        'total_reaches': data.get('summary', {}).get('total_reaches', 0),
        'n_segments': data.get('n_segments', 0),
        'notes': data.get('validation_notes', '')
    }
    
    with open(validation_path, 'w') as f:
        json.dump(validation_record, f, indent=2)
    
    return validation_path


def save_as_ground_truth(data: dict, reaches_path: Path):
    """Save as ground truth file (for dev/accuracy testing)"""
    video_name = data.get('video_name', reaches_path.stem.replace('_reaches', ''))
    gt_path = reaches_path.parent / f"{video_name}_reach_ground_truth.json"
    
    gt_data = {
        'video_name': video_name,
        'type': 'ground_truth',
        'created_by': get_username(),
        'created_at': datetime.now().isoformat(),
        'n_segments': data.get('n_segments', 0),
        'segments': data.get('segments', []),
        'summary': data.get('summary', {}),
        'annotation_notes': ''
    }
    
    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    return gt_path


def interactive_review(reaches_path: Path):
    """Interactive CLI review session"""
    data = load_reaches(reaches_path)
    original_data = json.loads(json.dumps(data))  # Deep copy for change tracking
    video_name = data.get('video_name', reaches_path.stem)
    
    print(f"\n{'='*60}")
    print(f"Reviewing: {video_name}")
    print(f"Segments: {data['n_segments']}, Total reaches: {data['summary']['total_reaches']}")
    print(f"Validated: {data.get('validated', False)}")
    print(f"{'='*60}")
    print("\nCommands:")
    print("  s <num>     - Show segment")
    print("  a <seg> <start> <apex> <end> - Add reach")
    print("  r <seg> <reach_num>   - Remove reach")
    print("  v [notes]   - Save as VALIDATED (for pipeline) + audit trail")
    print("  g           - Save as GROUND TRUTH (for dev/accuracy testing)")
    print("  w           - Save without validating")
    print("  q           - Quit without saving")
    print("")
    
    modified = False
    
    while True:
        try:
            cmd = input("> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == 'q':
                if modified:
                    confirm = input("Unsaved changes. Quit anyway? (y/n) ")
                    if confirm.lower() != 'y':
                        continue
                break
                
            elif cmd[0] == 's' and len(cmd) >= 2:
                print_segment_summary(data, int(cmd[1]))
                
            elif cmd[0] == 'a' and len(cmd) >= 5:
                seg, start, apex, end = int(cmd[1]), int(cmd[2]), int(cmd[3]), int(cmd[4])
                if add_reach(data, seg, start, apex, end):
                    print(f"Added reach to segment {seg}")
                    modified = True
                else:
                    print(f"Failed to add reach")
                    
            elif cmd[0] == 'r' and len(cmd) >= 3:
                seg, reach_num = int(cmd[1]), int(cmd[2])
                if remove_reach(data, seg, reach_num):
                    print(f"Removed reach {reach_num} from segment {seg}")
                    modified = True
                else:
                    print(f"Reach not found")
                    
            elif cmd[0] == 'v':
                notes = ' '.join(cmd[1:]) if len(cmd) > 1 else ""
                mark_validated(data, notes)
                save_reaches(data, reaches_path)
                val_path = save_validation_record(data, reaches_path, original_data)
                print(f"Validated and saved: {reaches_path}")
                print(f"Validation record: {val_path}")
                modified = False
                break
            
            elif cmd[0] == 'g':
                gt_path = save_as_ground_truth(data, reaches_path)
                print(f"Saved ground truth: {gt_path}")
                print("(Original file unchanged)")
                
            elif cmd[0] == 'w':
                save_reaches(data, reaches_path)
                print(f"Saved: {reaches_path}")
                modified = False
                
            else:
                print("Unknown command")
                
        except (ValueError, IndexError) as e:
            print(f"Invalid command: {e}")
        except KeyboardInterrupt:
            print("\nUse 'q' to quit")


def main():
    parser = argparse.ArgumentParser(description="Review and correct reach detection")
    parser.add_argument('--reaches', type=Path, help="Reaches JSON file to review")
    parser.add_argument('--dir', type=Path, help="Directory with files to review")
    parser.add_argument('--video', type=Path, help="Video file (for reference)")
    
    args = parser.parse_args()
    
    if args.reaches:
        interactive_review(args.reaches)
        
    elif args.dir:
        reach_files = list(args.dir.glob("*_reaches.json"))
        print(f"Found {len(reach_files)} files to review")
        
        for i, rf in enumerate(reach_files, 1):
            print(f"\n[{i}/{len(reach_files)}]")
            interactive_review(rf)
            
            if i < len(reach_files):
                cont = input("\nContinue to next? (y/n) ")
                if cont.lower() != 'y':
                    break
    else:
        parser.print_help()
        print("\nNote: For GUI-based review with video overlay, a Napari-based tool")
        print("can be implemented. This CLI tool provides basic correction capability.")


if __name__ == "__main__":
    main()
