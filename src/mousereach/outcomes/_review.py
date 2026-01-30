#!/usr/bin/env python3
"""
review_tool.py - Human review and correction for pellet outcomes

Usage:
    python revise_or_review/review_tool.py --outcomes video_pellet_outcomes.json
    python revise_or_review/review_tool.py --dir A:\\Processing\\

Commands:
    s <num>              - Show segment details
    c <num> <outcome>    - Change outcome (retrieved, displaced_sa, displaced_outside, untouched, no_pellet, uncertain)
    v [notes]            - Save as VALIDATED (for pipeline) + audit trail
    g                    - Save as GROUND TRUTH (for dev/accuracy testing)
    w                    - Save without validating
    q                    - Quit without saving
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

from mousereach.outcomes.core import get_username

OUTCOMES = ['retrieved', 'displaced_sa', 'displaced_outside', 'untouched', 'no_pellet', 'uncertain']


def save_validation_record(data: dict, outcomes_path: Path):
    """Save a separate validation record (audit trail)"""
    video_name = data.get('video_name', outcomes_path.stem.replace('_pellet_outcomes', ''))
    validation_path = outcomes_path.parent / f"{video_name}_outcomes_validation.json"
    
    validation_record = {
        'video_name': video_name,
        'validated_by': get_username(),
        'validated_at': datetime.now().isoformat(),
        'changes_made': data.get('corrections_made', 0),
        'summary': data.get('summary', {}),
        'n_segments': data.get('n_segments', 0),
        'notes': data.get('validation_notes', '')
    }
    
    with open(validation_path, 'w') as f:
        json.dump(validation_record, f, indent=2)
    
    return validation_path


def save_as_ground_truth(data: dict, outcomes_path: Path):
    """Save as ground truth file (for dev/accuracy testing)"""
    video_name = data.get('video_name', outcomes_path.stem.replace('_pellet_outcomes', ''))
    gt_path = outcomes_path.parent / f"{video_name}_outcome_ground_truth.json"
    
    gt_data = {
        'video_name': video_name,
        'type': 'ground_truth',
        'created_by': get_username(),
        'created_at': datetime.now().isoformat(),
        'n_segments': data.get('n_segments', 0),
        'segments': [
            {
                'segment_num': s['segment_num'],
                'outcome': s['outcome'],
                'notes': ''
            }
            for s in data.get('segments', [])
        ],
        'summary': data.get('summary', {}),
        'annotation_notes': ''
    }
    
    with open(gt_path, 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    return gt_path


def interactive_review(path: Path):
    with open(path) as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Reviewing: {data.get('video_name')}")
    disp = data['summary'].get('displaced_sa', 0) + data['summary'].get('displaced_outside', 0)
    print(f"Summary: R={data['summary'].get('retrieved', 0)}/D={disp}/U={data['summary'].get('untouched', 0)}")
    print(f"Validated: {data.get('validated', False)}")
    print(f"{'='*60}")
    print("\nCommands:")
    print("  s <num>              - Show segment details")
    print("  c <num> <outcome>    - Change outcome")
    print("  v [notes]            - Save as VALIDATED (for pipeline)")
    print("  g                    - Save as GROUND TRUTH (for dev)")
    print("  w                    - Save without validating")
    print("  q                    - Quit without saving")
    print(f"\nValid outcomes: {', '.join(OUTCOMES)}")
    
    modified = False
    while True:
        try:
            cmd = input("> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == 'q':
                if modified and input("Unsaved changes. Quit? (y/n) ").lower() != 'y':
                    continue
                break
                
            elif cmd[0] == 's' and len(cmd) >= 2:
                seg_num = int(cmd[1])
                for s in data['segments']:
                    if s['segment_num'] == seg_num:
                        print(f"  Segment {seg_num}: {s['outcome']} (conf={s.get('confidence', 'N/A')})")
                        if 'distance_from_pillar_start' in s:
                            print(f"  Start dist: {s['distance_from_pillar_start']:.2f}, End dist: {s['distance_from_pillar_end']:.2f}")
                        if 'causal_reach_id' in s:
                            print(f"  Causal reach: {s['causal_reach_id']}")
                        break
                else:
                    print(f"Segment {seg_num} not found")
                    
            elif cmd[0] == 'c' and len(cmd) >= 3:
                seg_num, new_outcome = int(cmd[1]), cmd[2].lower()
                if new_outcome not in OUTCOMES:
                    print(f"Invalid outcome. Use: {OUTCOMES}")
                    continue
                for s in data['segments']:
                    if s['segment_num'] == seg_num:
                        old = s['outcome']
                        s['outcome'] = new_outcome
                        s['manually_corrected'] = True
                        data['corrections_made'] = data.get('corrections_made', 0) + 1
                        modified = True
                        print(f"  Changed segment {seg_num}: {old} → {new_outcome}")
                        break
                else:
                    print(f"Segment {seg_num} not found")
                    
            elif cmd[0] == 'v':
                notes = ' '.join(cmd[1:]) if len(cmd) > 1 else ""
                data['validated'] = True
                data['validated_by'] = get_username()
                data['validated_at'] = datetime.now().isoformat()
                if notes:
                    data['validation_notes'] = notes
                # Recalculate summary
                data['summary']['retrieved'] = sum(1 for s in data['segments'] if s['outcome'] == 'retrieved')
                data['summary']['displaced_sa'] = sum(1 for s in data['segments'] if s['outcome'] == 'displaced_sa')
                data['summary']['displaced_outside'] = sum(1 for s in data['segments'] if s['outcome'] == 'displaced_outside')
                data['summary']['untouched'] = sum(1 for s in data['segments'] if s['outcome'] == 'untouched')
                data['summary']['no_pellet'] = sum(1 for s in data['segments'] if s['outcome'] == 'no_pellet')
                data['summary']['uncertain'] = sum(1 for s in data['segments'] if s['outcome'] == 'uncertain')
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                val_path = save_validation_record(data, path)
                print(f"Validated and saved: {path}")
                print(f"Validation record: {val_path}")
                modified = False
                break
            
            elif cmd[0] == 'g':
                gt_path = save_as_ground_truth(data, path)
                print(f"Saved ground truth: {gt_path}")
                print("(Original file unchanged)")
                
            elif cmd[0] == 'w':
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved: {path}")
                modified = False
                
            else:
                print("Unknown command")
                
        except (ValueError, IndexError) as e:
            print(f"Invalid command: {e}")
        except KeyboardInterrupt:
            print("\nUse 'q' to quit")


def main():
    parser = argparse.ArgumentParser(description="Review pellet outcomes")
    parser.add_argument('--outcomes', type=Path, help="Outcomes JSON file")
    parser.add_argument('--dir', type=Path, help="Directory with files")
    args = parser.parse_args()
    
    if args.outcomes:
        interactive_review(args.outcomes)
    elif args.dir:
        for f in args.dir.glob("*_pellet_outcomes.json"):
            interactive_review(f)
            if input("\nContinue to next? (y/n) ").lower() != 'y':
                break
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
