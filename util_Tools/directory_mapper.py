"""
Directory Tree Mapper (with progress)
Generates a text map of folder structure (no file contents).
"""

import os
from pathlib import Path
from datetime import datetime

def map_directory(root_path, max_depth=4, output_file=None):
    """
    Generate a directory tree map.
    """
    root = Path(root_path)
    lines = []
    folder_count = 0
    
    print(f"Starting scan of: {root}")
    print(f"Max depth: {max_depth}")
    print("-" * 40)
    
    lines.append(f"Directory Map: {root}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Max depth: {max_depth}")
    lines.append("=" * 60)
    lines.append("")
    
    def walk(path, prefix="", depth=0):
        nonlocal folder_count
        
        if depth >= max_depth:
            return
        
        try:
            dirs = sorted([d for d in path.iterdir() if d.is_dir()])
        except PermissionError:
            lines.append(f"{prefix}[ACCESS DENIED]")
            return
        
        for i, d in enumerate(dirs):
            folder_count += 1
            
            # Progress every 50 folders
            if True:
                print(f"  Scanned {folder_count} folders... (currently: {d.name})")
            
            is_last = (i == len(dirs) - 1)
            connector = "└── " if is_last else "├── "
            
            try:
                n_items = len(list(d.iterdir()))
                item_count = f" ({n_items} items)"
            except PermissionError:
                item_count = " [access denied]"
            
            lines.append(f"{prefix}{connector}{d.name}{item_count}")
            
            extension = "    " if is_last else "│   "
            walk(d, prefix + extension, depth + 1)
    
    walk(root)
    
    print("-" * 40)
    print(f"Done! Scanned {folder_count} folders.")
    
    result = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Saved to: {output_file}")
    else:
        print(result)
    
    return result


if __name__ == "__main__":
    # === EDIT THIS PATH ===
    target = r"X:\! DLC Output"
    
    output = f"directory_map_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    
    map_directory(target, max_depth=4, output_file=output)
    
    print("\nPress Enter to close...")
    input()