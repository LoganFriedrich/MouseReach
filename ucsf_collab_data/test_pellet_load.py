"""Quick test to see if we can load DLC files"""
import pandas as pd
from pathlib import Path

DLC_BASE = Path(r"X:\! DLC Output\Analyzed")

# Try Group L first
group_l = DLC_BASE / "L" / "Post-Processing"
print(f"Checking: {group_l}")
print(f"Exists: {group_l.exists()}")

if group_l.exists():
    csv_files = list(group_l.glob("*DLC*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) > 0:
        test_file = csv_files[0]
        print(f"\nTrying to load: {test_file.name}")

        try:
            df = pd.read_csv(test_file, header=[0, 1, 2], index_col=0)
            print(f"Success! Shape: {df.shape}")
            print(f"Columns: {df.columns[:5]}")

            # Check for Pellet and Pillar
            for col in df.columns:
                if 'pellet' in str(col).lower():
                    print(f"Found pellet column: {col}")
                    break

            for col in df.columns:
                if 'pillar' in str(col).lower():
                    print(f"Found pillar column: {col}")
                    break

        except Exception as e:
            print(f"Error: {e}")
