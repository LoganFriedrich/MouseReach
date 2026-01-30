#!/usr/bin/env python3
"""
Fix PowerShell execution policy for network drives.

This sets the execution policy to Bypass for the current user,
which allows conda activate scripts to run from network drives (Y:, etc.)
"""

import subprocess
import sys


def fix_powershell_policy():
    """Set PowerShell execution policy to Bypass for current user."""
    print("Fixing PowerShell execution policy for network drives...")
    print()

    try:
        # Set execution policy to Bypass for current user
        result = subprocess.run(
            ["powershell", "-Command",
             "Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser -Force"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("SUCCESS: PowerShell execution policy set to Bypass")
            print()
            print("You can now run conda activate without errors.")
            print("Restart your PowerShell terminal for changes to take effect.")
            return 0
        else:
            print(f"ERROR: {result.stderr}")
            return 1

    except FileNotFoundError:
        print("ERROR: PowerShell not found. This fix is only needed on Windows.")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def main():
    """Entry point."""
    sys.exit(fix_powershell_policy())


if __name__ == "__main__":
    main()
