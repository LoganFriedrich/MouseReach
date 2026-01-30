"""
Common utility functions shared across MouseReach modules.
"""
import os
from pathlib import Path
from typing import Optional


def get_username() -> str:
    """Get current username for logging and validation tracking.

    Returns:
        Username string, or 'unknown' if not determinable.
    """
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get('USERNAME', os.environ.get('USER', 'unknown'))
