"""Shared utility functions for the compression pipeline."""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


def run_command(
    cmd: str,
    check: bool = True,
    capture_output: bool = True,
    shell: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.
    
    Args:
        cmd: Command to run
        check: Raise exception on non-zero exit
        capture_output: Capture stdout/stderr
        shell: Run in shell mode
        
    Returns:
        CompletedProcess instance
    """
    result = subprocess.run(
        cmd,
        shell=shell,
        capture_output=capture_output,
        text=True,
    )
    
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    
    return result


def get_file_size_gb(path: Path) -> float:
    """Get file size in GB."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 ** 3)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[Any, Any]:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_dirs(*paths: Path) -> None:
    """Ensure directories exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
