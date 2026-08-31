"""Shared utility functions for the compression pipeline."""

import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable
from datetime import datetime
import json


def _fail_if_needed(result: subprocess.CompletedProcess, check: bool, cmd) -> subprocess.CompletedProcess:
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


def _set_pty_winsize(fd: int, rows: int = 24, cols: int = 120) -> None:
    """Give the PTY a real size so tqdm / rich draw a bar instead of staying silent."""
    try:
        packed = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
    except OSError:
        pass


def _echo(data: bytes, echo_filter: Optional[Callable[[bytes], bytes]]) -> None:
    if echo_filter is not None:
        data = echo_filter(data)
    if data:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def _run_command_stream(
    cmd: Union[str, list],
    check: bool,
    shell: bool,
    echo_filter: Optional[Callable[[bytes], bytes]] = None,
) -> subprocess.CompletedProcess:
    """Run a command, echoing bytes live (tqdm / \\r progress) while capturing text."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_MININTERVAL", "0.5")
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("COLUMNS", "120")
    env["TQDM_DISABLE"] = "0"

    master_fd, slave_fd = pty.openpty()
    _set_pty_winsize(master_fd)
    _set_pty_winsize(slave_fd)
    try:
        proc = subprocess.Popen(
            cmd,
            shell=shell,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        os.close(slave_fd)

    chunks: list[bytes] = []
    interrupted = False

    def _drain(timeout: float) -> bool:
        ready, _, _ = select.select([master_fd], [], [], timeout)
        if not ready:
            return False
        try:
            data = os.read(master_fd, 8192)
        except OSError:
            return False
        if not data:
            return False
        chunks.append(data)
        _echo(data, echo_filter)
        return True

    try:
        while True:
            _drain(0.2)
            if proc.poll() is not None:
                while _drain(0.05):
                    pass
                break
    except KeyboardInterrupt:
        interrupted = True
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except OSError:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
        while _drain(0.05):
            pass
        raise
    finally:
        os.close(master_fd)
        if proc.poll() is None and not interrupted:
            proc.wait()
        leftover = getattr(echo_filter, "flush", None)
        if callable(leftover):
            extra = leftover()
            if extra:
                sys.stdout.buffer.write(extra)
                sys.stdout.buffer.flush()

    output = b"".join(chunks).decode("utf-8", errors="replace")
    result = subprocess.CompletedProcess(cmd, proc.returncode or 0, stdout=output, stderr="")
    return _fail_if_needed(result, check, cmd)


def run_command(
    cmd: Union[str, list],
    check: bool = True,
    capture_output: bool = True,
    shell: bool = True,
    stream: bool = False,
    echo_filter: Optional[Callable[[bytes], bytes]] = None,
) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.

    Args:
        cmd: Command to run
        check: Raise exception on non-zero exit
        capture_output: Capture stdout/stderr (ignored when stream=True)
        shell: Run in shell mode
        stream: Echo output live (PTY) so tqdm / llama.cpp progress is visible
        echo_filter: Optional transform applied only to echoed bytes (full output is still captured)

    Returns:
        CompletedProcess instance
    """
    if stream:
        return _run_command_stream(
            cmd, check=check, shell=shell, echo_filter=echo_filter
        )

    result = subprocess.run(
        cmd,
        shell=shell,
        capture_output=capture_output,
        text=True,
    )
    return _fail_if_needed(result, check, cmd)


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
