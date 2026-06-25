"""
Background job manager for long-running CLI commands.

Jobs run via subprocess.Popen (non-blocking). A child process
(dashboard._job_runner) owns the wait and writes the final status JSON
when done. The dashboard polls read_status() / poll_status() to show
progress without ever blocking the Streamlit event loop.

Status lifecycle:
    idle  →  running  →  completed
                      →  failed
                      →  interrupted   (runner died without finishing)

Status file: data/cache/jobs/<job_name>_status.json
Log file:    data/cache/jobs/<job_name>.log
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

JOBS_DIR = Path("data/cache/jobs")


def _status_path(name: str) -> Path:
    return JOBS_DIR / f"{name}_status.json"


def _log_path(name: str) -> Path:
    return JOBS_DIR / f"{name}.log"


# ---------------------------------------------------------------------------
# Status I/O
# ---------------------------------------------------------------------------

def read_status(name: str) -> dict:
    """Return the current status dict. Returns {"status": "idle"} if not found."""
    p = _status_path(name)
    if not p.exists():
        return {"status": "idle"}
    try:
        with p.open() as f:
            return json.load(f)
    except Exception:
        return {"status": "idle"}


def _write_status(name: str, s: dict) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with _status_path(name).open("w") as f:
        json.dump(s, f, indent=2)


def read_log_tail(name: str, n: int = 25) -> list[str]:
    """Return the last n lines of the job log."""
    p = _log_path(name)
    if not p.exists():
        return []
    return p.read_text().splitlines()[-n:]


# ---------------------------------------------------------------------------
# Process liveness check
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_running(name: str) -> bool:
    """Return True if a job with this name is currently running."""
    s = read_status(name)
    if s.get("status") != "running":
        return False
    pid = s.get("pid")
    if not pid:
        return False
    if _pid_alive(pid):
        return True
    # Runner died unexpectedly — mark interrupted
    s["status"] = "interrupted"
    s["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_status(name, s)
    return False


def start_job(name: str, main_args: list[str]) -> dict:
    """
    Start a background job if one isn't already running.

    Spawns `python3 -m dashboard._job_runner <name> [main_args...]` as a
    detached child process. Returns the initial status dict.

    If a job is already running, returns the current status without starting a
    new one.
    """
    if is_running(name):
        return read_status(name)

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    status: dict = {
        "status":      "running",
        "args":        main_args,
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code":   None,
        "last_lines":  [],
        "pid":         None,
    }
    _write_status(name, status)

    proc = subprocess.Popen(
        [sys.executable, "-m", "dashboard._job_runner", name] + main_args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path(__file__).resolve().parent.parent,
    )

    status["pid"] = proc.pid
    _write_status(name, status)
    return status


def poll_status(name: str) -> dict:
    """
    Read current status. If the job shows as "running" but the runner PID is
    gone, re-read the status file (the runner updates it on finish) or mark it
    interrupted if the file wasn't updated.
    """
    s = read_status(name)
    if s.get("status") != "running":
        return s

    pid = s.get("pid")
    if pid and not _pid_alive(pid):
        # Runner finished — re-read to pick up the update it wrote
        s2 = read_status(name)
        if s2.get("status") == "running":
            # Runner died without updating — interrupted
            s2["status"] = "interrupted"
            s2["finished_at"] = datetime.now(timezone.utc).isoformat()
            s2["last_lines"] = read_log_tail(name)
            _write_status(name, s2)
            return s2
        return s2

    return s


def any_job_running(*names: str) -> bool:
    return any(is_running(n) for n in names)
