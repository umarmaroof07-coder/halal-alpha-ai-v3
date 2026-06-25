"""
Background job subprocess entry point.

Invoked by dashboard/jobs.py via Popen:
    python3 -m dashboard._job_runner <job_name> [main.py args...]

Runs the real command, then updates the status JSON with the result.
Streamlit is NOT imported here — this runs in a plain Python process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 -m dashboard._job_runner <job_name> [args...]")
        sys.exit(1)

    job_name  = sys.argv[1]
    main_args = sys.argv[2:]

    jobs_dir    = Path("data/cache/jobs")
    status_path = jobs_dir / f"{job_name}_status.json"
    log_path    = jobs_dir / f"{job_name}.log"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Run the real main.py command (blocking — this subprocess owns the wait)
    proc = subprocess.run(
        [sys.executable, "main.py"] + main_args,
        capture_output=True,
        text=True,
    )

    combined = proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
    log_path.write_text(combined)

    lines = combined.splitlines()

    try:
        with status_path.open() as f:
            s = json.load(f)
    except Exception:
        s = {"job_name": job_name, "args": main_args}

    s.update({
        "status":      "completed" if proc.returncode == 0 else "failed",
        "exit_code":   proc.returncode,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "last_lines":  lines[-25:],
    })

    with status_path.open("w") as f:
        json.dump(s, f, indent=2)


if __name__ == "__main__":
    main()
