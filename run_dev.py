from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from backend.app.config import DEFAULT_HOST, DEFAULT_PORT


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
FRONTEND_DIST_INDEX = FRONTEND / "dist" / "index.html"


def ensure_frontend_build() -> None:
    if FRONTEND_DIST_INDEX.exists():
        return

    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise SystemExit(
            "The frontend has not been built yet, and npm was not found on PATH.\n"
            "Install Node.js, then run `python run_dev.py` again."
        )

    if not (FRONTEND / "node_modules").exists():
        print("Installing frontend dependencies...")
        subprocess.run([npm, "ci"], cwd=FRONTEND, check=True)

    print("Building frontend...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)


if __name__ == "__main__":
    import uvicorn

    try:
        ensure_frontend_build()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

    uvicorn.run("backend.app.main:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=True)
