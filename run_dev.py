from __future__ import annotations

import shutil
import socket
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
FRONTEND_DIST_INDEX = FRONTEND / "dist" / "index.html"
DEFAULT_HOST = os.environ.get("LLM_MODEL_LOADER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LLM_MODEL_LOADER_PORT", "8174"))


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


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) != 0


def select_port(host: str, preferred: int) -> int:
    if port_available(host, preferred):
        return preferred
    for port in range(preferred + 1, preferred + 20):
        if port_available(host, port):
            return port
    raise SystemExit(f"No free port found from {preferred} through {preferred + 19}.")


if __name__ == "__main__":
    import uvicorn

    try:
        ensure_frontend_build()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

    port = select_port(DEFAULT_HOST, DEFAULT_PORT)
    os.environ["LLM_MODEL_LOADER_HOST"] = DEFAULT_HOST
    os.environ["LLM_MODEL_LOADER_PORT"] = str(port)
    print(f"Starting LLM Model Loader at http://{DEFAULT_HOST}:{port}")

    uvicorn.run("backend.app.main:app", host=DEFAULT_HOST, port=port, reload=False)
