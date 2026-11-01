"""
Cross-Platform Test Runner for Backend (Virtual Environment Detector)
--------------------------------------------------------------------
Automatically detects and uses local backend virtual environment (Windows/Linux/Mac),
falling back to system Python if already activated.
"""

import os
import sys
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Potential virtual environment python binaries across OSes
VENV_PYTHON_CANDIDATES = [
    os.path.join(REPO_ROOT, "backend", "venv", "Scripts", "python.exe"),       # Windows venv
    os.path.join(REPO_ROOT, "backend", "venv", "bin", "python"),               # Linux/macOS venv
    os.path.join(REPO_ROOT, "backend", ".venv", "Scripts", "python.exe"),      # Alternate Windows
    os.path.join(REPO_ROOT, "backend", ".venv", "bin", "python"),              # Alternate Linux/macOS
]

def find_python_executable():
    for candidate in VENV_PYTHON_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return sys.executable

def main():
    py_bin = find_python_executable()
    args = sys.argv[1:] if len(sys.argv) > 1 else ["backend/tests/", "-v"]

    # Ensure backend directory is in PYTHONPATH
    env = os.environ.copy()
    backend_dir = os.path.join(REPO_ROOT, "backend")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{backend_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else backend_dir

    cmd = [py_bin, "-m", "pytest"] + args
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
