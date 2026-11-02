"""
Cross-Platform ML Command Runner (Smart Virtual Environment Detector)
---------------------------------------------------------------------
Automatically detects and uses local backend virtual environment (Windows/Linux/Mac),
preventing 'ModuleNotFoundError: No module named pandas' when executing ML preprocessor
or dataset verification scripts on host machines.
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
    os.path.join(REPO_ROOT, "ml_engine", "venv_ml", "Scripts", "python.exe"),  # ML dedicated venv
    os.path.join(REPO_ROOT, "ml_engine", "venv_ml", "bin", "python"),          # Linux ML venv
]


def find_python_executable():
    for candidate in VENV_PYTHON_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return sys.executable


COMMAND_MAP = {
    "prepare": os.path.join(REPO_ROOT, "ml_engine", "pipeline", "prepare_dataset.py"),
    "verify": os.path.join(REPO_ROOT, "ml_engine", "scripts", "verify_dataset_readiness.py"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMAND_MAP:
        print("Usage: python scripts/run_ml_command.py [prepare|verify] [extra_args...]")
        sys.exit(1)

    cmd_type = sys.argv[1]
    target_script = COMMAND_MAP[cmd_type]
    extra_args = sys.argv[2:]

    py_bin = find_python_executable()

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else REPO_ROOT
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [py_bin, target_script] + extra_args
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

