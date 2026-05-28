# Runs transcript anonymization in the transcript virtual environment.
# Usage:
# python run/run_anonymization.py

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "src" / "anonymize.py"


# Select transcript venv Python
if os.name == "nt":
    PYTHON = ROOT_DIR / "venvs" / "transcript" / "Scripts" / "python.exe"
else:
    PYTHON = ROOT_DIR / "venvs" / "transcript" / "bin" / "python"


# Run anonymization script
subprocess.run(
    [
        str(PYTHON),
        str(SCRIPT),
    ],
    cwd=ROOT_DIR,
    check=True,
)