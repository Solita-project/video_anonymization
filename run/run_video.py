# Runs video processing in the video virtual environment.
# Usage:
# python run/run_video.py

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "src" / "video.py"


# Select video venv Python
if os.name == "nt":
    PYTHON = ROOT_DIR / "venvs" / "video" / "Scripts" / "python.exe"
else:
    PYTHON = ROOT_DIR / "venvs" / "video" / "bin" / "python"


# Run video script
subprocess.run(
    [
        str(PYTHON),
        "-m",
        "src.video"
    ],
    cwd=ROOT_DIR,
    check=True,
)