# Runs speaker diarization in the speaker virtual environment.
# Usage:
# python run/run_speaker.py

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "src" / "diarization.py"


# Select speaker venv Python
if os.name == "nt":
    PYTHON = ROOT_DIR / "venvs" / "speaker" / "Scripts" / "python.exe"
else:
    PYTHON = ROOT_DIR / "venvs" / "speaker" / "bin" / "python"


# Run diarization script
subprocess.run(
    [
        str(PYTHON),
        str(SCRIPT),
    ],
    cwd=ROOT_DIR,
    check=True,
)