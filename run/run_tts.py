# Runs text-to-speech generation in the TTS virtual environment.
# Usage:
# python run/run_tts.py

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "src" / "tts.py"


# Select TTS venv Python
if os.name == "nt":
    PYTHON = ROOT_DIR / "venvs" / "tts" / "Scripts" / "python.exe"
else:
    PYTHON = ROOT_DIR / "venvs" / "tts" / "bin" / "python"


# Run TTS script
subprocess.run(
    [
        str(PYTHON),
        str(SCRIPT),
    ],
    cwd=ROOT_DIR,
    check=True,
)