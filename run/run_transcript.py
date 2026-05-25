# Runs WhisperX transcription in the transcript virtual environment.
# Usage:
# python run/run_transcript.py

from pathlib import Path
import os
import subprocess


ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ROOT_DIR / "src" / "transcript.py"
TOOLS_DIR = ROOT_DIR / "tools"


# Select transcript venv Python
if os.name == "nt":
    PYTHON = ROOT_DIR / "venvs" / "transcript" / "Scripts" / "python.exe"
else:
    PYTHON = ROOT_DIR / "venvs" / "transcript" / "bin" / "python"


# Add tools folder to PATH so WhisperX can find ffmpeg
env = os.environ.copy()
env["PATH"] = str(TOOLS_DIR) + os.pathsep + env.get("PATH", "")


# Run transcript script
subprocess.run(
    [
        str(PYTHON),
        str(SCRIPT),
    ],
    cwd=ROOT_DIR,
    env=env,
    check=True,
)