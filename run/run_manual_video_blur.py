# Wrapper for manual video blur.
# This script runs src/manual_video_blur.py with the video virtual environment.
# Usage:
# python run/run_manual_video_blur.py extract --time 12.5
# python run/run_manual_video_blur.py apply

from pathlib import Path
import subprocess
import sys


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define script path
SCRIPT_FILE = ROOT_DIR / "src" / "manual_video_blur.py"

# Define video virtual environment Python paths
WINDOWS_VIDEO_PYTHON = ROOT_DIR / "venvs" / "video" / "Scripts" / "python.exe"
LINUX_VIDEO_PYTHON = ROOT_DIR / "venvs" / "video" / "bin" / "python"


def get_video_python():
    # Use Windows video environment if it exists
    if WINDOWS_VIDEO_PYTHON.exists():
        return WINDOWS_VIDEO_PYTHON

    # Otherwise use Linux/macOS video environment
    if LINUX_VIDEO_PYTHON.exists():
        return LINUX_VIDEO_PYTHON

    raise FileNotFoundError("Video virtual environment Python was not found")


def main():
    # Pass all command line arguments to src/manual_video_blur.py
    video_python = get_video_python()

    command = [
        str(video_python),
        str(SCRIPT_FILE),
    ] + sys.argv[1:]

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()