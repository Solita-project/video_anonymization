# Wrapper for video anonymization.
# This script runs src.video as a module with the video virtual environment.
# Usage:
# python run/run_video.py

from pathlib import Path
import argparse
import subprocess
import sys


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

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
    # Run src.video as a module so imports like src.video_config work
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["cpr", "intubation"],
        default="cpr",
        help="Video anonymization profile",
    )
    args = parser.parse_args()

    video_python = get_video_python()

    command = [
        str(video_python),
        "-c",
        (
            "from src.video import process_video; "
            f"process_video(profile_name={args.profile!r})"
        ),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
