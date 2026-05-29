# Usage:
# source venvs/core/Scripts/activate
# python src/audio.py

from pathlib import Path
import subprocess
import shutil

# Project root (src/audio.py -> project root directory)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input and output file paths
VIDEO_PATH = ROOT_DIR / "data" / "output" / "video_blurred.mp4"
AUDIO_PATH = ROOT_DIR / "data" / "output" / "clean_audio.wav"
OUTPUT_PATH = ROOT_DIR / "data" / "output" / "final_video.mp4"

# Local bundled ffmpeg (recommended for portability across machines)
LOCAL_FFMPEG = ROOT_DIR / "tools" / "ffmpeg.exe"


def resolve_ffmpeg():
    """
    Resolves ffmpeg executable path in a portable way.

    Priority order:
    1. Project-local ffmpeg (tools/ffmpeg.exe)
    2. System-installed ffmpeg (PATH)
    """

    # Check if ffmpeg exists inside the project
    if LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)

    # Fallback to system PATH ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    # If nothing is found, stop execution with clear error
    raise RuntimeError(
        "ffmpeg not found. Install tools/ffmpeg.exe or add ffmpeg to PATH."
    )


def merge(video_file, audio_file, output_file):

    # Resolve ffmpeg path (portable or system fallback)
    ffmpeg = resolve_ffmpeg()

    # Ensures the input video exists before processing
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    # Ensures the input audio exists before processing
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio not found: {audio_file}")

    # Ensures output directory exists before writing audio file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command for audio extraction
    cmd = [
        ffmpeg,
        "-y",               # overwrite output file if exists
        "-i", str(video_file),
        "-i", str(audio_file),
        "-map", "0:v:0",    # take video from first input
        "-map", "1:a:0",    # take audio from second input
        "-c:v", "copy",     # copy video stream (no re-encoding)
        "-c:a", "aac",      # encode audio to AAC (widely supported)
        "-b:a", "192k",     # set audio bitrate
        str(output_file)
    ]

    try:
        # Execute ffmpeg process
        subprocess.run(cmd, check=True)

        # Success message
        print(f"Final video successfully saved:\n{output_file}")

    except subprocess.CalledProcessError as e:
        # Handle ffmpeg execution errors
        raise RuntimeError(f"Video merging failed: {e}")

def main():
    merge(VIDEO_PATH, AUDIO_PATH, OUTPUT_PATH)

    print(f"Final video is in: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()