from pathlib import Path
import subprocess
import os
import shutil

# Project root (src/audio.py -> project root directory)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input and output file paths
VIDEO_PATH = ROOT_DIR / "data" / "input" / "video.mp4"
AUDIO_PATH = ROOT_DIR / "data" / "input" / "audio.wav"

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


def extract_audio():
    """
    Extracts mono 16kHz WAV audio from a video file using ffmpeg.

    This ensures:
    - consistent sample rate (16kHz)
    - mono audio (1 channel)
    - PCM 16-bit format (compatible with speech models)
    """

    # Resolve ffmpeg path (portable or system fallback)
    ffmpeg = resolve_ffmpeg()

    # Ensures the input video exists before processing
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # Ensures output directory exists before writing audio file
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command for audio extraction
    cmd = [
        ffmpeg,
        "-y",               # overwrite output file if exists
        "-i", str(VIDEO_PATH),
        "-vn",             # disable video
        "-acodec", "pcm_s16le",  # 16-bit PCM WAV
        "-ar", "16000",    # 16 kHz sample rate (required for ASR models)
        "-ac", "1",        # mono audio
        str(AUDIO_PATH)
    ]

    try:
        # Execute ffmpeg process
        subprocess.run(cmd, check=True)

        # Success message
        print(f"Audio successfully saved:\n{AUDIO_PATH}")

    except subprocess.CalledProcessError as e:
        # Handle ffmpeg execution errors
        raise RuntimeError(f"Audio extraction failed: {e}")


if __name__ == "__main__":
    extract_audio()

# test:
# source venvs/core/Scripts/activate
# python src/audio.py