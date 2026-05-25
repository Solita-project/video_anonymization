# Extracts audio from data/input/video.mp4
# Saves the extracted audio to data/input/audio.wav
# Usage:
# source venvs/core/Scripts/activate
# python src/audio.py

from pathlib import Path
import shutil
import subprocess


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define input video path
VIDEO_PATH = ROOT_DIR / "data" / "input" / "video.mp4"

# Define output audio path
AUDIO_PATH = ROOT_DIR / "data" / "input" / "audio.wav"

# Define local FFmpeg path
LOCAL_FFMPEG = ROOT_DIR / "tools" / "ffmpeg.exe"


def get_ffmpeg():
    # Use local ffmpeg.exe if it exists
    if LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)

    # Otherwise use ffmpeg from system PATH
    return shutil.which("ffmpeg") or "ffmpeg"


def extract_audio():
    # Get FFmpeg command
    ffmpeg = get_ffmpeg()

    # Create output folder
    AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    command = [
        ffmpeg,
        "-y",
        "-i", str(VIDEO_PATH),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(AUDIO_PATH),
    ]

    # Run FFmpeg
    subprocess.run(command, check=True)

    # Show output file
    print(f"Audio saved: {AUDIO_PATH}")


if __name__ == "__main__":
    # Start audio extraction
    extract_audio()