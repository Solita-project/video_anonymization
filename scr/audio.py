import os
import subprocess

# Resolves project base directory (root of the project inside container)
from src.config import BASE_DIR

# Input and output file paths
VIDEO_PATH = os.path.join(BASE_DIR, "data", "input", "video.mp4")
AUDIO_PATH = os.path.join(BASE_DIR, "data", "input", "audio.wav")

# Extract audio from a video file using ffmpeg
def extract_audio():
    """
    Extracts mono 16kHz WAV audio from video using ffmpeg.
    """

    # Ensures the input video exists before attempting extraction
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # Builds an ffmpeg command that converts video audio into a standardized WAV format
    cmd = [
        "ffmpeg",
        "-y",
        "-i", VIDEO_PATH,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        AUDIO_PATH,
    ]

    # Executes the ffmpeg process and raises an error if it fails
    subprocess.run(cmd, check=True)

    # Confirms successful audio extraction and output location
    print(f"Audio saved to {AUDIO_PATH}")