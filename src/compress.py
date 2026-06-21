# Usage:
# source venvs/core/Scripts/activate
# python src/compress.py

from pathlib import Path
import subprocess
import shutil


# Project root: src/compress.py -> project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input and output file paths
INPUT_VIDEO_PATH = ROOT_DIR / "data" / "output" / "video.mp4"
OUTPUT_VIDEO_PATH = ROOT_DIR / "data" / "output" / "video_compressed.mp4"

# Local bundled ffmpeg
LOCAL_FFMPEG = ROOT_DIR / "tools" / "ffmpeg.exe"


def resolve_ffmpeg():
    """
    Resolves ffmpeg executable path in a portable way.

    Priority order:
    1. Project-local ffmpeg: tools/ffmpeg.exe
    2. System-installed ffmpeg from PATH
    """

    if LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)

    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "ffmpeg not found. Install tools/ffmpeg.exe or add ffmpeg to PATH."
    )


def compress(video_file, output_file):
    """
    Compresses the input video and saves it as output_file.
    """

    ffmpeg = resolve_ffmpeg()

    if not video_file.exists():
        raise FileNotFoundError(f"Input video not found: {video_file}")

    if video_file.resolve() == output_file.resolve():
        raise RuntimeError(
            "Input and output video paths are the same. "
            "Please use a different output filename."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",                    # overwrite output file if it exists
        "-i", str(video_file),   # input video file
        "-c:v", "libx264",       # encode video to H.264
        "-b:v", "800k",          # video bitrate
        "-preset", "medium",     # encoding speed/quality tradeoff
        "-pix_fmt", "yuv420p",   # compatibility with most players
        "-c:a", "aac",           # encode audio to AAC
        "-b:a", "128k",          # audio bitrate
        str(output_file),
    ]

    print("Running ffmpeg command:")
    print(subprocess.list2cmdline(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"Compressed video successfully saved:\n{output_file}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video compression failed: {e}") from e


def main():
    print(f"Input video: {INPUT_VIDEO_PATH}")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")

    compress(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)

    print(f"Final video is in: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()