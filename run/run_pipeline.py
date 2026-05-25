# Runs the full video anonymization pipeline.
# Usage:
# source venvs/core/Scripts/activate
# python run/run_pipeline.py

from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent


def run_step(title, script):
    # Run one pipeline step with the current core Python
    print(f"\n{title}")

    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / script),
        ],
        cwd=ROOT_DIR,
        check=True,
    )


def main():
    # Run all pipeline steps in the correct order
    print("\nVIDEO ANONYMIZATION PIPELINE")

    run_step("[1/6] Extract audio", "src/audio.py")
    run_step("[2/6] Transcription", "run/run_transcript.py")
    run_step("[3/6] Speaker diarization", "run/run_speaker.py")
    run_step("[4/6] Merge transcript", "src/merged.py")
    run_step("[5/6] Generate TTS", "run/run_tts.py")
    run_step("[6/6] Process video", "run/run_video.py")

    print("\nPipeline complete")


if __name__ == "__main__":
    main()