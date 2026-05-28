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
    run_step("[1/7] Extract audio", "src/audio.py")
    run_step("[2/7] Transcription", "run/run_transcript.py")
    run_step("[3/7] Anonymize transcription", "run/run_anonymize.py")
    run_step("[4/7] Speaker diarization", "run/run_speaker.py")
    run_step("[5/7] Merge transcript", "src/merged.py")
    run_step("[6/7] Generate TTS", "run/run_tts.py")
    run_step("[7/7] Process video", "run/run_video.py")

    print("\nPipeline complete")


if __name__ == "__main__":
    main()