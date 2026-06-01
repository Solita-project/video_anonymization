# Runs the full video anonymization pipeline.
# Usage:
# source venvs/core/Scripts/activate
# python run/run_pipeline.py

from pathlib import Path
import subprocess
import sys
import time


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
    # Start timer
    start_time = time.perf_counter()

    # Run all pipeline steps in the correct order
    run_step("[1/8] Extract audio", "src/audio.py")
    run_step("[2/8] Transcription", "run/run_transcript.py")
    run_step("[3/8] Anonymize transcription", "run/run_anonymize.py")
    run_step("[4/8] Speaker diarization", "run/run_speaker.py")
    run_step("[5/8] Merge transcript", "src/merged.py")
    run_step("[6/8] Generate TTS", "run/run_tts.py")
    run_step("[7/8] Process video", "run/run_video.py")
    run_step("[8/8] Merge audio and video", "src/final_merge.py")

    # Stop timer
    end_time = time.perf_counter()

    # Calculate total runtime
    total_seconds = end_time - start_time
    total_minutes = total_seconds / 60

    print("\nPipeline complete")
    print(f"Total runtime: {total_minutes:.2f} minutes")


if __name__ == "__main__":
    main()