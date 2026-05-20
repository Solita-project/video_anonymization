# Main pipeline controller
#
# This script:
# 1. Detects available hardware
# 2. Runs all processing steps
# 3. Executes steps in correct order


import subprocess
import torch


def run_step(name, script):
    """
    Execute a pipeline step
    and stop if it fails.
    """

    print(
        f"\n{name}"
    )

    result = subprocess.run(
        ["python", script]
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Failed:\n{script}"
        )


def main():

    print(
        "\nVIDEO ANONYMIZATION PIPELINE"
    )

    print(
        "============================"
    )

    # Display hardware information
    if torch.cuda.is_available():

        print(
            f"GPU detected:"
        )

        print(
            torch.cuda.get_device_name(
                0
            )
        )

    else:

        print(
            "CPU mode"
        )

    # Step 1
    run_step(
        "[1/6] Extract audio",
        "src/audio.py"
    )

    # Step 2
    run_step(
        "[2/6] Transcription",
        "run/run_whisperx.py"
    )

    # Step 3
    run_step(
        "[3/6] Speaker diarization",
        "run/run_speaker.py"
    )

    # Step 4
    run_step(
        "[4/6] Merge transcript",
        "src/merged.py"
    )

    # Step 5
    run_step(
        "[5/6] Generate TTS",
        "run/run_tts.py"
    )

    # Step 6
    run_step(
        "[6/6] Video processing",
        "run/run_video.py"
    )

    print(
        "\nPipeline complete"
    )


if __name__ == "__main__":
    main()

# TEST:
# source venvs/core/Scripts/activate
# python src/main.py