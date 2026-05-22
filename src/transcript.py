# This script:
# 1. Loads an extracted WAV audio file
# 2. Uses WhisperX to transcribe speech
# 3. Aligns spoken words with precise timestamps
# 4. Saves structured transcription output as JSON


from pathlib import Path
import json

import torch
import whisperx


# Resolve project root directory automatically
# Makes file paths work on Windows, macOS and Linux
ROOT_DIR = Path(__file__).resolve().parent.parent


# Input and output file locations
AUDIO_FILE = (
    ROOT_DIR
    / "data"
    / "input"
    / "audio.wav"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "transcription.json"
)


# WhisperX configuration
MODEL_NAME = "turbo"

# Finnish language
LANGUAGE = "fi"


def get_device():
    """
    Automatically choose GPU when available.

    Returns:
        str: "cuda" or "cpu"
    """

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_compute_type(device):
    """
    Select suitable compute type based on device.

    GPU:
        float16

    CPU:
        int8

    Args:
        device (str)

    Returns:
        str
    """

    if device == "cuda":
        return "float16"

    return "int8"


def transcribe():
    """
    Main transcription pipeline.
    """

    # Verify audio file exists before processing starts
    if not AUDIO_FILE.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_FILE}"
        )

    # Create output folder automatically
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Detect available hardware
    device = get_device()

    compute_type = get_compute_type(
        device
    )

    print(
        "\n===== SYSTEM INFO ====="
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "Running on CPU"
        )

    print(
        f"Device: {device}"
    )

    print(
        f"Compute type: "
        f"{compute_type}"
    )

    print(
        "=======================\n"
    )

    try:

        # Load WhisperX speech model
        print(
            "Loading WhisperX model..."
        )

        model = whisperx.load_model(
            MODEL_NAME,
            device,
            compute_type=compute_type,
            language=LANGUAGE
        )

        # Load audio into memory
        print(
            "Loading audio..."
        )

        audio = whisperx.load_audio(
            str(AUDIO_FILE)
        )

        # Perform initial transcription
        print(
            "Starting transcription..."
        )

        result = model.transcribe(
            audio,
            batch_size=2
        )

        # Load alignment model
        # Used to improve timestamp accuracy
        print(
            "Loading alignment model..."
        )

        model_a, metadata = (
            whisperx.load_align_model(
                language_code=result[
                    "language"
                ],
                device=device
            )
        )

        # Align words precisely
        print(
            "Aligning words..."
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

    except Exception as e:

        raise RuntimeError(
            f"Transcription failed:\n{e}"
        )

    # Final JSON structure
    transcription = []

    # Iterate through each transcription segment
    for segment in result["segments"]:

        words = []

        # Extract word-level timestamps
        for word in segment.get(
            "words",
            []
        ):

            if (
                "start" in word
                and "end" in word
            ):

                words.append({

                    "word":
                    word["word"].strip(),

                    "start":
                    round(
                        word["start"],
                        2
                    ),

                    "end":
                    round(
                        word["end"],
                        2
                    )

                })

        # Store complete segment structure
        transcription.append({

            "segment_start":
            round(
                segment["start"],
                2
            ),

            "segment_end":
            round(
                segment["end"],
                2
            ),

            "text":
            segment["text"].strip(),

            "words":
            words
        })

    # Save formatted JSON output
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            transcription,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        "\nTranscription complete"
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

# TEST!
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == "__main__":
    transcribe()