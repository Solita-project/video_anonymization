from pathlib import Path
import json
import os
import shutil

import torch
import whisperx


# Resolve project root:
# src/transcript.py -> project root
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


# Project-local tools directory.
# On Windows this should contain:
#     tools/ffmpeg.exe
TOOLS_DIR = ROOT_DIR / "tools"


# WhisperX configuration
MODEL_NAME = "turbo"

# Finnish language
LANGUAGE = "fi"

# Lower values use less VRAM.
# Increase this later if the GPU has enough memory.
BATCH_SIZE = 2


def add_tools_to_path():
    """
    Add the project-local tools directory to PATH.

    WhisperX calls ffmpeg internally when loading audio.
    This makes tools/ffmpeg.exe discoverable without requiring
    a global FFmpeg installation.
    """

    if not TOOLS_DIR.exists():
        return

    current_path = os.environ.get(
        "PATH",
        "",
    )

    os.environ["PATH"] = (
        str(TOOLS_DIR)
        + os.pathsep
        + current_path
    )


def verify_ffmpeg():
    """
    Verify that ffmpeg is available.

    Raises:
        FileNotFoundError: If ffmpeg cannot be found.
    """

    add_tools_to_path()

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )

    if not ffmpeg_path:
        raise FileNotFoundError(
            "ffmpeg was not found.\n\n"
            "Expected one of these:\n"
            f"  {TOOLS_DIR / 'ffmpeg.exe'}\n"
            "  or ffmpeg available globally in PATH"
        )

    print(
        f"ffmpeg found: {ffmpeg_path}"
    )


def get_device():
    """
    Automatically choose GPU when available.

    Returns:
        str: 'cuda' if CUDA is available, otherwise 'cpu'.
    """

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def get_compute_type(device):
    """
    Select a suitable compute type.

    GPU:
        float16

    CPU:
        int8

    Args:
        device (str): 'cuda' or 'cpu'.

    Returns:
        str: WhisperX compute type.
    """

    if device == "cuda":
        return "float16"

    return "int8"


def print_system_info(device, compute_type):
    """
    Print hardware and WhisperX runtime information.

    Args:
        device (str): Selected device.
        compute_type (str): Selected compute type.
    """

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
        f"Compute type: {compute_type}"
    )

    print(
        "=======================\n"
    )


def load_audio():
    """
    Load audio with WhisperX.

    Returns:
        Any: Audio object returned by whisperx.load_audio().
    """

    if not AUDIO_FILE.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_FILE}"
        )

    verify_ffmpeg()

    print(
        "Loading audio..."
    )

    return whisperx.load_audio(
        str(AUDIO_FILE)
    )


def save_transcription(result):
    """
    Save WhisperX transcription result as a simplified JSON file.

    Args:
        result (dict): WhisperX aligned transcription result.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    transcription = []

    for segment in result.get(
        "segments",
        [],
    ):

        words = []

        for word in segment.get(
            "words",
            [],
        ):

            if (
                "start" in word
                and "end" in word
            ):

                words.append({

                    "word":
                    word.get(
                        "word",
                        "",
                    ).strip(),

                    "start":
                    round(
                        float(word["start"]),
                        2,
                    ),

                    "end":
                    round(
                        float(word["end"]),
                        2,
                    ),

                })

        transcription.append({

            "segment_start":
            round(
                float(segment["start"]),
                2,
            ),

            "segment_end":
            round(
                float(segment["end"]),
                2,
            ),

            "text":
            segment.get(
                "text",
                "",
            ).strip(),

            "words":
            words,

        })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            transcription,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\nTranscription complete"
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


def transcribe():
    """
    Run the complete WhisperX transcription and alignment pipeline.
    """

    device = get_device()

    compute_type = get_compute_type(
        device
    )

    print_system_info(
        device,
        compute_type,
    )

    try:

        print(
            "Loading WhisperX model..."
        )

        model = whisperx.load_model(
            MODEL_NAME,
            device,
            compute_type=compute_type,
            language=LANGUAGE,
        )

        audio = load_audio()

        print(
            "Starting transcription..."
        )

        result = model.transcribe(
            audio,
            batch_size=BATCH_SIZE,
        )

        print(
            "Loading alignment model..."
        )

        model_a, metadata = whisperx.load_align_model(
            language_code=result[
                "language"
            ],
            device=device,
        )

        print(
            "Aligning words..."
        )

        result = whisperx.align(
            result[
                "segments"
            ],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

    except Exception as e:

        raise RuntimeError(
            f"Transcription failed:\n{e}"
        )

    save_transcription(
        result
    )

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


if __name__ == "__main__":

    transcribe()