# This script:
# 1. Loads final_transcript.json
# 2. Extracts cleaned speaker text
# 3. Loads Chatterbox TTS model
# 4. Generates speech audio
# 5. Saves clean_audio.wav


from pathlib import Path
import json

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS


# Resolve project root automatically
ROOT_DIR = Path(__file__).resolve().parent.parent


# Input/output paths
FINAL_TRANSCRIPT_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "final_transcript.json"
)

VOICE_FILE = (
    ROOT_DIR
    / "voices"
    / "speaker.wav"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "clean_audio.wav"
)


def get_device():
    """
    Automatically use GPU if available.
    Otherwise fallback to CPU.
    """

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_transcript():
    """
    Load transcript JSON.
    """

    if not FINAL_TRANSCRIPT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{FINAL_TRANSCRIPT_FILE}"
        )

    with open(
        FINAL_TRANSCRIPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def build_text(data):
    """
    Combine transcript segments
    into one text block.
    """

    text_parts = []

    for segment in data.get(
        "segments",
        []
    ):

        text = (
            segment.get(
                "text",
                ""
            )
            .strip()
        )

        if text:

            text_parts.append(
                text
            )

    return " ".join(
        text_parts
    )


def generate_audio():

    device = get_device()

    print(
        f"\nUsing device: {device}"
    )

    # Verify voice sample exists
    if not VOICE_FILE.exists():

        raise FileNotFoundError(
            f"Missing voice file:\n{VOICE_FILE}"
        )

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Loading transcript..."
    )

    data = load_transcript()

    print(
        "Preparing text..."
    )

    text = build_text(
        data
    )

    if not text:

        raise ValueError(
            "No text found in transcript"
        )

    try:

        print(
            "Loading TTS model..."
        )

        model = (
            ChatterboxTTS
            .from_pretrained(
                "Finnish-NLP/Chatterbox-Finnish",
                device=device
            )
        )

        print(
            "Generating speech..."
        )

        wav = model.generate(

            text,

            audio_prompt_path=
            str(VOICE_FILE)

        )

    except Exception as e:

        raise RuntimeError(
            f"TTS generation failed:\n{e}"
        )

    # Ensure waveform has channel dimension
    if wav.dim() == 1:

        wav = wav.unsqueeze(
            0
        )

    print(
        "Saving audio..."
    )

    ta.save(

        str(OUTPUT_FILE),

        wav.cpu(),

        model.sr

    )

    print(
        f"\nAudio saved:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    generate_audio()