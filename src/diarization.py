from pathlib import Path
import json
import os

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# Load .env variables
load_dotenv()

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Input/output paths
AUDIO_FILE = ROOT_DIR / "data" / "input" / "audio.wav"

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "diarization.json"
)

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN missing in .env"
    )


def get_device():
    """
    Automatically select GPU if available,
    otherwise fallback to CPU.
    """

    return "cuda" if torch.cuda.is_available() else "cpu"


def diarize():

    # Verify audio exists
    if not AUDIO_FILE.exists():

        raise FileNotFoundError(
            f"Audio not found:\n{AUDIO_FILE}"
        )

    # Create output folder
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    print(
        f"Loading pyannote pipeline on: {device}"
    )

    try:

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN
        )

        pipeline.to(
            torch.device(device)
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to load pipeline:\n{e}"
        )

    try:

        diarization = pipeline(
            str(AUDIO_FILE)
        )

    except Exception as e:

        raise RuntimeError(
            f"Diarization failed:\n{e}"
        )

    segments = []

    for turn, _, speaker in (
        diarization.speaker_diarization.itertracks(
            yield_label=True
        )
    ):

        segments.append({

            "start": round(
                turn.start,
                2
            ),

            "end": round(
                turn.end,
                2
            ),

            "speaker": speaker
        })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            segments,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nDiarization complete"
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    diarize()

# test:
# source venvs/speaker_gpu/Scripts/activate
# python src/diarization.py