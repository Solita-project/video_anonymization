# Runs speaker diarization with Pyannote.
# Reads audio.wav without TorchCodec to avoid Windows audio loading errors.
# Saves speaker timestamps to data/output/diarization.json.
# Usage:
# source venvs/core/Scripts/activate
# python run/run_speaker.py

from pathlib import Path
import json
import os
import wave

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# Load .env file
load_dotenv()

# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Set input and output paths
AUDIO_FILE = ROOT_DIR / "data" / "input" / "audio.wav"
OUTPUT_FILE = ROOT_DIR / "data" / "output" / "diarization.json"

# Read Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")


def load_audio():
    # Read WAV file without TorchCodec
    with wave.open(str(AUDIO_FILE), "rb") as f:
        channels = f.getnchannels()
        sample_rate = f.getframerate()
        audio = f.readframes(f.getnframes())

    # Convert WAV bytes to float audio
    audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    # Convert stereo to mono
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Convert audio to Pyannote format
    return {
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": sample_rate,
    }


def diarize():
    # Stop if Hugging Face token is missing
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN missing in .env")

    # Create output folder
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Pyannote pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
    )

    # Move pipeline to GPU or CPU
    pipeline.to(device)

    # Load audio without TorchCodec
    audio = load_audio()

    # Run speaker diarization
    diarization = pipeline(audio)

    # Get speaker timeline
    annotation = getattr(diarization, "speaker_diarization", diarization)

    # Collect speaker segments
    segments = []

    # Save start, end and speaker ID
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(float(turn.start), 2),
            "end": round(float(turn.end), 2),
            "speaker": speaker,
        })

    # Write JSON output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4, ensure_ascii=False)

    # Show result path
    print(f"Diarization saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    # Start diarization
    diarize()