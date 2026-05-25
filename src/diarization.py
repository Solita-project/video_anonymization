from pathlib import Path
import json
import os
import wave

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent.parent

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
    / "diarization.json"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN missing in .env"
    )


def get_device():
    """
    Automatically select GPU if available.

    Returns:
        torch.device: CUDA device when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def load_wav_without_torchcodec():
    """
    Load a PCM WAV file without torchaudio or TorchCodec.

    The audio.py step creates:
        - WAV
        - 16 kHz
        - mono
        - signed 16-bit PCM

    This function still handles mono/stereo safely.

    Returns:
        dict: Pyannote-compatible dictionary:
            {
                "waveform": torch.Tensor with shape (channels, samples),
                "sample_rate": int
            }
    """

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Audio not found:\n{AUDIO_FILE}"
        )

    print(
        "Loading audio without TorchCodec..."
    )

    with wave.open(
        str(AUDIO_FILE),
        "rb",
    ) as wav_file:

        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()

        raw_audio = wav_file.readframes(
            frames
        )

    if sample_width != 2:
        raise ValueError(
            "Unsupported WAV format.\n"
            f"Expected 16-bit PCM WAV, got sample width: {sample_width} bytes.\n"
            "Make sure src/audio.py creates pcm_s16le WAV audio."
        )

    audio = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).astype(
        np.float32
    )

    # Convert int16 PCM to float32 range [-1.0, 1.0].
    audio = audio / 32768.0

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels,
        )

        # Convert stereo/multi-channel audio to mono.
        audio = audio.mean(
            axis=1,
        )

    # Pyannote expects shape: (channels, samples)
    waveform = torch.from_numpy(
        audio
    ).unsqueeze(
        0
    )

    print(
        f"Audio loaded: {sample_rate} Hz, "
        f"{waveform.shape[0]} channel(s), "
        f"{waveform.shape[1]} samples"
    )

    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def diarize():
    """
    Run speaker diarization and save the result as JSON.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        f"Loading pyannote pipeline on: {device}"
    )

    try:

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN,
        )

        pipeline.to(
            device
        )

    except Exception as e:

        raise RuntimeError(
            f"Failed to load pipeline:\n{e}"
        )

    try:

        audio = load_wav_without_torchcodec()

        print(
            "Starting speaker diarization..."
        )

        diarization = pipeline(
            audio
        )

    except Exception as e:

        raise RuntimeError(
            f"Diarization failed:\n{e}"
        )

    segments = []

    annotation = getattr(
        diarization,
        "speaker_diarization",
        diarization,
    )

    for turn, _, speaker in annotation.itertracks(
        yield_label=True
    ):

        segments.append(
            {
                "start": round(
                    float(turn.start),
                    2,
                ),
                "end": round(
                    float(turn.end),
                    2,
                ),
                "speaker": speaker,
            }
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            segments,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\nDiarization complete"
    )

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    diarize()