# Transcribes data/input/audio.wav with WhisperX
# Creates word-level timestamps and saves them to data/output/transcription.json
# Usage:
# source venvs/transcript/Scripts/activate
# python run/run_transcript.py

from pathlib import Path
import json
import os
import shutil

import torch
import whisperx


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define input audio path
AUDIO_FILE = ROOT_DIR / "data" / "input" / "audio.wav"

# Define output JSON path
OUTPUT_FILE = ROOT_DIR / "data" / "output" / "transcription.json"

# Define local tools folder for ffmpeg
TOOLS_DIR = ROOT_DIR / "tools"

# Set WhisperX model
MODEL_NAME = "turbo"

# Set transcription language
LANGUAGE = "fi"

# Set batch size for transcription
BATCH_SIZE = 2


def add_ffmpeg_to_path():
    # Add tools folder to PATH
    if TOOLS_DIR.exists():
        os.environ["PATH"] = str(TOOLS_DIR) + os.pathsep + os.environ.get("PATH", "")


def check_ffmpeg():
    # Make local ffmpeg visible to WhisperX
    add_ffmpeg_to_path()

    # Stop if ffmpeg cannot be found
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg not found")


def get_device():
    # Use GPU if available
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_type(device):
    # Use float16 on GPU and int8 on CPU
    return "float16" if device == "cuda" else "int8"


def save_result(result):
    # Create output folder
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Store simplified transcript segments
    transcript = []

    # Read WhisperX segments
    for segment in result.get("segments", []):

        # Store words from this segment
        words = []

        # Read word-level timestamps
        for word in segment.get("words", []):

            # Skip words without timestamps
            if "start" not in word or "end" not in word:
                continue

            # Save one word
            words.append({
                "word": word.get("word", "").strip(),
                "start": round(float(word["start"]), 2),
                "end": round(float(word["end"]), 2),
            })

        # Save one transcript segment
        transcript.append({
            "segment_start": round(float(segment["start"]), 2),
            "segment_end": round(float(segment["end"]), 2),
            "text": segment.get("text", "").strip(),
            "words": words,
        })

    # Write transcription JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)

    # Show output file
    print(f"Transcription saved: {OUTPUT_FILE}")


def transcribe():
    # Check that ffmpeg is available
    check_ffmpeg()

    # Choose GPU or CPU
    device = get_device()

    # Choose WhisperX compute type
    compute_type = get_compute_type(device)

    # Load WhisperX model
    model = whisperx.load_model(
        MODEL_NAME,
        device,
        compute_type=compute_type,
        language=LANGUAGE,
    )

    # Load audio file
    audio = whisperx.load_audio(str(AUDIO_FILE))

    # Run transcription
    result = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
    )

    # Load alignment model
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )

    # Align words with timestamps
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Save transcription JSON
    save_result(result)

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Start transcription
    transcribe()