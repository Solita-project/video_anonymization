# This script:
# 1. Loads a WAV audio file
# 2. Transcribes speech using WhisperX
# 3. Aligns every spoken word with timestamps
# 4. Saves the transcription as JSON
import json
import os
import torch
import whisperx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Prints CUDA availability for debugging purposes
print(f"CUDA available: {torch.cuda.is_available()}")

# Reports GPU information if CUDA is available, otherwise falls back to CPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Resolves project base directory (root of the project inside container)
from src.config import BASE_DIR

# Input and output file paths
AUDIO_FILE = os.path.join(BASE_DIR, "data", "input", "audio.wav")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "output", "transcription.json")

# WhisperX configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPUTE_TYPE = (
    "float16"
    if DEVICE == "cuda"
    else "int8"
)

MODEL_NAME = "turbo"
LANGUAGE = "fi"

print(f"Using device: {DEVICE}")
print(f"Compute type: {COMPUTE_TYPE}")


def transcribe():
    # Ensure the audio file exists before processing
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

    # Create output directory if it does not already exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load WhisperX transcription model
    print("Loading WhisperX model...")
    model = whisperx.load_model(
        MODEL_NAME,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
    )

    # Load audio into memory
    print("Loading audio...")
    audio = whisperx.load_audio(AUDIO_FILE)

    # Run WhisperX transcription
    print("Starting transcription...")
    result = model.transcribe(audio, batch_size=2)

    # Load alignment model for accurate word-level timestamps
    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=DEVICE,
    )

    # Align words with precise timestamps
    print("Aligning words...")
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    # Final JSON structure
    transcription = []

    # Process every transcription segment
    for segment in result["segments"]:
        # Store word-level timestamps for the current segment
        words = []

        # Extract every aligned word and its timestamps
        for word in segment.get("words", []):
            if "start" in word and "end" in word:
                words.append({
                    "word": word["word"].strip(),
                    "start": round(word["start"], 2),
                    "end": round(word["end"], 2),
                })

        # Appends structured segment data to final output
        transcription.append(
            {
                "segment_start": round(segment["start"], 2),
                "segment_end": round(segment["end"], 2),
                "text": segment["text"].strip(),
                "words": words,
            }
        )

    # Save transcription result as formatted JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(transcription, f, indent=4, ensure_ascii=False)

        print(f"Transcription successfully saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    transcribe()
