# uses pyannote to diarize every speaker
# gives every speaker an ID and timestamps (start, end)
# outputs json whith all information above
import json;
import os;
from pyannote.audio import Pipeline;
from dotenv import load_dotenv;

# Loads environment variables from a local .env file
load_dotenv()

# Retrieves the Hugging Face authentication token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF token missing")

# Resolves the project base directory used for all file paths
from src.config import BASE_DIR

# Input and output file paths
AUDIO_FILE = os.path.join(BASE_DIR, "data", "input", "audio.wav")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "output", "diarization.json")

# Initializes the pretrained pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN
)

def diarize():
    # Verifies that the input audio file exists before processing begins
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"No audio file was found {AUDIO_FILE}")

    # Ensures that the output directory exists before writing results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Runs speaker diarization on the input audio file
    diarization = pipeline(AUDIO_FILE)

    # Stores diarization results as a list of speaker segments
    file = []

    # Iterates over speaker turns with timestamps and labels
    # Each segment represents a continuous speech region by one speaker
    for turn, _, speaker in diarization.speaker_diarization.itertracks(
        yield_label=True
    ):
        file.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })

     # Writes the diarization results to a JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(file, f, indent=4, ensure_ascii=False)

    # Indicates successful completion and output location
    print(f"Diarization ready in {OUTPUT_FILE}")

if __name__ == "__main__":
    diarize()