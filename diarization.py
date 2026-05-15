# uses pyannote to diarize every speaker
# gives every speaker an ID and timestamps (start, end)
# outputs json whith all information above

# CPU powered

import json;
import os;
from pyannote.audio import Pipeline;
from dotenv import load_dotenv;

# For the .env file
load_dotenv()

# HF token listed in .env file
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF token missing")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_FILE = os.path.join(BASE_DIR, "input.wav")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "diarization.json")

pipeline = Pipeline.from_pretrained (
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN
)

def diarize():
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"No audio file was found {AUDIO_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    diarization = pipeline(AUDIO_FILE)

    file = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        file.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(file, f, indent=4, ensure_ascii=False)

    print(f"Diarization ready in {OUTPUT_FILE}")

if __name__ == "__main__":
    diarize()