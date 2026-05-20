# Gets clean_transcript.json
# Gets SPEAKER_ID.json
# Uses tts to create the new audio.wav file
# > final_transcript

import os
import json

from chatterbox.tts import ChatterboxTTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


FINAL_TRANSCRIPT_FILE = os.path.join(BASE_DIR, "output", "final_transcript.json")
VOICES_FILE = os.path.join(BASE_DIR, "voices", "*.wav")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "clean_audio.wav")

model = ChatterboxTTS.from_pretrained(
    "Finnish-NLP/Chatterbox-Finnish",
    device="cpu"
)

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

text = " ".join (
    segments["text"].strip()
    for segment in data["segments"]:
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]
        speaker = segment.get("speaker", "unknown")
)

# Take text from the final transcript
# Take speaker id from diarization
# Seperate speakers by speaker id
wav = model.generate(
    text,
    audio_prompt_path=VOICES_FILE
)

if wav.dim() == 1:
    wav = wav.unsqueeze(0)

os.makedirs(os.path.dirname(OUTPUT_FILE), exit_ok=True)

ta.save(
    OUTPUT_FILE,
    wav.cpu(),
    model.sr
)

print(f"Saved audio to {OUTPUT_FILE}")