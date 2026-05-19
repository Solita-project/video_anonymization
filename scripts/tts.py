# Gets clean_transcript.json
# Gets SPEAKER_ID.json
# Uses tts to create the new audio.wav file

import os;
import json;

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEAKER_ID_FILE = os.path.join(BASE_DIR, "output", "diarization.json")
TRANSCRIPT_FILE = os.path.join(BASE_DIR, "output", "clean_transcript.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "clean_audio.wav")