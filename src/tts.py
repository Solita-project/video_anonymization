# Gets final_transcript.json (start, end, text, speaker_id)
# Gets voices/*.wav files and gives them to the speaker id's
# Loads Chatterbox TTS model
# Generates speech audio
# Usage:
# source venvs/tts/Scripts/activate
# python src/tts.py

from pathlib import Path
import json

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define input transcript path
FINAL_TRANSCRIPT_FILE = ROOT_DIR / "data" / "output" / "final_transcript.json"

# Define voice sample folder
VOICES_DIR = ROOT_DIR / "voices"

# Define final audio output path
OUTPUT_FILE = ROOT_DIR / "data" / "output" / "clean_audio.wav"

# Define temporary TTS segment folder
SEGMENTS_DIR = ROOT_DIR / "data" / "output" / "tts_segments"

# Choose GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Chatterbox model
model = ChatterboxTTS.from_pretrained(
    device=DEVICE,
)

# checking if voices directory exists
def get_voice():
    voices = sorted(VOICES_DIR.glob("*.wav"))
    if not voices:
        raise FileNotFoundError(f"No voice files found in {VOICES_DIR}")
    return voices


# mapping speakers to voices
def voice_map(segments, voices):
    speaker = sorted(set(seg.get("speaker", "unknown") for seg in segments))

    speaker_voice = {}
    for index, speaker in enumerate(speaker):
        voice = voices[index % len(voices)]
        speaker_voice[speaker] = voice

    return speaker_voice


def gen_seg(segment, voice_path):
    text = segment.get("text", "").strip()
    segment_id = segment["segment_id"]

    if not text:
        return None

    output_path = SEGMENTS_DIR / f"{segment_id}.wav"

    wav = model.generate(
        text,
        audio_prompt_path=str(voice_path)
    )

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    ta.save(str(output_path), wav.cpu(), model.sr)
    return output_path


def concatenate_audio(segment_files):
    waves = []

    for file in segment_files:
        wav, sr = ta.load(str(file))

        if sr != model.sr:
            wav = ta.functional.resample(wav, sr, model.sr)

        waves.append(wav)

    if not waves:
        raise ValueError("No audio segments to concatenate")

    final_wav = torch.cat(waves, dim=1)
    ta.save(str(OUTPUT_FILE), final_wav, model.sr)
    print(f"Final audio saved to {OUTPUT_FILE}")


# main function
def main():
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(FINAL_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]

    voices = get_voice()
    speaker_voice = voice_map(segments, voices)

    print(f"Speaker to voice mapping: {speaker_voice}")

    files = []

    for segment in segments:
        speaker = segment.get("speaker", "unknown")
        voice = speaker_voice.get(speaker)

        if voice is None:
            voice = voices[0]

        print(f"Generating audio for speaker: {speaker} using voice: {voice}")

        output_path = gen_seg(segment, voice)

        if output_path:
            files.append(output_path)

    concatenate_audio(files)
    print(f"Audio generated in {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()