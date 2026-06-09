# Gets final_transcript.json (start, end, text, speaker_id)
# Gets voices/*.wav files and gives them to the speaker id's
# Loads Chatterbox TTS model
# Generates speech audio
# Saves clean_audio.wav
# Usage:
# source venvs/tts/Scripts/activate
# python src/tts.py

import json
import re
from pathlib import Path

import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

BASE_DIR = Path(__file__).resolve().parent.parent

FINAL_TRANSCRIPT_FILE = BASE_DIR / "data" / "output" / "final_transcript.json"
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_FILE = BASE_DIR / "data" / "output" / "clean_audio.wav"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

def get_voices():
    voices = sorted(VOICES_DIR.glob("*.wav"))

    if not voices:
        raise FileNotFoundError(f"No voice files found in {VOICES_DIR}")

    return voices

def normalize_tts(text):
    replace = {
        r"\biv\b": "ii-vee",
        r"\bIV\b": "ii-vee",
        r"\bEKG\b": "ee-koo-gee",
        r"\bMRI\b": "äm-ärr-ii",
        r"\bCT\b": "see-tee",
    }

    for pattern, replacement in replace.items():
        text = re.sub(pattern, replacement, text)

    return text

# Get the speaker ID for a word based on the diarization segments
def create_voice_map(segments, voices):
    speakers = sorted(set(seg.get("speaker", "unknown") for seg in segments))

    speaker_voice = {}

    for index, speaker in enumerate(speakers):
        speaker_voice[speaker] = voices[index % len(voices)]

    return speaker_voice

# Generate audio for a single segment using the TTS model and the given voice
def generate_segment_audio(text, voice_path):
    wav = model.generate(
        text,
        audio_prompt_path=str(voice_path),
        language_id="fi"
 )

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    return wav

# Pad or trim the generated audio to match the target duration
def audio_duration(wav, target_samples):
    current_samples = wav.shape[1]

# If the generated audio is longer than the target duration, trim it
    if current_samples >= target_samples:
        return wav

    padding_samples = target_samples - current_samples
    padding = torch.zeros(
        wav.shape[0],
        padding_samples,
        dtype=wav.dtype,
        device=wav.device
    )
    return torch.cat([wav, padding], dim=1)

# Merge consecutive segments with the same speaker into one segment
def build_full_audio(segments, speaker_voice):
    sample_rate = model.sr

    if not segments:
        raise ValueError("No segments found in final_transcript.json")

    total_duration = max(float(seg["end"]) for seg in segments)
    total_samples = int(total_duration * sample_rate) + sample_rate * 30

    final_wav = torch.zeros(1, total_samples)

    previous_speaker = None
    current_sample = 0

    same_speaker_pause = 0.01
    different_speaker_pause = 0.25

    for segment in segments:
        text = segment.get("text", "").strip()

        if not text:
            continue

        text = normalize_tts(text)

        speaker = segment.get("speaker", "unknown")
        voice_path = speaker_voice.get(speaker)

        if voice_path is None:
            raise ValueError(f"No voice found for speaker: {speaker}")

        original_start = float(segment["start"])
        original_end = float(segment["end"])
        original_start_sample = int(original_start * sample_rate)

        if previous_speaker is None:
            pause_seconds = 0
        elif speaker == previous_speaker:
            pause_seconds = same_speaker_pause
        else:
            pause_seconds = different_speaker_pause

        pause_samples = int(pause_seconds * sample_rate)

        actual_start_sample = max(
            original_start_sample,
            current_sample + pause_samples
 )

        print(
            f"Generating: {speaker} | "
            f"{original_start:.2f}-{original_end:.2f} | "
            f"pause {pause_seconds:.2f}s | "
            f"{voice_path.name}"
 )

        wav = generate_segment_audio(text, voice_path)

        if wav.device != final_wav.device:
            wav = wav.cpu()

        insert_end = actual_start_sample + wav.shape[1]

        if insert_end > final_wav.shape[1]:
            extra_samples = insert_end - final_wav.shape[1]

            padding = torch.zeros(
                final_wav.shape[0],
                extra_samples,
                dtype=final_wav.dtype,
                device=final_wav.device
 )

            final_wav = torch.cat([final_wav, padding], dim=1)

        final_wav[:, actual_start_sample:insert_end] += wav

        current_sample = insert_end
        previous_speaker = speaker

    return final_wav
# Save the final audio to a file
def save_audio(output_file, wav, sample_rate):
    wav = wav.detach().cpu()

    if wav.dim() == 2:
        wav = wav.squeeze(0)

    audio = wav.numpy()

    sf.write(str(output_file), audio, sample_rate)



def main():
    with open(FINAL_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]

    voices = get_voices()
    speaker_voice = create_voice_map(segments, voices)

    print("Speaker to voice mapping:")
    for speaker, voice in speaker_voice.items():
        print(f"{speaker} -> {voice}")

    final_wav = build_full_audio(segments, speaker_voice)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    save_audio(OUTPUT_FILE, final_wav, model.sr)

    print(f"Clean audio saved to: {OUTPUT_FILE}")



if __name__ == "__main__":
    main()