# > diarization.json
# > transcript.json
# = final_transcript.json (start, end, text, speaker_id)
# Merges WhisperX transcription and Pyannote speaker diarization
# Saves the final transcript to data/output/final_transcript.json
# Usage:
# source venvs/core/Scripts/activate
# python src/merged.py

from pathlib import Path
import json

import os
import json

# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define input transcript path
TRANSCRIPT_FILE = ROOT_DIR / "data" / "output" / "cleaned_transcription.json"

# Define input diarization path
DIARIZATION_FILE = ROOT_DIR / "data" / "output" / "diarization.json"

# Define output final transcript path
OUTPUT_FILE = ROOT_DIR / "data" / "output" / "final_transcript.json"

# function for cleaning text
def clean_text(words):
    text = " ".join(words)
    return (
        text.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .strip()
    )

def overlap(start, end, diar_start, diar_end):
    return max(0.0, min(end, diar_end) - max(start, diar_start))

def load_diarization(diarization):
    if isinstance(diarization, list):
        return diarization

    if isinstance(diarization, dict) and "segments" in diarization:
        return diarization["segments"]

    raise ValueError("could not read diarization.json")



def speaker_word(start, end, diarization):
    best_speaker = "unknown"
    best_overlap = 0.0

    for diar in diarization:
        diar_start = diar.get("start")
        diar_end = diar.get("end")
        speaker = diar.get("speaker", "unknown")

        if diar_start is None or diar_end is None:
            continue

        amount = overlap(
            float(start),
            float(end),
            float(diar_start),
            float(diar_end)
        )

        if amount > best_overlap:
            best_overlap = amount
            best_speaker = speaker

    return best_speaker

# save current segment into segment-list
def flush_current(new_segments, current):
    if current is None:
        return None

    if not current["words"]:
        return None

    new_segments.append({
        "segment_id": len(new_segments),
        "start": round(current["start"], 3),
        "end": round(current["end"], 3),
        "speaker": current["speaker"],
        "text": clean_text(current["words"])
    })
    return None

def merge_same_speaker(segments):
    if not segments:
        return []

    merged = []

    for segment in segments:
        if not merged:
            merged.append(segment.copy())
            continue

        previous = merged[-1]

        if previous["speaker"] == segment["speaker"]:
            previous["end"] = segment["end"]
            previous["text"] = clean_text([
                previous["text"],
                segment["text"]
 ])
        else:
            merged.append(segment.copy())

    for index, segment in enumerate(merged):
        segment["segment_id"] = index

    return merged

# Merge transcripts words to diarized word-stamps
def merge():
    with open(TRANSCRIPT_FILE, encoding="utf-8") as f:
        transcript = json.load(f)

    with open(DIARIZATION_FILE, encoding="utf-8") as f:
        diarization = json.load(f)

    diar = load_diarization(diarization)

    new_segments = []
    current = None

    for seg in transcript:
        words = seg.get("words", [])

        if not words:
            current = flush_current(new_segments, current)

            start = float(seg["segment_start"])
            end = float(seg["segment_end"])
            speaker = speaker_word(start, end, diar)

            new_segments.append({
                "segment_id": len(new_segments),
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": seg.get("text", " ").strip()
            })
            continue

        # if word-stamp found, go thrue every stamp and clean
        for word in words:
            word_text = (word.get("word") or " ").strip()
            start = word.get("start")
            end = word.get("end")

            if not word_text:
                continue

            # if current segment already exists, add word -> else add to previous segment
            if start is None or end is None:
                if current is not None:
                    current["words"].append(word_text)
                elif new_segments:
                    new_segments[-1]["text"] = clean_text([
                        new_segments[-1]["text"],
                        word_text
                    ])
                continue

            start = float(start)
            end = float(end)
            speaker = speaker_word(start, end, diar)

            if current is None:
                current = {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "words": [word_text],
                }
                continue

            same_speaker = current["speaker"] == speaker
            small_gap = start - current["end"] < 0.8

            if same_speaker and small_gap:
                current["end"] = end
                current["words"].append(word_text)
            else:
                current = flush_current(new_segments, current)

                current = {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "words": [word_text]
                }

        current = flush_current(new_segments, current)

    new_segments = merge_same_speaker(new_segments)

    output = {
        "language": "fi",
        "segments": new_segments,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Final transcript is in: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge()