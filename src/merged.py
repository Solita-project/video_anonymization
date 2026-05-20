# This script:
# 1. Loads transcript.json
# 2. Loads diarization.json
# 3. Matches words to speakers
# 4. Groups words into speaker segments
# 5. Saves final_transcript.json


from pathlib import Path
import json


# Resolve project root automatically
ROOT_DIR = Path(__file__).resolve().parent.parent


# Input/output file locations
TRANSCRIPT_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "transcription.json"
)

DIARIZATION_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "diarization.json"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "data"
    / "output"
    / "final_transcript.json"
)


def clean_text(words):
    """
    Join words and remove spacing issues around punctuation.
    """

    text = " ".join(words)

    return (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .strip()
    )


def overlap(
    start,
    end,
    diar_start,
    diar_end
):
    """
    Calculate time overlap between
    transcript and speaker segment.
    """

    return max(
        0.0,
        min(end, diar_end)
        - max(start, diar_start)
    )


def load_diarization(diarization):
    """
    Support multiple possible
    diarization JSON formats.
    """

    if isinstance(
        diarization,
        list
    ):
        return diarization

    if (
        isinstance(
            diarization,
            dict
        )
        and "segments"
        in diarization
    ):
        return diarization[
            "segments"
        ]

    raise ValueError(
        "Could not read diarization.json"
    )


def speaker_word(
    start,
    end,
    diarization
):
    """
    Determine which speaker
    overlaps most with a word.
    """

    best_speaker = "unknown"

    best_overlap = 0.0

    for diar in diarization:

        diar_start = diar.get(
            "start"
        )

        diar_end = diar.get(
            "end"
        )

        speaker = diar.get(
            "speaker",
            "unknown"
        )

        if (
            diar_start is None
            or diar_end is None
        ):
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


def flush_current(
    new_segments,
    current
):
    """
    Save currently accumulated
    speaker segment.
    """

    if current is None:
        return None

    if not current["words"]:
        return None

    new_segments.append({

        "segment_id":
        len(new_segments),

        "start":
        round(
            current["start"],
            3
        ),

        "end":
        round(
            current["end"],
            3
        ),

        "speaker":
        current["speaker"],

        "text":
        clean_text(
            current["words"]
        )

    })

    return None


def merge():

    # Verify input files exist
    if not TRANSCRIPT_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{TRANSCRIPT_FILE}"
        )

    if not DIARIZATION_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{DIARIZATION_FILE}"
        )

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load transcript
    with open(
        TRANSCRIPT_FILE,
        encoding="utf-8"
    ) as f:

        transcript = json.load(f)

    # Load diarization
    with open(
        DIARIZATION_FILE,
        encoding="utf-8"
    ) as f:

        diarization = json.load(f)

    diar = load_diarization(
        diarization
    )

    new_segments = []

    current = None

    # Process each transcript segment
    for seg in transcript:

        words = seg.get(
            "words",
            []
        )

        # Handle empty segments
        if not words:

            current = flush_current(
                new_segments,
                current
            )

            start = float(
                seg["segment_start"]
            )

            end = float(
                seg["segment_end"]
            )

            speaker = speaker_word(
                start,
                end,
                diar
            )

            new_segments.append({

                "segment_id":
                len(new_segments),

                "start":
                start,

                "end":
                end,

                "speaker":
                speaker,

                "text":
                seg.get(
                    "text",
                    ""
                ).strip()

            })

            continue

        # Process words one-by-one
        for word in words:

            word_text = (
                word.get(
                    "word"
                )
                or ""
            ).strip()

            start = word.get(
                "start"
            )

            end = word.get(
                "end"
            )

            if not word_text:
                continue

            if (
                start is None
                or end is None
            ):

                if current:

                    current[
                        "words"
                    ].append(
                        word_text
                    )

                continue

            start = float(start)

            end = float(end)

            speaker = speaker_word(
                start,
                end,
                diar
            )

            if current is None:

                current = {

                    "speaker":
                    speaker,

                    "start":
                    start,

                    "end":
                    end,

                    "words":
                    [word_text]

                }

                continue

            same_speaker = (
                current[
                    "speaker"
                ]
                == speaker
            )

            small_gap = (
                start
                - current["end"]
                < 0.8
            )

            if (
                same_speaker
                and small_gap
            ):

                current[
                    "end"
                ] = end

                current[
                    "words"
                ].append(
                    word_text
                )

            else:

                current = flush_current(
                    new_segments,
                    current
                )

                current = {

                    "speaker":
                    speaker,

                    "start":
                    start,

                    "end":
                    end,

                    "words":
                    [word_text]

                }

    current = flush_current(
        new_segments,
        current
    )

    output = {

        "segments":
        new_segments

    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(
        f"\nFinal transcript saved:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    merge()