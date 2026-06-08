# Streamlit user interface for the video anonymization pipeline.
# Upload a video, run automatic anonymization, review the transcript manually,
# and continue the pipeline to create the final anonymized video.
# Usage:
# source venvs/core/Scripts/activate
# streamlit run app.py

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st


# Find project root
ROOT_DIR = Path(__file__).resolve().parent

# Define project folders
INPUT_DIR = ROOT_DIR / "data" / "input"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

# Define important files
INPUT_VIDEO_FILE = INPUT_DIR / "video.mp4"
CLEANED_TRANSCRIPT_FILE = OUTPUT_DIR / "cleaned_transcription.json"
FINAL_VIDEO_FILE = OUTPUT_DIR / "final_video.mp4"

# Replacement labels shown in the manual review step
REPLACEMENTS = [
    "KEEP",
    "REMOVE",
    "NIMI",
    "HENKILÖTUNNUS",
    "SÄHKÖPOSTI",
    "PUHELINNUMERO",
    "POTILASTUNNUS",
    "SYNTYMÄAIKA",
    "OSOITE",
    "SIJAINTI",
    "ORGANISAATIO",
    "CUSTOM",
]

# Labels that should be collapsed if repeated in a row
SENSITIVE_LABELS = set(REPLACEMENTS) - {"KEEP", "REMOVE", "CUSTOM"}


def hide_streamlit_buttons():
    # Hide default Streamlit menu, header, footer and toolbar
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_script(title, script):
    # Run one pipeline script without showing terminal output to the user
    message = st.empty()
    message.info(f"Running: {title}")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / script),
        ],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        message.error(f"Step failed: {title}")

        # Show only the last error line
        if result.stderr:
            st.error(result.stderr.splitlines()[-1])

        st.stop()

    message.success(f"Completed: {title}")


def clean_folder_contents(folder):
    # Remove all files and subfolders inside a folder
    folder.mkdir(parents=True, exist_ok=True)

    for path in folder.iterdir():

        # Keep placeholder files if the project uses them
        if path.name == ".gitkeep":
            continue

        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()


def clean_old_outputs():
    # Remove old input and output files so the next run starts clean
    clean_folder_contents(INPUT_DIR)
    clean_folder_contents(OUTPUT_DIR)


def clean_files_on_new_session():
    # Clean previous files when a new Streamlit session starts
    if "startup_cleanup_done" in st.session_state:
        return

    clean_old_outputs()

    st.session_state.startup_cleanup_done = True
    st.session_state.review_ready = False
    st.session_state.updated_segments = {}

    if "review_data" in st.session_state:
        del st.session_state.review_data


def save_uploaded_video(uploaded_file):
    # Save uploaded video as data/input/video.mp4
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_VIDEO_FILE, "wb") as f:
        f.write(uploaded_file.getbuffer())


def load_cleaned_transcript():
    # Load cleaned_transcription.json
    with open(CLEANED_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cleaned_transcript(data):
    # Save cleaned_transcription.json
    CLEANED_TRANSCRIPT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(CLEANED_TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def clean_text(words):
    # Build readable text from word list
    text_words = []

    for word in words:
        word_text = word.get("word", "").strip()

        if word_text:
            text_words.append(word_text)

    text = " ".join(text_words)

    # Fix spaces before punctuation
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")

    return text.strip()


def make_editor_rows(segment):
    # Convert segment words into table rows
    rows = []

    for word in segment.get("words", []):
        word_text = word.get("word", "")

        # Keep existing anonymization labels visible in the dropdown
        if word_text in SENSITIVE_LABELS:
            action = word_text
        elif word_text == "":
            action = "REMOVE"
        else:
            action = "KEEP"

        rows.append({
            "word": word_text,
            "action": action,
            "custom_replacement": "",
        })

    return rows


def apply_editor_rows(segment, edited_rows):
    # Apply manual word edits and replacements back to the segment
    words = segment.get("words", [])

    last_sensitive_label = None

    for index, row in enumerate(edited_rows):
        if index >= len(words):
            continue

        edited_word = str(row["word"]).strip()
        action = row["action"]
        custom_replacement = str(row.get("custom_replacement", "")).strip()

        if action == "KEEP":
            new_word = edited_word

        elif action == "REMOVE":
            new_word = ""

        elif action == "CUSTOM":
            new_word = custom_replacement

        else:
            new_word = action

        # Collapse repeated standard labels, for example NIMI NIMI -> NIMI
        if new_word in SENSITIVE_LABELS:
            if new_word == last_sensitive_label:
                new_word = ""
            else:
                last_sensitive_label = new_word

        elif new_word:
            last_sensitive_label = None

        words[index]["word"] = new_word

    # Rebuild segment text from the reviewed words
    segment["words"] = words
    segment["text"] = clean_text(words)


def run_first_part():
    # Run pipeline until manual review is ready
    run_script("Extract audio from video", "src/audio.py")
    run_script("Transcribe speech", "run/run_transcript.py")
    run_script("Run automatic anonymization", "run/run_anonymize.py")

    st.session_state.review_data = load_cleaned_transcript()
    st.session_state.review_ready = True


def run_second_part():
    # Run the rest of the pipeline after manual review
    run_script("Detect speakers", "run/run_speaker.py")
    run_script("Merge transcript and speakers", "src/merged.py")
    run_script("Generate anonymized audio", "run/run_tts.py")
    run_script("Anonymize video", "run/run_video.py")
    run_script("Merge anonymized audio and video", "src/final_merge.py")

    st.success("Anonymization complete.")


def show_updated_segments():
    # Show all transcript segments updated by the user
    updated_segments = st.session_state.get("updated_segments", {})

    if not updated_segments:
        return

    st.write("### Updated text segments")

    for index, text in updated_segments.items():
        st.write(f"**Segment {index + 1}**")
        st.info(text)


def show_review_ui():
    # Show manual transcript review interface
    st.write("## 2. Review anonymized transcript")

    data = st.session_state.review_data

    if not data:
        st.warning("No transcript data found.")
        return

    st.info(
        "Review the transcript one segment at a time. "
        "Use the selection box below to move between transcript segments. "
        "You can fix spelling mistakes in the Word column, choose an anonymization label in the Action column, "
        "or choose CUSTOM and write your own replacement in the Custom replacement column."
    )

    segment_options = []

    for index, segment in enumerate(data):
        text = segment.get("text", "")
        preview = text[:140] + "..." if len(text) > 140 else text
        segment_options.append(f"Segment {index + 1}: {preview}")

    selected = st.selectbox(
        "Select a transcript segment to review",
        options=list(range(len(segment_options))),
        format_func=lambda index: segment_options[index],
        help="Open this menu to choose another transcript segment.",
    )

    segment = data[selected]

    st.write("### Current text segment")
    st.info(segment.get("text", ""))

    rows = make_editor_rows(segment)
    df = pd.DataFrame(rows)

    st.write("### Edit words")
    st.write(
        "Edit spelling mistakes directly in the Word column. "
        "Use Action to keep, remove or replace a word. "
        "If you choose CUSTOM, write the replacement text in the Custom replacement column."
    )

    edited_df = st.data_editor(
        df,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "word": st.column_config.TextColumn("Word"),
            "action": st.column_config.SelectboxColumn(
                "Action",
                options=REPLACEMENTS,
            ),
            "custom_replacement": st.column_config.TextColumn("Custom replacement"),
        },
        key=f"editor_{selected}",
    )

    if st.button("Save changes"):
        edited_rows = edited_df.to_dict("records")
        apply_editor_rows(segment, edited_rows)

        st.session_state.review_data[selected] = segment
        save_cleaned_transcript(st.session_state.review_data)

        st.session_state.updated_segments[selected] = segment.get("text", "")

        st.success("Changes saved.")
        show_updated_segments()

    else:
        show_updated_segments()


def show_video_area():
    # Show final video in Streamlit and download button
    if not FINAL_VIDEO_FILE.exists():
        return

    st.write("## 4. Final anonymized video")

    with open(FINAL_VIDEO_FILE, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes, format="video/mp4")

    st.download_button(
        label="Download video",
        data=video_bytes,
        file_name=FINAL_VIDEO_FILE.name,
        mime="video/mp4",
    )


def reset_app_state():
    # Reset Streamlit state for a new video
    st.session_state.review_ready = False
    st.session_state.updated_segments = {}

    if "review_data" in st.session_state:
        del st.session_state.review_data


def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Video anonymization",
        layout="wide",
    )

    # Hide default Streamlit UI controls
    hide_streamlit_buttons()

    # Clean old files when the app is opened in a new session
    clean_files_on_new_session()

    st.title("Video anonymization")

    st.write(
        "Upload a video, run automatic anonymization, manually review the transcript "
        "and create the final anonymized video."
    )

    if "review_ready" not in st.session_state:
        st.session_state.review_ready = False

    if "updated_segments" not in st.session_state:
        st.session_state.updated_segments = {}

    st.write("## 1. Upload video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4"],
    )

    if uploaded_file:
        st.write(f"Selected file: `{uploaded_file.name}`")

        if st.button("Anonymize video"):
            reset_app_state()
            clean_old_outputs()
            save_uploaded_video(uploaded_file)

            with st.spinner("Running anonymization..."):
                run_first_part()

            st.success("Transcript is ready for manual review.")

    st.write("---")

    if st.session_state.review_ready:
        show_review_ui()

        st.write("---")
        st.write("## 3. Continue anonymization")

        if st.button("Continue anonymization"):
            save_cleaned_transcript(st.session_state.review_data)

            with st.spinner("Running the rest of the anonymization..."):
                run_second_part()

    show_video_area()


if __name__ == "__main__":
    main()