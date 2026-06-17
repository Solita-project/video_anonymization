# Streamlit user interface for the video anonymization pipeline.
# Upload a video, review and improve video anonymization, review the transcript manually,
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
import time

from PIL import Image
import pandas as pd
import streamlit as st

# Compatibility fix for streamlit-drawable-canvas with newer Streamlit versions.
try:
    from streamlit.elements import image as st_image
    from streamlit.elements.lib.image_utils import image_to_url

    if not hasattr(st_image, "image_to_url"):
        st_image.image_to_url = image_to_url

except Exception:
    pass

from streamlit_drawable_canvas import st_canvas


# Find project root
ROOT_DIR = Path(__file__).resolve().parent

# Define project folders
INPUT_DIR = ROOT_DIR / "data" / "input"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

# Define important files
INPUT_VIDEO_FILE = INPUT_DIR / "video.mp4"
BLURRED_VIDEO_FILE = OUTPUT_DIR / "video_blurred.mp4"
BLURRED_PREVIEW_FILE = OUTPUT_DIR / "video_blurred_preview.mp4"
CLEANED_TRANSCRIPT_FILE = OUTPUT_DIR / "cleaned_transcription.json"
FINAL_VIDEO_FILE = OUTPUT_DIR / "final_video.mp4"

# Define transcript preparation files
TRANSCRIPT_SCRIPT = ROOT_DIR / "run" / "run_prepare_transcript.py"
TRANSCRIPT_LOG_FILE = OUTPUT_DIR / "transcript_background.log"

# Define manual video blur files
MANUAL_BLUR_SCRIPT = ROOT_DIR / "run" / "run_manual_video_blur.py"
MANUAL_BLUR_FILE = OUTPUT_DIR / "manual_video_blurs.json"
MANUAL_BLUR_FRAME_FILE = OUTPUT_DIR / "manual_blur_frame.jpg"

# Define local FFmpeg path
LOCAL_FFMPEG = ROOT_DIR / "tools" / "ffmpeg.exe"

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
]

# Labels that should be collapsed if repeated in a row
SENSITIVE_LABELS = set(REPLACEMENTS) - {"KEEP", "REMOVE"}


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


def run_script(title, script, show_status=True):
    # Run pipeline script.
    # Script output is shown in the terminal where Streamlit is running.
    message = None

    if show_status:
        message = st.empty()
        message.info(f"Running: {title}")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / script),
        ],
        cwd=ROOT_DIR,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        if message:
            message.error(f"Step failed: {title}")
        else:
            st.error(f"Step failed: {title}")

        st.error("Check the terminal for details.")
        st.stop()

    if show_status and message:
        message.success(f"Completed: {title}")


def clean_folder_contents(folder):
    # Remove all files and subfolders inside a folder
    folder.mkdir(parents=True, exist_ok=True)

    for path in folder.iterdir():
        if path.name == ".gitkeep":
            continue

        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()


def clean_old_outputs():
    # Remove old input and output files
    clean_folder_contents(INPUT_DIR)
    clean_folder_contents(OUTPUT_DIR)


def stop_transcript_process():
    # Stop transcript preparation if it is still running
    process = st.session_state.get("transcript_process")

    if process is not None and process.poll() is None:
        process.terminate()


def reset_app_state():
    # Reset app state for a new video
    stop_transcript_process()

    st.session_state.page = "upload"
    st.session_state.video_approved = False
    st.session_state.show_manual_blur = False
    st.session_state.updated_segments = {}

    # Bump widget keys to avoid old widget state with a new video
    st.session_state.canvas_key_version = st.session_state.get("canvas_key_version", 0) + 1
    st.session_state.review_editor_version = st.session_state.get("review_editor_version", 0) + 1

    if "review_data" in st.session_state:
        del st.session_state.review_data

    if "transcript_process" in st.session_state:
        del st.session_state.transcript_process

    if "selected_segment" in st.session_state:
        del st.session_state.selected_segment


def clean_files_on_new_session():
    # Clean previous files when a new Streamlit session starts
    if "startup_cleanup_done" in st.session_state:
        return

    clean_old_outputs()
    reset_app_state()

    st.session_state.startup_cleanup_done = True


def resolve_ffmpeg():
    # Use local ffmpeg.exe if it exists
    if LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)

    # Otherwise use ffmpeg from system PATH
    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError("ffmpeg not found")


def create_browser_preview(input_file, output_file):
    # Convert video to browser-friendly MP4
    ffmpeg = resolve_ffmpeg()

    if not input_file.exists():
        raise FileNotFoundError(f"Video not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i", str(input_file),
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_file),
    ]

    subprocess.run(command, check=True)


def run_video_anonymization():
    # Run automatic video anonymization and create a browser-friendly preview
    run_script("Anonymize video", "run/run_video.py", show_status=False)
    create_browser_preview(BLURRED_VIDEO_FILE, BLURRED_PREVIEW_FILE)


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


def start_transcript_process():
    # Start audio extraction, transcription and automatic transcript anonymization in the background
    process = st.session_state.get("transcript_process")

    if process is not None and process.poll() is None:
        return

    if CLEANED_TRANSCRIPT_FILE.exists():
        return

    if not TRANSCRIPT_SCRIPT.exists():
        st.error(f"Missing script: {TRANSCRIPT_SCRIPT}")
        st.stop()

    # Do not redirect stdout or stderr (this lets the background script print to the terminal)
    process = subprocess.Popen(
        [
            sys.executable,
            str(TRANSCRIPT_SCRIPT),
        ],
        cwd=ROOT_DIR,
        env=os.environ.copy(),
    )

    st.session_state.transcript_process = process


def get_transcript_status():
    # Check transcript preparation status
    if CLEANED_TRANSCRIPT_FILE.exists():
        return "ready"

    process = st.session_state.get("transcript_process")

    if process is None:
        return "not_started"

    if process.poll() is None:
        return "running"

    return "failed"


def read_transcript_log():
    # Read transcript preparation log
    if not TRANSCRIPT_LOG_FILE.exists():
        return ""

    with open(TRANSCRIPT_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_review_data_if_ready():
    # Load transcript review data only once
    if "review_data" in st.session_state:
        return True

    if not CLEANED_TRANSCRIPT_FILE.exists():
        return False

    st.session_state.review_data = load_cleaned_transcript()
    return True


def run_manual_blur_command(arguments, title):
    # Run manual video blur wrapper
    if not MANUAL_BLUR_SCRIPT.exists():
        st.error(f"Missing script: {MANUAL_BLUR_SCRIPT}")
        st.stop()

    message = st.empty()
    message.info(f"Running: {title}")

    result = subprocess.run(
        [
            sys.executable,
            str(MANUAL_BLUR_SCRIPT),
        ] + arguments,
        cwd=ROOT_DIR,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        message.error(f"Step failed: {title}")
        st.error("Check the terminal for details.")
        st.stop()

    message.success(f"Completed: {title}")


def load_manual_blurs():
    # Load saved manual blur areas
    if not MANUAL_BLUR_FILE.exists():
        return {"annotations": []}

    with open(MANUAL_BLUR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        data["annotations"] = []

    return data


def save_manual_blurs(data):
    # Save manual blur areas to JSON
    MANUAL_BLUR_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MANUAL_BLUR_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def resize_frame_for_canvas(image, max_width=900):
    # Resize large video frame for drawing in Streamlit
    if image.width <= max_width:
        return image

    scale = max_width / image.width
    new_width = max_width
    new_height = int(image.height * scale)

    if hasattr(Image, "Resampling"):
        resampling_filter = Image.Resampling.LANCZOS
    else:
        resampling_filter = Image.LANCZOS

    return image.resize(
        (new_width, new_height),
        resampling_filter,
    )


def get_latest_canvas_rectangle(canvas_result):
    # Get the latest rectangle drawn by the user
    if canvas_result.json_data is None:
        return None

    objects = canvas_result.json_data.get("objects", [])

    if not objects:
        return None

    rectangles = []

    for item in objects:
        if item.get("type") == "rect":
            rectangles.append(item)

    if not rectangles:
        return None

    rectangle = rectangles[-1]

    left = float(rectangle.get("left", 0))
    top = float(rectangle.get("top", 0))
    width = float(rectangle.get("width", 0)) * float(rectangle.get("scaleX", 1))
    height = float(rectangle.get("height", 0)) * float(rectangle.get("scaleY", 1))

    if width <= 0 or height <= 0:
        return None

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def show_saved_blur_areas():
    # Show saved manual blur areas
    data = load_manual_blurs()
    annotations = data.get("annotations", [])

    if not annotations:
        return

    st.write("### Saved blur areas")

    for index, annotation in enumerate(annotations):
        start_time = float(annotation["start_time"])
        end_time = float(annotation["end_time"])
        x = annotation["x"]
        y = annotation["y"]
        width = annotation["width"]
        height = annotation["height"]

        st.write(
            f"Area {index + 1}: "
            f"{start_time:.1f}s–{end_time:.1f}s, "
            f"x={x}, y={y}, width={width}, height={height}"
        )

    if st.button("Clear saved blur areas"):
        save_manual_blurs({"annotations": []})
        st.rerun()


def show_transcript_status(status):
    # Show short transcript preparation status below the video
    if status == "running":
        st.info("Preparing transcript...")

    elif status == "ready":
        st.success("Transcript is ready.")

    elif status == "failed":
        st.error("Transcript preparation failed.")

        log_text = read_transcript_log()

        if log_text:
            with st.expander("Show technical details"):
                st.code(log_text)


def show_upload_section():
    # Show video upload section
    st.write("## Upload video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4"],
    )

    if not uploaded_file:
        return

    st.write(f"Selected file: `{uploaded_file.name}`")

    if st.button("Anonymize video"):
        reset_app_state()
        clean_old_outputs()
        save_uploaded_video(uploaded_file)

        with st.spinner("Running video anonymization..."):
            run_video_anonymization()

        start_transcript_process()

        st.session_state.page = "processing"
        st.rerun()


def show_video_section(status):
    # Show anonymized video and video review buttons
    st.write("## Review video anonymization")

    if BLURRED_PREVIEW_FILE.exists():
        with open(BLURRED_PREVIEW_FILE, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes, format="video/mp4")
    else:
        st.warning("Anonymized video was not found.")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Video anonymization looks good"):
            st.session_state.video_approved = True
            st.session_state.show_manual_blur = False
            st.rerun()

    with col2:
        if st.button("I want to improve video anonymization"):
            st.session_state.video_approved = False
            st.session_state.show_manual_blur = True
            st.rerun()

    show_transcript_status(status)


def show_manual_blur_section():
    # Show manual blur tools if the user wants to improve video anonymization
    if not st.session_state.get("show_manual_blur"):
        return

    st.write("---")
    st.write("## Improve video anonymization")

    frame_time = st.number_input(
        "Frame time in seconds",
        min_value=0.0,
        step=0.1,
        format="%.1f",
        key="manual_frame_time",
    )

    if st.button("Show frame"):
        run_manual_blur_command(
            ["extract", "--time", str(frame_time)],
            "Extract frame",
        )

        st.session_state.canvas_key_version += 1
        st.rerun()

    if MANUAL_BLUR_FRAME_FILE.exists():
        original_image = Image.open(MANUAL_BLUR_FRAME_FILE).convert("RGB")
        display_image = resize_frame_for_canvas(original_image)

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.20)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=display_image,
            update_streamlit=True,
            height=display_image.height,
            width=display_image.width,
            drawing_mode="rect",
            key=f"manual_blur_canvas_{st.session_state.canvas_key_version}",
        )

        start_time = st.number_input(
            "Start time in seconds",
            min_value=0.0,
            value=max(0.0, frame_time - 1.0),
            step=0.1,
            format="%.1f",
            key="manual_blur_start_time",
        )

        end_time = st.number_input(
            "End time in seconds",
            min_value=0.0,
            value=frame_time + 1.0,
            step=0.1,
            format="%.1f",
            key="manual_blur_end_time",
        )

        blur_strength = st.slider(
            "Blur strength",
            min_value=15,
            max_value=99,
            value=45,
            step=2,
        )

        if st.button("Save blur area"):
            if end_time <= start_time:
                st.error("End time must be greater than start time.")
                return

            rectangle = get_latest_canvas_rectangle(canvas_result)

            if rectangle is None:
                st.error("Draw one rectangle on the frame first.")
                return

            scale_x = original_image.width / display_image.width
            scale_y = original_image.height / display_image.height

            annotation = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "x": int(round(rectangle["left"] * scale_x)),
                "y": int(round(rectangle["top"] * scale_y)),
                "width": int(round(rectangle["width"] * scale_x)),
                "height": int(round(rectangle["height"] * scale_y)),
                "blur_strength": int(blur_strength),
            }

            data = load_manual_blurs()
            data["annotations"].append(annotation)
            save_manual_blurs(data)

            st.success("Blur area saved.")
            st.rerun()

    show_saved_blur_areas()

    data = load_manual_blurs()
    annotations = data.get("annotations", [])

    if annotations:
        if st.button("Run additional blurs"):
            with st.spinner("Running additional blurs..."):
                run_manual_blur_command(
                    ["apply"],
                    "Run additional blurs",
                )

                create_browser_preview(
                    BLURRED_VIDEO_FILE,
                    BLURRED_PREVIEW_FILE,
                )

            # Clear annotations after applying them to avoid applying the same areas twice
            save_manual_blurs({"annotations": []})

            st.session_state.show_manual_blur = False
            st.session_state.video_approved = False
            st.session_state.canvas_key_version += 1

            st.success("Additional blurs were applied.")
            st.rerun()


def handle_video_decision(status):
    # Continue to transcript review only after video approval
    if status == "failed":
        return

    if not st.session_state.get("video_approved"):
        return

    if status == "ready":
        load_review_data_if_ready()
        return

    st.info("Waiting for transcript preparation to finish...")
    time.sleep(5)
    st.rerun()


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

        if word_text in SENSITIVE_LABELS:
            action = word_text
        elif word_text == "":
            action = "REMOVE"
        else:
            action = "KEEP"

        rows.append({
            "word": word_text,
            "action": action,
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

        if action == "KEEP":
            new_word = edited_word
        elif action == "REMOVE":
            new_word = ""
        else:
            new_word = action

        # Collapse repeated labels, for example NIMI NIMI -> NIMI
        if new_word in SENSITIVE_LABELS:
            if new_word == last_sensitive_label:
                new_word = ""
            else:
                last_sensitive_label = new_word
        elif new_word:
            last_sensitive_label = None

        words[index]["word"] = new_word

    segment["words"] = words
    segment["text"] = clean_text(words)


def show_updated_segments():
    # Show transcript segments updated by the user
    updated_segments = st.session_state.get("updated_segments", {})

    if not updated_segments:
        return

    st.write("### Updated text segments")

    for index, text in updated_segments.items():
        st.write(f"**Segment {index + 1}**")
        st.info(text)


def show_transcript_review_section():
    # Show manual transcript review section
    if not st.session_state.get("video_approved"):
        return

    if not load_review_data_if_ready():
        return

    st.write("---")
    st.write("## Review anonymized transcript")

    data = st.session_state.review_data

    if not data:
        st.warning("No transcript data found.")
        return

    if "selected_segment" not in st.session_state:
        st.session_state.selected_segment = 0

    segment_options = []

    for index, segment in enumerate(data):
        text = segment.get("text", "")
        preview = text[:140] + "..." if len(text) > 140 else text
        segment_options.append(f"Segment {index + 1}: {preview}")

    selected = st.selectbox(
        "Select segment to edit",
        options=list(range(len(segment_options))),
        format_func=lambda index: segment_options[index],
        key="selected_segment",
    )

    segment = data[selected]

    st.write("### Current selected segment")
    st.info(segment.get("text", ""))

    rows = make_editor_rows(segment)
    df = pd.DataFrame(rows)

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
        },
        key=f"editor_{st.session_state.review_editor_version}_{selected}",
    )

    if st.button("Save changes"):
        edited_rows = edited_df.to_dict("records")
        apply_editor_rows(segment, edited_rows)

        st.session_state.review_data[selected] = segment
        save_cleaned_transcript(st.session_state.review_data)

        st.session_state.updated_segments[selected] = segment.get("text", "")

        st.success("Changes saved.")

    show_updated_segments()

    st.write("---")
    st.write("## Continue anonymization")

    if st.button("Continue anonymization"):
        save_cleaned_transcript(st.session_state.review_data)

        with st.spinner("Running the rest of the anonymization..."):
            run_final_pipeline()

        st.session_state.page = "done"
        st.success("Anonymization complete.")
        show_final_video()


def run_final_pipeline():
    # Run the rest of the pipeline after manual review
    run_script("Detect speakers", "run/run_speaker.py")
    run_script("Merge transcript and speakers", "src/merged.py")
    run_script("Generate anonymized audio", "run/run_tts.py")
    run_script("Merge anonymized audio and video", "src/final_merge.py")


def show_final_video():
    # Show final video and download button
    if not FINAL_VIDEO_FILE.exists():
        return

    st.write("---")
    st.write("## Final anonymized video")

    with open(FINAL_VIDEO_FILE, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes, format="video/mp4")

    st.download_button(
        label="Download video",
        data=video_bytes,
        file_name=FINAL_VIDEO_FILE.name,
        mime="video/mp4",
    )


def show_processing_page():
    # Show video review first, then manual blur tools and transcript review when available
    status = get_transcript_status()

    show_video_section(status)
    show_manual_blur_section()
    handle_video_decision(status)
    show_transcript_review_section()


def initialize_session_state():
    # Initialize Streamlit session variables
    if "page" not in st.session_state:
        st.session_state.page = "upload"

    if "video_approved" not in st.session_state:
        st.session_state.video_approved = False

    if "show_manual_blur" not in st.session_state:
        st.session_state.show_manual_blur = False

    if "updated_segments" not in st.session_state:
        st.session_state.updated_segments = {}

    if "canvas_key_version" not in st.session_state:
        st.session_state.canvas_key_version = 0

    if "review_editor_version" not in st.session_state:
        st.session_state.review_editor_version = 0


def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Video anonymization",
        layout="wide",
    )

    hide_streamlit_buttons()
    initialize_session_state()
    clean_files_on_new_session()

    st.title("Video anonymization")

    if st.session_state.page == "upload":
        show_upload_section()

    elif st.session_state.page == "processing":
        show_processing_page()

    elif st.session_state.page == "done":
        show_final_video()


if __name__ == "__main__":
    main()