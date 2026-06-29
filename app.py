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
import streamlit as st

from ui.final_ui import show_final_video
from ui.manual_blur_ui import show_manual_blur_section
from ui.transcript_review_ui import show_transcript_review_section as show_transcript_review_ui
from ui.upload_ui import show_upload_section
from ui.video_review_ui import show_video_section


# Find project root
ROOT_DIR = Path(__file__).resolve().parent

# Define project folders
INPUT_DIR = ROOT_DIR / "data" / "input"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

# Define important files
INPUT_VIDEO_FILE = INPUT_DIR / "video.mp4"
BLURRED_VIDEO_FILE = OUTPUT_DIR / "video_blurred.mp4"
BLURRED_PREVIEW_FILE = OUTPUT_DIR / "video_blurred_preview.mp4"
VIDEO_REPORT_FILE = OUTPUT_DIR / "video_report.json"
ORIGINAL_TRANSCRIPT_FILE = OUTPUT_DIR / "transcription.json"
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
    "RESTORE_ORIGINAL",
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
    # Run pipeline script, script output is shown in the terminalS
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

    for key in [
        "review_data",
        "original_review_data",
        "transcript_process",
        "selected_segment",
        "active_review_segment",
        "reviewed_segments",
        "automatic_review_done",
        "editing_segment",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def clean_files_on_new_session():
    # Clean previous files when a new Streamlit session starts
    if "startup_cleanup_done" in st.session_state:
        return

    clean_old_outputs()
    reset_app_state()

    st.session_state.startup_cleanup_done = True


def resolve_ffmpeg():
    # Use bundled Windows ffmpeg only on Windows
    if os.name == "nt" and LOCAL_FFMPEG.exists():
        return str(LOCAL_FFMPEG)

    # Otherwise use ffmpeg from system PATH
    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg or add it to PATH before creating browser video previews."
    )


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


def run_video_anonymization(profile_name):
    # Run automatic video anonymization and create a browser-friendly preview
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "run" / "run_video.py"),
            "--profile",
            profile_name,
        ],
        cwd=ROOT_DIR,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        st.error("Step failed: Anonymize video")
        st.error("Check the terminal for details.")
        st.stop()

    try:
        create_browser_preview(BLURRED_VIDEO_FILE, BLURRED_PREVIEW_FILE)
    except RuntimeError as error:
        st.error(str(error))
        st.stop()


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


def load_video_report():
    # Load video review report if it exists
    if not VIDEO_REPORT_FILE.exists():
        return None

    with open(VIDEO_REPORT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_processed_video_duration():
    # Use the report to limit manual blur inputs to the processed video length.
    report = load_video_report()

    if report is None:
        return None

    duration = report.get("processed_duration_seconds")

    if duration:
        return float(duration)

    processed_frames = report.get("processed_frames")
    fps = report.get("fps")

    if not processed_frames or not fps:
        return None

    return float(processed_frames) / float(fps)


def get_last_extractable_frame_time():
    # Frame extraction needs an actual frame timestamp, not the end boundary
    report = load_video_report()

    if report is None:
        return None

    processed_frames = report.get("processed_frames")
    fps = report.get("fps")

    if not processed_frames or not fps:
        return None

    return max(0.0, (float(processed_frames) - 1.0) / float(fps))


def clamp_time_input_state(key, max_value):
    # Clamp old Streamlit widget state when a new shorter video/report is loaded
    if max_value is None:
        return

    current_value = st.session_state.get(key)

    if current_value is not None and current_value > max_value:
        st.session_state[key] = max_value


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

    # Do not redirect stdout or stderr (this lets the background script print to the terminal)S
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


def run_final_pipeline():
    # Run the rest of the pipeline after manual review
    run_script("Detect speakers", "run/run_speaker.py")
    run_script("Merge transcript and speakers", "src/merged.py")
    run_script("Generate anonymized audio", "run/run_tts.py")
    run_script("Merge anonymized audio and video", "src/final_merge.py")


def show_processing_page():
    # Show video review first, then manual blur tools and transcript review when available
    status = get_transcript_status()

    show_video_section(
        status=status,
        blurred_preview_file=BLURRED_PREVIEW_FILE,
        load_video_report=load_video_report,
        read_transcript_log=read_transcript_log,
    )

    show_manual_blur_section(
        manual_blur_file=MANUAL_BLUR_FILE,
        manual_blur_frame_file=MANUAL_BLUR_FRAME_FILE,
        blurred_video_file=BLURRED_VIDEO_FILE,
        blurred_preview_file=BLURRED_PREVIEW_FILE,
        get_processed_video_duration=get_processed_video_duration,
        get_last_extractable_frame_time=get_last_extractable_frame_time,
        clamp_time_input_state=clamp_time_input_state,
        run_manual_blur_command=run_manual_blur_command,
        create_browser_preview=create_browser_preview,
    )

    handle_video_decision(status)

    continue_clicked = show_transcript_review_ui(
        transcript_file=CLEANED_TRANSCRIPT_FILE,
        original_transcript_file=ORIGINAL_TRANSCRIPT_FILE,
        replacements=REPLACEMENTS,
    )

    if continue_clicked:
        save_cleaned_transcript(st.session_state.review_data)

        with st.spinner("Running the rest of the anonymization..."):
            run_final_pipeline()

        st.session_state.page = "done"
        st.success("Anonymization complete.")
        show_final_video(FINAL_VIDEO_FILE)


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
        show_upload_section(
            reset_app_state=reset_app_state,
            clean_old_outputs=clean_old_outputs,
            save_uploaded_video=save_uploaded_video,
            run_video_anonymization=run_video_anonymization,
            start_transcript_process=start_transcript_process,
        )

    elif st.session_state.page == "processing":
        show_processing_page()

    elif st.session_state.page == "done":
        show_final_video(FINAL_VIDEO_FILE)


if __name__ == "__main__":
    main()
