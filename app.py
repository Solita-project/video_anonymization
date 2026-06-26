# Streamlit user interface for the video anonymization pipeline.
# Upload a video, review and improve video anonymization, review the transcript manually,
# and continue the pipeline to create the final anonymized video.
# Usage:
# source venvs/core/Scripts/activate
# streamlit run app.py

from pathlib import Path
import inspect
import json
import os
import shutil
import subprocess
import sys
import time

from PIL import Image
import streamlit as st

# Compatibility fix for streamlit-drawable-canvas with newer Streamlit versions.
# streamlit-drawable-canvas expects image_to_url in streamlit.elements.image.
# Newer Streamlit versions moved it to streamlit.elements.lib.image_utils
# and changed how image width is handled.
try:
    from streamlit.elements import image as st_image
    from streamlit.elements.lib.image_utils import image_to_url as streamlit_image_to_url

    def image_to_url_compat(
        image,
        width=-1,
        clamp=True,
        channels="RGB",
        output_format="PNG",
        image_id="",
        *args,
        **kwargs,
    ):
        # streamlit-drawable-canvas passes width using an older Streamlit API.
        # Make sure width is a normal integer before calling Streamlit's current function.
        try:
            width = int(width)
        except Exception:
            width_value = getattr(width, "width", -1)

            try:
                width = int(width_value)
            except Exception:
                width = -1

        signature = inspect.signature(streamlit_image_to_url)
        supported_parameters = signature.parameters

        call_kwargs = {}

        if "layout_config" in supported_parameters:
            try:
                from streamlit.elements.lib.layout_utils import LayoutConfig

                call_kwargs["layout_config"] = LayoutConfig(width=width)
            except Exception:
                pass

        if "width" in supported_parameters:
            call_kwargs["width"] = width

        if "clamp" in supported_parameters:
            call_kwargs["clamp"] = clamp

        if "channels" in supported_parameters:
            call_kwargs["channels"] = channels

        if "output_format" in supported_parameters:
            call_kwargs["output_format"] = output_format

        if "image_format" in supported_parameters:
            call_kwargs["image_format"] = output_format

        if "image_id" in supported_parameters:
            call_kwargs["image_id"] = image_id

        return streamlit_image_to_url(
            image,
            **call_kwargs,
        )

    st_image.image_to_url = image_to_url_compat

except Exception:
    pass

from streamlit_drawable_canvas import st_canvas
from src.transcript_review_ui import show_transcript_review_section as show_transcript_review_ui

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

    if "original_review_data" in st.session_state:
        del st.session_state.original_review_data

    if "transcript_process" in st.session_state:
        del st.session_state.transcript_process

    for key in [
    "active_review_segment",
    "reviewed_segments",
    "selected_segment",
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
    # Use bundled Windows ffmpeg only on Windows.
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
    # Frame extraction needs an actual frame timestamp, not the end boundary.
    report = load_video_report()

    if report is None:
        return None

    processed_frames = report.get("processed_frames")
    fps = report.get("fps")

    if not processed_frames or not fps:
        return None

    return max(0.0, (float(processed_frames) - 1.0) / float(fps))


def clamp_time_input_state(key, max_value):
    # Clamp old Streamlit widget state when a new shorter video/report is loaded.
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
        keep_blur_seconds = annotation.get("keep_blur_seconds")
        x = annotation["x"]
        y = annotation["y"]
        width = annotation["width"]
        height = annotation["height"]

        if keep_blur_seconds is None:
            end_time = float(annotation["end_time"])
            keep_blur_seconds = end_time - start_time

        st.write(
            f"Area {index + 1}: "
            f"start={start_time:.1f}s, keep={float(keep_blur_seconds):.1f}s, "
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


def show_video_report():
    # Show structured video review report below the anonymized video
    report = load_video_report()

    if report is None:
        st.info("Video review report was not found.")
        return

    review = report.get("review", {})
    summary = review.get("summary", {})
    warnings = review.get("warnings", [])
    suggested_ranges = review.get("suggested_ranges", [])

    st.write("### Video review report")

    st.write(
        f"Profile: `{report.get('profile', 'unknown')}`  "
        f"| Processed frames: `{report.get('processed_frames', 0)}` / `{report.get('total_frames', 0)}`"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Suggested ranges",
        summary.get("suggested_review_range_count", 0),
    )
    col2.metric(
        "Review seconds",
        summary.get("suggested_review_duration_seconds", 0),
    )
    col3.metric(
        "Warnings",
        summary.get("warning_count", 0),
    )

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No priority review warnings in this report.")

    if suggested_ranges:
        st.write("#### Suggested review ranges")

        for index, item in enumerate(suggested_ranges, start=1):
            st.write(
                f"{index}. `{item.get('priority', 'unknown')}` "
                f"{item.get('start_time_seconds')}s–{item.get('end_time_seconds')}s: "
                f"{item.get('reason', '')}"
            )
            st.caption(item.get("suggested_action", ""))
    else:
        st.info("No suggested review ranges.")


def show_upload_section():
    # Show video upload section
    st.write("## Upload video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4"],
    )

    profile_name = st.selectbox(
        "Video anonymization profile",
        options=["cpr", "intubation"],
        format_func=lambda value: "CPR" if value == "cpr" else "Intubation",
        key="video_profile",
    )

    if not uploaded_file:
        return

    st.write(f"Selected file: `{uploaded_file.name}`")

    if st.button("Anonymize video"):
        reset_app_state()
        clean_old_outputs()
        save_uploaded_video(uploaded_file)

        with st.spinner("Running video anonymization..."):
            run_video_anonymization(profile_name)

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

    show_video_report()

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

    video_duration = get_processed_video_duration()
    last_frame_time = get_last_extractable_frame_time()

    if video_duration is not None:
        st.caption(f"Processed video duration: {video_duration:.1f} seconds")

    if last_frame_time is not None:
        clamp_time_input_state("manual_frame_time", last_frame_time)

        frame_time = st.slider(
            "Frame time in seconds",
            min_value=0.0,
            max_value=float(last_frame_time),
            step=0.1,
            format="%.1f",
            key="manual_frame_time",
        )
    else:
        frame_time = st.number_input(
            "Frame time in seconds",
            min_value=0.0,
            step=0.1,
            format="%.1f",
            key="manual_frame_time",
        )

    if st.button("Show frame"):
        if MANUAL_BLUR_FRAME_FILE.exists():
            MANUAL_BLUR_FRAME_FILE.unlink()

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

        default_start_time = frame_time

        if video_duration is not None:
            default_start_time = min(default_start_time, video_duration)

        clamp_time_input_state("manual_blur_start_time", video_duration)

        start_time_kwargs = {
            "label": "Start time in seconds",
            "min_value": 0.0,
            "value": default_start_time,
            "step": 0.1,
            "format": "%.1f",
            "key": "manual_blur_start_time",
        }

        if video_duration is not None:
            start_time_kwargs["max_value"] = video_duration

        start_time = st.number_input(**start_time_kwargs)

        max_keep_seconds = None
        if video_duration is not None:
            max_keep_seconds = max(0.0, video_duration - start_time)
            clamp_time_input_state("manual_blur_keep_seconds", max_keep_seconds)

        if max_keep_seconds is not None and max_keep_seconds < 0.1:
            st.warning("Start time is too close to the end of the processed video. Choose an earlier start time.")
            return

        keep_seconds_kwargs = {
            "label": "Keep this blur for seconds",
            "min_value": 0.1,
            "value": min(1.0, max_keep_seconds) if max_keep_seconds is not None else 1.0,
            "step": 0.1,
            "format": "%.1f",
            "key": "manual_blur_keep_seconds",
        }

        if max_keep_seconds is not None:
            keep_seconds_kwargs["max_value"] = max_keep_seconds

        keep_blur_seconds = st.number_input(**keep_seconds_kwargs)

        blur_strength = st.slider(
            "Blur strength",
            min_value=15,
            max_value=99,
            value=45,
            step=2,
        )

        if st.button("Save blur area"):
            end_time = start_time + keep_blur_seconds

            if video_duration is not None:
                end_time = min(end_time, video_duration)

            if end_time <= start_time:
                st.error("Blur duration must be greater than zero.")
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
                "keep_blur_seconds": float(keep_blur_seconds),
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
        show_final_video()


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
