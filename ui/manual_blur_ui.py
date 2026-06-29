# Manual video blur UI for the Streamlit app
# (the actual OpenCV video processing in src/manual_video_blur.py)

import inspect
import json
from PIL import Image
import streamlit as st

# Compatibility fix for streamlit-drawable-canvas
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
        # (Make sure width is always a normal integer)
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

        return streamlit_image_to_url(image, **call_kwargs)

    st_image.image_to_url = image_to_url_compat

except Exception:
    pass

from streamlit_drawable_canvas import st_canvas


def load_manual_blurs(manual_blur_file):
    # Load saved manual blur areas
    if not manual_blur_file.exists():
        return {"annotations": []}

    with open(manual_blur_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        data["annotations"] = []

    return data


def save_manual_blurs(manual_blur_file, data):
    # Save manual blur areas to JSON
    manual_blur_file.parent.mkdir(parents=True, exist_ok=True)

    with open(manual_blur_file, "w", encoding="utf-8") as f:
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


def show_saved_blur_areas(manual_blur_file):
    # Show saved manual blur areas
    data = load_manual_blurs(manual_blur_file)
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
        save_manual_blurs(manual_blur_file, {"annotations": []})
        st.rerun()


def show_manual_blur_section(
    manual_blur_file,
    manual_blur_frame_file,
    blurred_video_file,
    blurred_preview_file,
    get_processed_video_duration,
    get_last_extractable_frame_time,
    clamp_time_input_state,
    run_manual_blur_command,
    create_browser_preview,
):
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
        if manual_blur_frame_file.exists():
            manual_blur_frame_file.unlink()

        run_manual_blur_command(
            ["extract", "--time", str(frame_time)],
            "Extract frame",
        )

        st.session_state.canvas_key_version += 1
        st.rerun()

    if manual_blur_frame_file.exists():
        original_image = Image.open(manual_blur_frame_file).convert("RGB")
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

            data = load_manual_blurs(manual_blur_file)
            data["annotations"].append(annotation)
            save_manual_blurs(manual_blur_file, data)

            st.success("Blur area saved.")
            st.rerun()

    show_saved_blur_areas(manual_blur_file)

    data = load_manual_blurs(manual_blur_file)
    annotations = data.get("annotations", [])

    if annotations:
        if st.button("Run additional blurs"):
            with st.spinner("Running additional blurs..."):
                run_manual_blur_command(
                    ["apply"],
                    "Run additional blurs",
                )

                create_browser_preview(
                    blurred_video_file,
                    blurred_preview_file,
                )

            # Clear annotations after applying them to avoid applying the same areas twice
            save_manual_blurs(manual_blur_file, {"annotations": []})

            st.session_state.show_manual_blur = False
            st.session_state.video_approved = False
            st.session_state.canvas_key_version += 1

            st.success("Additional blurs were applied.")
            st.rerun()
