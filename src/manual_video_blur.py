# Manual video blur helper.
# This script can extract one frame from the anonymized video
# and apply manually selected blur areas to the video.
# Usage:
# python src/manual_video_blur.py extract --time 12.5
# python src/manual_video_blur.py apply

from pathlib import Path
import argparse
import json
import os
import cv2


# Find project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define input and output files
VIDEO_FILE = ROOT_DIR / "data" / "output" / "video_blurred.mp4"
TEMP_VIDEO_FILE = ROOT_DIR / "data" / "output" / "video_blurred_manual_tmp.mp4"
FRAME_FILE = ROOT_DIR / "data" / "output" / "manual_blur_frame.jpg"
ANNOTATION_FILE = ROOT_DIR / "data" / "output" / "manual_video_blurs.json"


def load_annotations():
    # Load manual blur annotations from JSON
    if not ANNOTATION_FILE.exists():
        return {"annotations": []}

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        data["annotations"] = []

    return data


def make_odd(value):
    # Gaussian blur kernel size must be an odd number
    value = int(value)

    if value < 3:
        value = 3

    if value % 2 == 0:
        value += 1

    return value


def clamp_box(annotation, frame_width, frame_height):
    # Keep blur box inside video frame boundaries
    x = int(round(annotation["x"]))
    y = int(round(annotation["y"]))
    width = int(round(annotation["width"]))
    height = int(round(annotation["height"]))

    x1 = max(0, min(frame_width - 1, x))
    y1 = max(0, min(frame_height - 1, y))
    x2 = max(0, min(frame_width, x + width))
    y2 = max(0, min(frame_height, y + height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def blur_area(frame, annotation):
    # Apply blur to one selected area in one frame
    frame_height, frame_width = frame.shape[:2]

    box = clamp_box(annotation, frame_width, frame_height)

    if box is None:
        return frame

    x1, y1, x2, y2 = box

    blur_strength = make_odd(annotation.get("blur_strength", 45))

    area = frame[y1:y2, x1:x2]

    if area.size == 0:
        return frame

    blurred_area = cv2.GaussianBlur(
        area,
        (blur_strength, blur_strength),
        0,
    )

    frame[y1:y2, x1:x2] = blurred_area

    return frame


def get_annotation_time_range(annotation):
    # Support both explicit end_time and start_time + keep_blur_seconds.
    start_time = float(annotation["start_time"])

    if "end_time" in annotation:
        end_time = float(annotation["end_time"])
    else:
        keep_blur_seconds = float(annotation["keep_blur_seconds"])
        end_time = start_time + keep_blur_seconds

    return start_time, end_time


def extract_frame(timestamp):
    # Extract one frame from video_blurred.mp4 at the selected timestamp
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_FILE}")

    FRAME_FILE.parent.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(VIDEO_FILE))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

    video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    success, frame = video.read()
    video.release()

    if not success:
        raise RuntimeError(f"Could not extract frame at {timestamp} seconds")

    cv2.imwrite(str(FRAME_FILE), frame)

    print(f"Frame saved: {FRAME_FILE}", flush=True)


def apply_manual_blurs():
    # Apply all saved manual blur annotations to video_blurred.mp4
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_FILE}")

    data = load_annotations()
    annotations = data.get("annotations", [])

    if not annotations:
        print("No manual blur annotations found.", flush=True)
        return

    video = cv2.VideoCapture(str(VIDEO_FILE))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        video.release()
        raise RuntimeError("Invalid video FPS")

    TEMP_VIDEO_FILE.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(TEMP_VIDEO_FILE),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Could not create output video: {TEMP_VIDEO_FILE}")

    frame_index = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        current_time = frame_index / fps

        for annotation in annotations:
            start_time, end_time = get_annotation_time_range(annotation)

            if start_time <= current_time <= end_time:
                frame = blur_area(frame, annotation)

        writer.write(frame)
        frame_index += 1

    video.release()
    writer.release()

    # Replace video_blurred.mp4 with the manually improved version
    os.replace(str(TEMP_VIDEO_FILE), str(VIDEO_FILE))

    print(f"Manual blurs applied to: {VIDEO_FILE}", flush=True)


def main():
    # Read command line arguments
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument(
        "--time",
        type=float,
        required=True,
        help="Timestamp in seconds",
    )

    subparsers.add_parser("apply")

    args = parser.parse_args()

    if args.command == "extract":
        extract_frame(args.time)

    elif args.command == "apply":
        apply_manual_blurs()


if __name__ == "__main__":
    main()
