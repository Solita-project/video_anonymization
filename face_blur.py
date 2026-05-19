# 1. Load dependencies and configuration
#    - input video path
#    - output video path
#    - YOLO face model path
#    - blur strength

# 2. Load the face detection model
#    - initialize YOLO with face-specific weights

# 3. Open the input video
#    - check that the file opens successfully
#    - read width, height, fps, and codec info

# 4. Create the output video writer
#    - same resolution and fps as input

# 5. Process video frame by frame
#    - read one frame
#    - run face detection
#    - loop over all detected face bounding boxes
#    - crop each face region
#    - apply blur
#    - paste blurred region back into frame
#    - write processed frame to output video

# 6. Clean up
#    - release video reader
#    - release video writer
#    - close OpenCV windows

# 7. Output
#    - saved anonymized video file



import cv2
from ultralytics import YOLO


INPUT_VIDEO_PATH = "hyva_intubointi.mp4"
OUTPUT_VIDEO_PATH = f"{INPUT_VIDEO_PATH.split('.')[0]}_blurred.mp4"
DEBUG_OUTPUT_VIDEO_PATH = f"{INPUT_VIDEO_PATH.split('.')[0]}_debug_boxes.mp4"

MODEL_PATH = "yolov8s-face-lindevs.pt"

BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30

CONFIDENCE_THRESHOLD = 0.08
BOX_PADDING = 0.35
HOLD_FRAMES = 15
IOU_THRESHOLD = 0.30


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b


    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video {INPUT_VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc,fps,(frame_width, frame_height)
)
debug_out = cv2.VideoWriter(DEBUG_OUTPUT_VIDEO_PATH,fourcc,fps,(frame_width, frame_height)
)

# active tracked faces
tracked_boxes = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]

    debug_frame = results.plot()

    current_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        box_width = x2 - x1
        box_height = y2 - y1

        pad_w = int(box_width * BOX_PADDING)
        pad_h = int(box_height * BOX_PADDING)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(frame.shape[1], x2 + pad_w)
        y2 = min(frame.shape[0], y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            print(
                f"Skipping invalid box: "
                f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"
            )
            continue

        current_boxes.append((x1, y1, x2, y2))


    updated_tracks = []
    for current_box in current_boxes:
        best_match = None
        best_iou = 0

        for track in tracked_boxes:
            iou = box_iou(current_box, track["box"])

            if iou > best_iou:
                best_iou = iou
                best_match = track


        if best_match is not None and best_iou >= IOU_THRESHOLD:

            best_match["box"] = current_box
            best_match["missed"] = 0

            updated_tracks.append(best_match)

        else:
            updated_tracks.append({
                "box": current_box,
                "missed": 0,
            })

    # keep temporarily missing tracks
    for old_track in tracked_boxes:
        already_kept = False

        for new_track in updated_tracks:
            if box_iou(
                old_track["box"],
                new_track["box"]
            ) >= IOU_THRESHOLD:

                already_kept = True
                break

        if not already_kept:
            old_track["missed"] += 1


            if old_track["missed"] <= HOLD_FRAMES:
                updated_tracks.append(old_track)

    tracked_boxes = updated_tracks

    boxes_to_blur = [
        track["box"]
        for track in tracked_boxes
    ]

    for (x1, y1, x2, y2) in boxes_to_blur:
        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            continue

        blurred_face = cv2.GaussianBlur(
            face_region,
            BLUR_KERNEL_SIZE,
            BLUR_SIGMA
        )

        frame[y1:y2, x1:x2] = blurred_face

    out.write(frame)
    debug_out.write(debug_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

    if frame_count >= 300:
        print("Reached 300 frames, stopping early for testing")
        break

cap.release()
out.release()
debug_out.release()

cv2.destroyAllWindows()

print(f"Saved blurred video to: {OUTPUT_VIDEO_PATH}")
print(f"Saved debug video to: {DEBUG_OUTPUT_VIDEO_PATH}")


# -----------
# fucntions
# -----------

def load_model(model_path):
    pass

def open_video(video_path):
    pass

def create_video_writer(output_path, frame_width, frame_height, fps):
    pass

def detect_faces(model, frame):
    pass

def blur_faces(frame, boxes):
    pass

def process_video(input_path, output_path, model_path):
    pass

if __name__ == "__main__":
    process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, MODEL_PATH)