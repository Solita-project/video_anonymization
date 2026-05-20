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
import time

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

# -----------
# fucntions
# -----------

def run_face_detection(frame, model,CONFIDENCE_THRESHOLD):
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    return results

def get_padded_face_boxes(results, frame_width, frame_height):

    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        padded_box = expand_box(
            (x1, y1, x2, y2),
            BOX_PADDING,
            frame_width,
            frame_height
        )
        if padded_box is None:
            continue

        boxes.append(padded_box)

    return boxes

def expand_box(box, BOX_PADDING, frame_width, frame_height):
    x1, y1, x2, y2 = box

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
    x2 = min(frame_width, x2 + pad_w)
    y2 = min(frame_height, y2 + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

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


def track_faces(current_boxes, tracked_boxes, IOU_THRESHOLD, HOLD_FRAMES):
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

    return updated_tracks


def blur_boxes(frame, boxes_to_blur):
    for(x1,y1,x2,y2) in boxes_to_blur:
        region = frame[y1:y2, x1:x2]
        if region.size==0:
            continue
        blurred_region = cv2.GaussianBlur(
            region,
            BLUR_KERNEL_SIZE,
            BLUR_SIGMA
        )

        frame[y1:y2, x1:x2] = blurred_region

start_time = time.perf_counter()
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
# -----------------
# main processing loop
# logic:
# 1. run face detection on current frame
# 2. get bounding boxes of detected faces
# 3. track faces across frames using IoU and a simple tracking logic
# 4. blur the tracked boxes in the current frame
# 5. write the processed frame to output video
# -----------------


tracked_boxes=[]
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = run_face_detection(frame, model, CONFIDENCE_THRESHOLD)

    current_boxes = get_padded_face_boxes(results, frame_width, frame_height)

    debug_frame = results.plot()

    tracked_boxes = track_faces(
        current_boxes,
        tracked_boxes,
        IOU_THRESHOLD,
        HOLD_FRAMES
    )
    boxes_to_blur = [track["box"] for track in tracked_boxes]
    blur_boxes(frame, boxes_to_blur)

    out.write(frame)
    debug_out.write(debug_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

    if frame_count >= 600:
        print("Reached 600 frames, stopping early for testing")
        break

cap.release()
out.release()
debug_out.release()

cv2.destroyAllWindows()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Saved blurred video to: {OUTPUT_VIDEO_PATH}")
print(f"Saved debug video to: {DEBUG_OUTPUT_VIDEO_PATH}")
print(f"Total runtime: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
