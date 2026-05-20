import cv2
from ultralytics import YOLO
import time
import easyocr

# -----------
# constants
# -----------
INPUT_VIDEO_PATH = "hyva_intubointi.mp4"
OUTPUT_VIDEO_PATH = f"{INPUT_VIDEO_PATH.split('.')[0]}_blurred.mp4"
DEBUG_OUTPUT_VIDEO_PATH = f"{INPUT_VIDEO_PATH.split('.')[0]}_debug_boxes.mp4"

MODEL_PATH = "yolov8s-face-lindevs.pt"

BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30

OCR_CONFIDENCE_THRESHOLD = 0.2
OCR_EVERY_N_FRAMES = 5
OCR_HOLD_FRAMES = 10

FACE_CONFIDENCE_THRESHOLD = 0.08
BOX_PADDING = 0.35
FACE_HOLD_FRAMES = 15
IOU_THRESHOLD = 0.30

# -----------
# fucntions
# -----------


def run_face_detection(frame, model,confidence_threshold):
    face_results = model(frame, conf=confidence_threshold, verbose=False)[0]

    return face_results

def get_padded_ocr_boxes(frame, reader, frame_width, frame_height):
    ocr_results = reader.readtext(frame,detail=1)
    boxes = []
    for (bbox, text, conf) in ocr_results:
        if conf < OCR_CONFIDENCE_THRESHOLD:
            continue
        x_values = [point[0] for point in bbox]
        y_values = [point[1] for point in bbox]
        x1 = int(min(x_values))
        y1 = int(min(y_values))
        x2 = int(max(x_values))
        y2 = int(max(y_values))
        padded_box = expand_box(
            (x1, y1, x2, y2),
            BOX_PADDING,
            frame_width,
            frame_height
        )
        if padded_box is not None:
            boxes.append(padded_box)
    return boxes

def get_padded_face_boxes(face_results, frame_width, frame_height):
    boxes = []
    for box in face_results.boxes:
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

def expand_box(box, padding, frame_width, frame_height):
    x1, y1, x2, y2 = box

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    box_width = x2 - x1
    box_height = y2 - y1

    pad_w = int(box_width * padding)
    pad_h = int(box_height * padding)

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


def track_faces(current_boxes, tracked_boxes, iou_threshold, hold_frames):
    updated_tracks = []
    for current_box in current_boxes:
        best_match = None
        best_iou = 0

        for track in tracked_boxes:
            iou = box_iou(current_box, track["box"])

            if iou > best_iou:
                best_iou = iou
                best_match = track

        if best_match is not None and best_iou >= iou_threshold:

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
            ) >= iou_threshold:

                already_kept = True
                break

        if not already_kept:
            old_track["missed"] += 1

            if old_track["missed"] <= hold_frames:
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
reader = easyocr.Reader(['en','sv']) # other languages can be added as well.

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
# -----------------
last_ocr_boxes=[]
last_ocr_frame = 0

tracked_boxes=[]
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    face_results = run_face_detection(frame, model, FACE_CONFIDENCE_THRESHOLD)


    current_face_boxes = get_padded_face_boxes(face_results, frame_width, frame_height)

    if frame_count % OCR_EVERY_N_FRAMES == 0:
        current_ocr_boxes = get_padded_ocr_boxes(frame, reader, frame_width, frame_height)
        last_ocr_boxes = current_ocr_boxes
        last_ocr_frame = frame_count
    elif frame_count - last_ocr_frame <= OCR_HOLD_FRAMES:
        current_ocr_boxes = last_ocr_boxes
    else:
        current_ocr_boxes = []


    debug_frame = face_results.plot()

    tracked_boxes = track_faces(
        current_face_boxes,
        tracked_boxes,
        IOU_THRESHOLD,
        FACE_HOLD_FRAMES
    )
    boxes_to_blur = [track["box"] for track in tracked_boxes]
    boxes_to_blur.extend(current_ocr_boxes)
    blur_boxes(frame, boxes_to_blur)

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

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Saved blurred video to: {OUTPUT_VIDEO_PATH}")
print(f"Saved debug video to: {DEBUG_OUTPUT_VIDEO_PATH}")

print(f"Total runtime: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
