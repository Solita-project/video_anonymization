import cv2
from ultralytics import YOLO
import time
from pathlib import Path

'''
model = YOLO("yolov8s.pt")
print(model.names)
YOLOv8s.pt class names:
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

Use: 62: 'tv', 63: 'laptop',67: 'cell phone'73: 'book'
'''

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO_PATH = ROOT_DIR / "data" / "input" / "video.mp4"
OUTPUT_VIDEO_PATH = ROOT_DIR / "data" / "output" / "video_blurred.mp4"
DEBUG_OUTPUT_VIDEO_PATH = ROOT_DIR / "data" / "output" / "video_debug_boxes.mp4"

BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30
FACE_MODEL_PATH = ROOT_DIR / "models" / "yolov8s-face-lindevs.pt"
# OBJECT_MODEL_PATH = ROOT_DIR / "models" / "yolov8s.pt"

# HIGH_RISK_OBJECT_CLASS_NAMES = ['tv', 'laptop', 'cell phone', 'book']

FACE_CONFIDENCE_THRESHOLD = 0.03
# OBJECT_CONFIDENCE_THRESHOLD = 0.25

FACE_BOX_PADDING = 0.45
# OBJECT_BOX_PADDING = 0.10

FACE_HOLD_FRAMES = 45
IOU_THRESHOLD = 0.15

# OBJECT_EVERY_N_FRAMES = 5
# OBJECT_HOLD_FRAMES = 15

WRITE_DEBUG_VIDEO = False
WRITE_BLURRED_VIDEO = True

MAX_FRAMES = None  # Set to None to process entire video

# -----------
# fucntions
# -----------

def run_face_detection(frame, model, confidence_threshold):

    face_results = model(frame, conf=confidence_threshold, verbose=False)[0]

    return face_results

'''
def run_object_detection(frame, model, confidence_threshold,class_ids):

    object_results = model(frame, conf=confidence_threshold, classes=class_ids, verbose=False)[0]

    return object_results
'''

def get_padded_face_boxes(face_results, frame_width, frame_height):
    boxes = []
    for box in face_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        padded_box = expand_box(
            (x1, y1, x2, y2),
            FACE_BOX_PADDING,
            frame_width,
            frame_height
        )
        if padded_box is None:
            continue

        boxes.append(padded_box)

    return boxes

'''
def get_padded_object_boxes(object_results, frame_width, frame_height):
    boxes = []
    for box in object_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        padded_box = expand_box(
            (x1, y1, x2, y2),
            OBJECT_BOX_PADDING,
            frame_width,
            frame_height
        )
        if padded_box is None:
            continue

        boxes.append(padded_box)

    return boxes
'''

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
    for x1, y1, x2, y2 in boxes_to_blur:
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            continue
        blurred_region = cv2.GaussianBlur(
            region,
            BLUR_KERNEL_SIZE,
            BLUR_SIGMA
        )

        frame[y1:y2, x1:x2] = blurred_region

def draw_boxes(frame, boxes, color, label):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def process_video():
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Track runtime when the video processing starts and ends
    start_time = time.perf_counter()
    print(f"Start to process the video. Start time: {time.ctime()}")

    face_model = YOLO(FACE_MODEL_PATH)
    # object_model = YOLO(OBJECT_MODEL_PATH)
    '''
    high_risk_object_class_ids = [class_id for class_id, class_name in object_model.names.items() if class_name in HIGH_RISK_OBJECT_CLASS_NAMES]
    if not high_risk_object_class_ids:
        raise ValueError("No valid high-risk object class IDs found." "Check HIGH_RISK_OBJECT_CLASS_NAMES and the object model")
    '''
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {INPUT_VIDEO_PATH}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = None
    if WRITE_BLURRED_VIDEO:
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc,fps,(frame_width, frame_height))

    debug_out = None
    if WRITE_DEBUG_VIDEO:
        debug_out = cv2.VideoWriter(DEBUG_OUTPUT_VIDEO_PATH, fourcc,fps,(frame_width, frame_height))


    tracked_boxes=[]
    frame_count = 0
    # last_object_boxes = []
    # last_object_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        face_results = run_face_detection(frame, face_model, FACE_CONFIDENCE_THRESHOLD)
        '''
        # Only run object detection every N frames, and keep results for a few frames to reduce flickering
        if frame_count ==1 or frame_count % OBJECT_EVERY_N_FRAMES == 0:
            object_results = run_object_detection(frame, object_model, OBJECT_CONFIDENCE_THRESHOLD,high_risk_object_class_ids)
            current_object_boxes = get_padded_object_boxes(object_results, frame_width, frame_height)
            last_object_boxes = current_object_boxes
            last_object_frame = frame_count
        elif frame_count - last_object_frame <= OBJECT_HOLD_FRAMES:
            current_object_boxes = last_object_boxes
        else:
            current_object_boxes = []
        '''
        current_face_boxes = get_padded_face_boxes(face_results, frame_width, frame_height)

        tracked_boxes = track_faces(
            current_face_boxes,
            tracked_boxes,
            IOU_THRESHOLD,
            FACE_HOLD_FRAMES
            )
        boxes_to_blur = [track["box"] for track in tracked_boxes]
        # boxes_to_blur += current_object_boxes

        if WRITE_DEBUG_VIDEO:
            debug_frame = frame.copy()
            tracked_face_boxes = [track["box"] for track in tracked_boxes]
            draw_boxes(debug_frame, current_face_boxes, (0, 255, 0), "Face")
            draw_boxes(debug_frame, tracked_face_boxes, (0, 0, 255), "Tracked Face")
            # draw_boxes(debug_frame, current_object_boxes, (255, 0, 0), "Object")
            debug_out.write(debug_frame)

        if WRITE_BLURRED_VIDEO:
            blur_boxes(frame, boxes_to_blur)
            out.write(frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

        if MAX_FRAMES is not None and frame_count >= MAX_FRAMES:
            print(f"Reached maximum frame limit of {MAX_FRAMES}. Stopping.")
            break

    # Release resources and close windows
    cap.release()
    if out is not None:
        out.release()
    if debug_out is not None:
        debug_out.release()

    cv2.destroyAllWindows()

    # Track runtime when the video processing ends
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time


    if WRITE_BLURRED_VIDEO:
        print(f"Saved blurred video to: {OUTPUT_VIDEO_PATH}")

    if WRITE_DEBUG_VIDEO:
        print(f"Saved debug video to: {DEBUG_OUTPUT_VIDEO_PATH}")

    print(f"Video processing stopped at: {time.ctime()}")
    print(f"Total runtime: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    process_video()