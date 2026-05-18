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
import numpy as np
from ultralytics import YOLO


INPUT_VIDEO_PATH = "hyva_intubointi.mp4"
OUTPUT_VIDEO_PATH = f"{INPUT_VIDEO_PATH.split('.')[0]}_blurred.mp4"
MODEL_PATH = "yolov8s-face-lindevs.pt"

BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30

CONFIDENCE_THRESHOLD = 0.10
BOX_PADDING = 0.25
HOLD_FRAMES = 10

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video {INPUT_VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
DEBUG_OUTPUT_VIDEO_PATH = f"{INPUT_VIDEO_PATH.split('.')[0]}_debug_boxes.mp4"
debug_out = cv2.VideoWriter(DEBUG_OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# last detected boxes
last_boxes = []
frame_count = 0
missed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    results = model(frame, conf=CONFIDENCE_THRESHOLD,verbose=False)[0]
    debug_frame = results.plot()

    current_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])

        x1 =int(x1)
        y1 =int(y1)
        x2 =int(x2)
        y2 =int(y2)

        box_width = x2 - x1
        box_height = y2 - y1

        pad_w = int(box_width * BOX_PADDING)
        pad_h = int(box_height * BOX_PADDING)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(frame.shape[1], x2 + pad_w)
        y2 = min(frame.shape[0], y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            print(f"Skipping invalid box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            continue
        current_boxes.append((x1, y1, x2, y2))

        # when YOLO detect faces
        if current_boxes:
            boxes_to_blur = current_boxes
            last_boxes = current_boxes
            missed_frames = 0
        # when YOLO miss faces, we can use the last detected boxes to blur for a few frames
        else:
            missed_frames += 1
            if missed_frames <= HOLD_FRAMES:
                boxes_to_blur = last_boxes
            else:
                boxes_to_blur = []
        for (x1, y1, x2, y2) in boxes_to_blur:
            face_region = frame[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, BLUR_KERNEL_SIZE, BLUR_SIGMA)
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