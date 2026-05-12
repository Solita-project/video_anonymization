# 1. Acquire video
    # load the video
    # read the video frame by frame
    # Use color conversion (skin tone needs to be detected)

# 2. blurring the faces in the video
    # use YOLO to detect faces in each frame
    # YOLO will return the bounding boxes which has coordinate of (x1, y1, x2, y2) of the detected faces

# 3. blur faces
    # take the detected face regions from the original frame using the bounding box coordinates
    # crop the face region from the original frame
    # apply Gaussian blur to the detected face regions
    # put the blurred face back to the original frame at the same location using the bounding box coordinates

# 4. openCV writes each processed frame to a new video file
# OUTPUT: a new video file with blurred faces

import cv2
import numpy as np
from ultralytics import YOLO

#Load the YOLO model
model = YOLO('yolov11n.pt')

# REMEMBER TO REPLACE THE VIDEO PATH
cap = cv2.VideoCapture('input_video.mp4')
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    break
cv2.destroyAllWindows()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# use YOLO for face detection