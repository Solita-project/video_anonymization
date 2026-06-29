from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO_PATH = ROOT_DIR / "data" / "input" / "video.mp4"
OUTPUT_VIDEO_PATH = ROOT_DIR / "data" / "output" / "video_blurred.mp4"
DEBUG_OUTPUT_VIDEO_PATH = ROOT_DIR / "data" / "output" / "video_debug_boxes.mp4"

FACE_MODEL_PATH = ROOT_DIR / "models" / "yolov8s-face-lindevs.pt"
OBJECT_MODEL_PATH = ROOT_DIR / "models" / "yolov8s.pt"
HEAD_MODEL_PATH = ROOT_DIR / "models" / "yolov8_head_medium.pt"

HIGH_RISK_OBJECT_CLASS_NAMES = ['tv', 'laptop', 'cell phone', 'book']

PROFILES = {
    "cpr":{
        "description": "Preserve body movement and compression context.",
        "face_confidence": 0.25,
        "head_confidence": 0.25,
        "object_confidence": 0.25,
        "face_padding": 0.45,
        "head_padding": 0.20,
        "object_padding": 0.10,
        "face_hold_frames": 30,
        "head_hold_frames": 30,
        "object_hold_frames": 30,
        "face_iou_threshold": 0.15,
        "head_iou_threshold": 0.15,
        "object_iou_threshold": 0.15,
        "object_every_n_frames": 3,
        "blur_kernel_size": (99, 99),
        "blur_sigma": 30,
    },
    "intubation":{
        "description": "Protect identity while preserving procedure context where possible.",
        "face_confidence": 0.25,
        "head_confidence": 0.25,
        "object_confidence": 0.25,
        "face_padding": 0.35,
        "head_padding": 0.15,
        "object_padding": 0.10,
        "face_hold_frames": 30,
        "head_hold_frames": 30,
        "object_hold_frames": 30,
        "face_iou_threshold": 0.15,
        "head_iou_threshold": 0.15,
        "object_iou_threshold": 0.15,
        "object_every_n_frames": 3,
        "blur_kernel_size": (99, 99),
        "blur_sigma": 30,
    }
}
