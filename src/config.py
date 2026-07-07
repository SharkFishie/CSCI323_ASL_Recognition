"""Configuration settings for the live webcam inference scaffold."""

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "SPACE", "DELETE", "NOTHING"
]

MODEL_PATH = "results/models/best_model.h5"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
PREDICTION_HISTORY = 5
