"""Configuration settings for the live webcam inference scaffold.

The classifier works on hand-*skeleton* images (MediaPipe landmarks drawn on
black), not raw photos. It is trained on the Kaggle "asl-alphabet-wireframes"
dataset, which covers the 24 static ASL letters (J and Z are motion signs and
are excluded). See src/skeleton.py for the shared rendering/normalization.
"""

# Fallback only. The authoritative order is written to CLASS_NAMES_PATH at
# train time (derived from the dataset's sorted class folders).
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]

MODEL_PATH = "results/models/best_model.h5"
CLASS_NAMES_PATH = "results/models/class_names.json"

# Per-letter reference hand poses for the tutor's graded feedback
# (built by scripts/build_reference_poses.py; small, committed to git).
REFERENCE_POSES_PATH = "results/models/reference_poses.json"
# Real-photo ASL dataset the reference landmarks are extracted from. The
# wireframes dataset can't be used here: MediaPipe does not detect hands in
# skeleton *drawings* (verified 0/30), and the renders don't carry coordinates.
REFERENCE_DATASET = "grassknoted/asl-alphabet"

# Square input size (px) shared by training prep and live inference.
IMAGE_SIZE = 96

# Kaggle dataset (downloaded via kagglehub) and the processed cache we train on.
KAGGLE_DATASET = "dylanpallickara129/asl-alphabet-wireframes"
PROCESSED_DIR = "data/wireframes_processed"
# Cap images per class during prep to keep CPU training time reasonable
# (set to None to use every image).
MAX_PER_CLASS = 2500
EPOCHS = 12
BATCH_SIZE = 64

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
PREDICTION_HISTORY = 5

# MediaPipe Tasks model (optional - place under repo `models/`)
HAND_LANDMARKER_MODEL_PATH = "models/hand_landmarker.task"
# number of hands to detect
NUM_HANDS = 1
