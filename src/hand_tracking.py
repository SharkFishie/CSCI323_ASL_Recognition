"""Small wrapper around MediaPipe Hands for webcam inference."""

import cv2

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover - depends on local environment
    mp = None
    MEDIAPIPE_IMPORT_ERROR = exc
else:
    MEDIAPIPE_IMPORT_ERROR = None

from .config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE


class HandTracker:
    """Detect a single hand and extract a simple bounding box."""

    def __init__(self):
        self.available = False
        self.error_message = None

        if mp is None:
            self.error_message = f"MediaPipe unavailable: {MEDIAPIPE_IMPORT_ERROR}"
            self.mp_hands = None
            self.mp_draw = None
            return

        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on local environment
            self.error_message = f"MediaPipe init failed: {exc}"
            self.mp_hands = None
            self.mp_draw = None

    def detect(self, frame):
        """Run MediaPipe on a BGR frame and return the results."""
        if not self.available:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def draw(self, frame, results):
        """Draw landmarks and connections if a hand is detected."""
        if not self.available or results is None or not getattr(results, "multi_hand_landmarks", None):
            return

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def extract_bbox(self, frame, hand_landmarks):
        """Create a padded bounding box from normalized hand landmarks."""
        if not self.available or hand_landmarks is None:
            return None

        height, width = frame.shape[:2]
        x_values = []
        y_values = []

        for landmark in hand_landmarks.landmark:
            x_values.append(landmark.x)
            y_values.append(landmark.y)

        x_min = int(min(x_values) * width)
        y_min = int(min(y_values) * height)
        x_max = int(max(x_values) * width)
        y_max = int(max(y_values) * height)

        pad_x = int(0.15 * (x_max - x_min + 1))
        pad_y = int(0.15 * (y_max - y_min + 1))

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(width - 1, x_max + pad_x)
        y_max = min(height - 1, y_max + pad_y)

        return x_min, y_min, x_max, y_max
