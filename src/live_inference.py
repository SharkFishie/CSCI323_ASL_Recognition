"""Simple webcam-based ASL inference scaffold."""

from collections import Counter, deque
import os
import time

import cv2
import numpy as np

from .config import CLASS_NAMES, FRAME_HEIGHT, FRAME_WIDTH, MODEL_PATH, PREDICTION_HISTORY
from .hand_tracking import HandTracker
from .preprocessing import preprocess_crop


def load_classifier(model_path=MODEL_PATH):
    """Load a trained Keras model if it exists."""
    if not os.path.exists(model_path):
        return None

    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None

    try:
        return load_model(model_path)
    except Exception:
        return None


def smooth_prediction(history, label, confidence):
    """Smooth predictions across a short frame history."""
    history.append((label, confidence))

    if len(history) == 0:
        return label, confidence

    most_common_label, _ = Counter(item[0] for item in history).most_common(1)[0]
    matching_confidences = [item[1] for item in history if item[0] == most_common_label]
    average_confidence = float(np.mean(matching_confidences)) if matching_confidences else float(confidence)
    return most_common_label, average_confidence


def classify_crop(frame, model, bbox, prediction_history):
    """Crop the hand region, run the classifier, and draw the label.

    Returns (label, confidence) or (None, None) when nothing was predicted.
    """
    x_min, y_min, x_max, y_max = bbox
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, None

    batch = preprocess_crop(crop)
    if batch is None:
        return None, None

    prediction = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    label = CLASS_NAMES[predicted_index] if 0 <= predicted_index < len(CLASS_NAMES) else "UNKNOWN"

    smoothed_label, smoothed_confidence = smooth_prediction(prediction_history, label, confidence)

    text = f"{smoothed_label}: {smoothed_confidence:.2f}"
    cv2.putText(frame, text, (x_min, max(0, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return smoothed_label, smoothed_confidence


def main():
    """Open the webcam and run the live inference scaffold."""
    model = load_classifier()
    hand_tracker = HandTracker()
    prediction_history = deque(maxlen=PREDICTION_HISTORY)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        # detect_for_video requires strictly increasing timestamps
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Run the Tasks HandLandmarker on this frame
        result = hand_tracker.detect(frame, timestamp_ms)
        hand_tracker.draw(frame, result)

        # Compute a padded bounding box around the first detected hand
        first_hand = hand_tracker.get_first_hand(result)
        bbox = hand_tracker.extract_bbox(frame, first_hand) if first_hand else None
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if model is None:
            cv2.putText(
                frame,
                "Warning: model file missing",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Expected: {MODEL_PATH}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        elif bbox is not None:
            # Classify the cropped hand region when we have a bbox
            classify_crop(frame, model, bbox, prediction_history)

        if not hand_tracker.available:
            cv2.putText(
                frame,
                "Hand tracking unavailable",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            if hand_tracker.error_message:
                cv2.putText(
                    frame,
                    hand_tracker.error_message[:70],
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        cv2.imshow("ASL Live Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
