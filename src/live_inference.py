"""Simple webcam-based ASL inference scaffold."""

from collections import Counter, deque
import json
import os
import time

import cv2
import numpy as np

from .config import (
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_PATH,
    PREDICTION_HISTORY,
)
from .hand_tracking import HandTracker
from .preprocessing import preprocess_skeleton
from .skeleton import render_skeleton


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


def load_class_names(path=CLASS_NAMES_PATH):
    """Return the class-name list saved at train time, or the config fallback.

    The saved list is authoritative: its order matches the model's output
    neurons, which need not match the alphabetical CLASS_NAMES fallback.
    """
    if os.path.exists(path):
        try:
            with open(path) as fh:
                names = json.load(fh)
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    return CLASS_NAMES


def smooth_prediction(history, label, confidence):
    """Smooth predictions across a short frame history."""
    history.append((label, confidence))

    if len(history) == 0:
        return label, confidence

    most_common_label, _ = Counter(item[0] for item in history).most_common(1)[0]
    matching_confidences = [item[1] for item in history if item[0] == most_common_label]
    average_confidence = float(np.mean(matching_confidences)) if matching_confidences else float(confidence)
    return most_common_label, average_confidence


def classify_hand(model, landmarks, prediction_history, class_names):
    """Render the hand skeleton and classify it.

    Returns (label, confidence, skeleton_rgb) or (None, None, None).
    """
    skeleton = render_skeleton(landmarks)
    batch = preprocess_skeleton(skeleton)
    if batch is None:
        return None, None, None

    prediction = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    label = class_names[predicted_index] if 0 <= predicted_index < len(class_names) else "UNKNOWN"

    smoothed_label, smoothed_confidence = smooth_prediction(prediction_history, label, confidence)
    return smoothed_label, smoothed_confidence, skeleton


SIDEBAR_WIDTH = 360


def draw_sidebar(frame, label, confidence, skeleton_rgb, min_confidence=0.5):
    """Draw a full-height right sidebar with the letter, confidence, and input.

    Consolidates the readout into one clear column: a big letter up top, a
    confidence bar, and the rendered skeleton ("model input") below.
    """
    h, w = frame.shape[:2]
    x0 = w - SIDEBAR_WIDTH

    # Solid-ish dark panel.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (x0, 0), (x0, h), (0, 200, 0), 2)

    cx = x0 + SIDEBAR_WIDTH // 2
    cv2.putText(frame, "DETECTED LETTER", (x0 + 24, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2)

    # Big centered glyph, vertically centered in a fixed band [70, 250].
    confident = label is not None and confidence >= min_confidence
    glyph = label if confident else ("?" if label is not None else "-")
    color = (0, 255, 0) if confident else (0, 165, 255)
    scale, thick = 6.5, 14
    (tw, th), _ = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    baseline_y = 70 + (180 + th) // 2
    cv2.putText(frame, glyph, (cx - tw // 2, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    # Confidence bar.
    bar_x, bar_y, bar_w, bar_h = x0 + 30, 290, SIDEBAR_WIDTH - 60, 26
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    if label is not None:
        fill = int(bar_w * max(0.0, min(1.0, confidence)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    conf_text = f"{confidence:.0%} confidence" if label is not None else "no hand detected"
    cv2.putText(frame, conf_text, (bar_x, bar_y + bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # Skeleton "model input" preview near the bottom of the sidebar.
    size = SIDEBAR_WIDTH - 80
    px = x0 + (SIDEBAR_WIDTH - size) // 2
    py = h - size - 40
    cv2.putText(frame, "model input", (px, py - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    if skeleton_rgb is not None:
        thumb = cv2.cvtColor(cv2.resize(skeleton_rgb, (size, size)), cv2.COLOR_RGB2BGR)
        frame[py:py + size, px:px + size] = thumb
    cv2.rectangle(frame, (px, py), (px + size, py + size), (70, 70, 70), 1)


def main():
    """Open the webcam and run the live inference scaffold."""
    model = load_classifier()
    class_names = load_class_names()
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
        elif first_hand is not None and bbox is not None:
            # Render the detected hand as a skeleton and classify that
            label, confidence, skeleton = classify_hand(
                model, first_hand, prediction_history, class_names
            )
            draw_sidebar(frame, label, confidence if confidence else 0.0, skeleton)
        elif model is not None:
            # No hand detected this frame — clear the readout.
            prediction_history.clear()
            draw_sidebar(frame, None, 0.0, None)

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
