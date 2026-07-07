# Realtime inference plan

The goal is to extend the existing static ASL alphabet image-classification project into a simple webcam-based inference scaffold.

## First pass

- Keep the current CNN image-classification workflow as the baseline.
- Use OpenCV for webcam capture and display.
- Use MediaPipe Hands for hand detection and landmark tracking.
 - Use MediaPipe Hands (Tasks API) for hand detection and landmark tracking when available. The project now prefers the MediaPipe Tasks `HandLandmarker` for long-term compatibility; place a `models/hand_landmarker.task` file in the repo to enable it. If the task file is missing, the scaffold falls back to a simple OpenCV contour heuristic.
- Use a TensorFlow model for live inference when a saved model is available.
- Smooth predictions across a short history of frames to reduce jitter.

## Next improvements

- Collect webcam-style validation samples for better robustness.
- Compare crop-based inference with landmark-based models.
- Export smaller and faster model formats later if needed.
- Consider a browser-based demo after the Python scaffold is stable.

## Notebook split suggestion

- 02_cnn_static_images.ipynb
- 03_landmark_model.ipynb
- 04_live_inference_demo.ipynb
