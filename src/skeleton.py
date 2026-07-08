"""Render a hand skeleton in MediaPipe's default drawing style.

The `asl-alphabet-wireframes` dataset is made of MediaPipe hand-landmark
skeletons drawn on a black background. To classify a live webcam hand with a
model trained on that dataset, we must draw the detected landmarks in the *same*
style, then normalize position/scale identically for training and inference.

Colors below (RGB) were reverse-engineered from the dataset and match
MediaPipe's default hand landmark/connection styling.
"""

import cv2
import numpy as np

# --- MediaPipe default style colors (RGB) ---
_RED = (255, 48, 48)       # palm landmarks
_PEACH = (255, 229, 180)   # thumb
_PURPLE = (128, 64, 128)   # index
_YELLOW = (255, 204, 0)    # middle
_GREEN = (48, 255, 48)     # ring
_BLUE = (21, 101, 192)     # pinky
_GRAY = (128, 128, 128)    # palm connections
_WHITE = (255, 255, 255)

# Landmark index -> dot color
_PALM_LANDMARKS = (0, 1, 5, 9, 13, 17)
_FINGER_LANDMARKS = {
    _PEACH: (2, 3, 4),
    _PURPLE: (6, 7, 8),
    _YELLOW: (10, 11, 12),
    _GREEN: (14, 15, 16),
    _BLUE: (18, 19, 20),
}

# Connection (start, end) grouped by drawing color
_PALM_CONNECTIONS = ((0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
_FINGER_CONNECTIONS = {
    _PEACH: ((1, 2), (2, 3), (3, 4)),
    _PURPLE: ((5, 6), (6, 7), (7, 8)),
    _YELLOW: ((9, 10), (10, 11), (11, 12)),
    _GREEN: ((13, 14), (14, 15), (15, 16)),
    _BLUE: ((17, 18), (18, 19), (19, 20)),
}


def _landmark_color(idx):
    if idx in _PALM_LANDMARKS:
        return _RED
    for color, idxs in _FINGER_LANDMARKS.items():
        if idx in idxs:
            return color
    return _RED


def render_skeleton(landmarks, size=400, margin=0.15):
    """Draw the 21 hand landmarks as a MediaPipe-style skeleton on black.

    `landmarks` is a list of 21 objects exposing normalized `.x`/`.y` (the
    Tasks-API hand). Landmarks are re-normalized to their own bounding box so
    the skeleton fills a `margin`-padded square canvas, matching the crop-to-
    content normalization used at train time. Returns an RGB uint8 image.
    """
    xs = np.array([lm.x for lm in landmarks], dtype=np.float32)
    ys = np.array([lm.y for lm in landmarks], dtype=np.float32)

    # Fit the hand into a centered square that preserves aspect ratio.
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
    usable = size * (1.0 - 2 * margin)
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    px = ((xs - cx) / span * usable + size / 2.0).astype(int)
    py = ((ys - cy) / span * usable + size / 2.0).astype(int)
    pts = list(zip(px.tolist(), py.tolist()))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # Line thickness / dot radius scaled to canvas size (tuned to dataset look).
    palm_th = max(2, round(size * 0.0075))
    finger_th = max(1, round(size * 0.005))
    radius = max(2, round(size * 0.011))

    # The canvas is kept in RGB order; cv2 just writes the raw byte tuples we
    # pass, and everything downstream (PIL, TF, inference) reads it as RGB.
    def line(a, b, color, th):
        cv2.line(canvas, pts[a], pts[b], color, th, cv2.LINE_AA)

    for a, b in _PALM_CONNECTIONS:
        line(a, b, _GRAY, palm_th)
    for color, conns in _FINGER_CONNECTIONS.items():
        for a, b in conns:
            line(a, b, color, finger_th)

    for idx, p in enumerate(pts):
        color = _landmark_color(idx)
        cv2.circle(canvas, p, radius + 1, _WHITE, -1, cv2.LINE_AA)
        cv2.circle(canvas, p, radius, color, -1, cv2.LINE_AA)

    return canvas


def crop_to_content(image, pad_frac=0.08):
    """Tight-crop non-black content and square-pad it.

    Applied identically to dataset images and live renders so the model never
    has to learn absolute position or scale. Returns an RGB uint8 image; if the
    frame is essentially empty it is returned unchanged.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 25)
    if len(xs) == 0:
        return image

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = image[y0:y1 + 1, x0:x1 + 1]

    h, w = crop.shape[:2]
    side = max(h, w)
    pad = int(side * pad_frac)
    canvas = np.zeros((side + 2 * pad, side + 2 * pad, 3), dtype=image.dtype)
    oy = pad + (side - h) // 2
    ox = pad + (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return canvas
