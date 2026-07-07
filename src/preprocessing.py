"""Helpers for preparing a cropped hand image for inference."""

import cv2
import numpy as np


def preprocess_crop(crop, image_size=(200, 200)):
    """Resize, normalize, and expand a crop into a batch-ready tensor."""
    if crop is None or crop.size == 0:
        return None

    resized = cv2.resize(crop, image_size)
    resized = resized.astype(np.float32) / 255.0
    return np.expand_dims(resized, axis=0)
