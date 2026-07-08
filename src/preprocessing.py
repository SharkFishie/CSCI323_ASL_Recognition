"""Prepare a rendered hand skeleton for the classifier.

Training images and live renders go through the *same* steps so the model sees
identical inputs: crop the skeleton to its content, square-pad, resize to the
model's input size, and scale to [0, 1].
"""

import cv2
import numpy as np

from .config import IMAGE_SIZE
from .skeleton import crop_to_content


def preprocess_skeleton(skeleton_rgb, image_size=IMAGE_SIZE):
    """Normalize an RGB skeleton image into a batch-ready tensor.

    Returns shape (1, IMAGE_SIZE, IMAGE_SIZE, 3) with raw [0, 255] values, or
    None if empty. The model applies its own Rescaling(1/255) layer, so we must
    NOT divide here (matching how image_dataset_from_directory feeds training).
    """
    if skeleton_rgb is None or skeleton_rgb.size == 0:
        return None

    cropped = crop_to_content(skeleton_rgb)
    resized = cv2.resize(cropped, (image_size, image_size)).astype(np.float32)
    return np.expand_dims(resized, axis=0)
