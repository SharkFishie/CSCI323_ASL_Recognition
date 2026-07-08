"""One-time preprocessing of the asl-alphabet-wireframes dataset.

Reads the raw skeleton images (downloaded via kagglehub), applies the same
crop-to-content + resize used at inference, and writes them to a processed
cache that training loads directly. Run from the repo root:

    .venv311/bin/python -m scripts.prep_wireframes
"""

import glob
import os
import random

import cv2
import kagglehub
import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE, KAGGLE_DATASET, MAX_PER_CLASS, PROCESSED_DIR
from src.skeleton import crop_to_content


def find_class_root(root):
    """Return the directory that directly contains the letter subfolders."""
    candidate = root
    for _ in range(3):
        subdirs = [
            d for d in os.listdir(candidate)
            if os.path.isdir(os.path.join(candidate, d))
        ]
        if len(subdirs) >= 20:
            return candidate
        if len(subdirs) == 1:
            candidate = os.path.join(candidate, subdirs[0])
        else:
            break
    return candidate


def main():
    raw_root = kagglehub.dataset_download(KAGGLE_DATASET)
    class_root = find_class_root(raw_root)
    classes = sorted(
        d for d in os.listdir(class_root)
        if os.path.isdir(os.path.join(class_root, d))
    )
    print(f"Source: {class_root}\n{len(classes)} classes: {classes}")

    random.seed(42)
    total = 0
    for cls in classes:
        files = glob.glob(os.path.join(class_root, cls, "*.jpg"))
        if MAX_PER_CLASS and len(files) > MAX_PER_CLASS:
            files = random.sample(files, MAX_PER_CLASS)

        out_dir = os.path.join(PROCESSED_DIR, cls.upper())
        os.makedirs(out_dir, exist_ok=True)
        for i, f in enumerate(files):
            img = np.array(Image.open(f).convert("RGB"))
            proc = cv2.resize(crop_to_content(img), (IMAGE_SIZE, IMAGE_SIZE))
            Image.fromarray(proc).save(os.path.join(out_dir, f"{i:05d}.png"))
        total += len(files)
        print(f"  {cls.upper()}: {len(files)} images")

    print(f"Done. {total} images -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
