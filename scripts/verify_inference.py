"""Smoke-test the trained skeleton classifier end to end.

Runs held-out dataset skeletons through the exact inference preprocessing and
reports accuracy, so we know the model + preprocessing agree before pointing a
webcam at it.

    .venv311/bin/python -m scripts.verify_inference
"""

import glob
import os
import random

import numpy as np
from PIL import Image

from src.config import PROCESSED_DIR
from src.live_inference import load_class_names, load_classifier
from src.preprocessing import preprocess_skeleton


def main():
    model = load_classifier()
    class_names = load_class_names()
    if model is None:
        raise SystemExit("No trained model found. Run scripts.train_asl first.")

    random.seed(1)
    correct = total = 0
    per_class_wrong = {}
    for cls in class_names:
        files = glob.glob(os.path.join(PROCESSED_DIR, cls, "*.png"))
        for f in random.sample(files, min(40, len(files))):
            img = np.array(Image.open(f).convert("RGB"))
            batch = preprocess_skeleton(img)
            pred = class_names[int(np.argmax(model.predict(batch, verbose=0)[0]))]
            total += 1
            if pred == cls:
                correct += 1
            else:
                per_class_wrong[cls] = per_class_wrong.get(cls, 0) + 1

    print(f"\nHeld-out accuracy: {correct}/{total} = {correct / total:.1%}")
    if per_class_wrong:
        worst = sorted(per_class_wrong.items(), key=lambda t: -t[1])[:6]
        print("Most-confused letters:", worst)


if __name__ == "__main__":
    main()
