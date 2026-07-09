"""Sanity-check the reference poses + similarity metric.

For held-out photos (a different sample than the builder used), detect the
hand and classify it as the geometrically nearest reference pose. High
accuracy means the similarity metric carries real signal, i.e. the tutor's
graded feedback bar is meaningful — the CNN remains the pass/fail authority.
Run from the repo root:

    .venv311/bin/python -m scripts.verify_reference_poses
"""

import glob
import os
import random

import cv2
import numpy as np

import mediapipe as mp

from src.config import CLASS_NAMES, REFERENCE_DATASET
from src.tutor.reference import ReferenceLibrary
from scripts.build_reference_poses import find_class_root, make_detector

import kagglehub

SAMPLES_PER_CLASS = 60
# Builder shuffles with seed 323 and takes the first 300; a different seed
# gives an (almost surely) different sample, and exactness doesn't matter for
# a sanity check.
SEED = 424


def main():
    library = ReferenceLibrary.load()
    if library is None:
        raise SystemExit("No reference poses. Run scripts.build_reference_poses first.")

    random.seed(SEED)
    class_root = find_class_root(kagglehub.dataset_download(REFERENCE_DATASET))
    detector = make_detector()

    correct = total = 0
    scores_true, scores_other = [], []
    confusions = {}
    for letter in library.letters:
        class_dir = os.path.join(class_root, letter)
        if not os.path.isdir(class_dir):
            class_dir = os.path.join(class_root, letter.lower())
        files = sorted(glob.glob(os.path.join(class_dir, "*")))
        random.shuffle(files)
        for f in files[:SAMPLES_PER_CLASS]:
            img = cv2.imread(f)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                continue
            hand = res.hand_landmarks[0]
            pred, _ = library.best_match(hand)
            true_score, _, _ = library.score(hand, letter)
            scores_true.append(true_score)
            scores_other.extend(
                library.score(hand, o)[0]
                for o in random.sample(
                    [l for l in library.letters if l != letter], 3))
            total += 1
            if pred == letter:
                correct += 1
            else:
                confusions[(letter, pred)] = confusions.get((letter, pred), 0) + 1

    print(f"\nNearest-reference accuracy: {correct}/{total} = {correct/total:.1%}")
    print(f"Similarity score, correct letter : mean {np.mean(scores_true):.2f}")
    print(f"Similarity score, wrong letters  : mean {np.mean(scores_other):.2f}")
    if confusions:
        worst = sorted(confusions.items(), key=lambda t: -t[1])[:6]
        print("Top confusions:", [f"{a}->{b} x{n}" for (a, b), n in worst])


if __name__ == "__main__":
    main()
