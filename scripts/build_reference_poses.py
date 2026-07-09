"""Build per-letter reference hand poses for the tutor.

Runs the MediaPipe HandLandmarker over real ASL photos (grassknoted/
asl-alphabet via kagglehub), normalizes each detection, and robust-averages
them into one canonical pose per static letter. Also prints a calibration
report (intra- vs inter-letter distances) that justifies D_MAX in
src/tutor/reference.py. Run from the repo root:

    .venv311/bin/python -m scripts.build_reference_poses

Writes REFERENCE_POSES_PATH (small JSON, committed) and a grid figure of all
reference skeletons to results/figures/reference_poses.png.
"""

import glob
import json
import os
import random
from types import SimpleNamespace

import cv2
import kagglehub
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.config import (
    CLASS_NAMES,
    HAND_LANDMARKER_MODEL_PATH,
    REFERENCE_DATASET,
    REFERENCE_POSES_PATH,
)
from src.skeleton import render_skeleton
from src.tutor.reference import normalize_pose, per_joint_distances

SAMPLES_PER_CLASS = 300   # photos attempted per letter
KEEP_FRACTION = 0.6       # keep detections closest to the class median
FIGURE_PATH = "results/figures/reference_poses.png"


def make_detector():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=HAND_LANDMARKER_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def find_class_root(root):
    """Find the directory that directly contains the letter folders.

    Breadth-first so sibling dirs (e.g. asl_alphabet_test next to
    asl_alphabet_train) don't send us down the wrong branch.
    """
    queue = [root]
    for _ in range(200):
        if not queue:
            break
        candidate = queue.pop(0)
        subdirs = [d for d in os.listdir(candidate)
                   if os.path.isdir(os.path.join(candidate, d))]
        if sum(d.upper() in CLASS_NAMES for d in subdirs) >= 20:
            return candidate
        queue.extend(os.path.join(candidate, d) for d in sorted(subdirs))
    raise SystemExit(f"Could not find letter folders under {root}")


def collect_poses(detector, class_dir):
    """Detect and normalize hand poses for one letter's photos."""
    files = sorted(glob.glob(os.path.join(class_dir, "*")))
    random.shuffle(files)
    poses = []
    for f in files[:SAMPLES_PER_CLASS]:
        img = cv2.imread(f)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            continue
        pts = np.array([[lm.x, lm.y] for lm in res.hand_landmarks[0]],
                       dtype=np.float32)
        poses.append(normalize_pose(pts))
    return poses


def robust_average(poses):
    """Median -> trim outliers -> mean. Returns (pose, spread, kept)."""
    stack = np.stack(poses)                      # (n, 21, 2)
    median = np.median(stack, axis=0)
    # Distance of each sample to the median pose (mirror-tolerant).
    dists = np.array([per_joint_distances(p, median)[0].mean() for p in stack])
    keep = np.argsort(dists)[: max(1, int(len(poses) * KEEP_FRACTION))]
    kept = stack[keep]
    pose = normalize_pose(kept.mean(axis=0))
    spread = float(np.mean([per_joint_distances(p, pose)[0].mean() for p in kept]))
    return pose, spread, len(keep)


def save_grid_figure(references, path=FIGURE_PATH):
    """Render every reference pose as a labeled skeleton grid."""
    letters = sorted(references)
    cols, cell = 6, 220
    rows = int(np.ceil(len(letters) / cols))
    canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for i, letter in enumerate(letters):
        pose = np.asarray(references[letter]["pose"], dtype=np.float32)
        # render_skeleton re-normalizes to its own bbox, so raw coords are fine.
        lms = [SimpleNamespace(x=float(x), y=float(y)) for x, y in pose]
        tile = render_skeleton(lms, size=cell)
        cv2.putText(tile, letter, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 2, cv2.LINE_AA)
        r, c = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"Reference grid figure -> {path}")


def calibration_report(references):
    """Print intra-class spread vs inter-class distances (justifies D_MAX)."""
    letters = sorted(references)
    spreads = [references[l]["spread"] for l in letters]
    inter = []
    closest = {}
    for i, a in enumerate(letters):
        pa = np.asarray(references[a]["pose"], dtype=np.float32)
        best = (np.inf, None)
        for b in letters:
            if a == b:
                continue
            pb = np.asarray(references[b]["pose"], dtype=np.float32)
            d = per_joint_distances(pa, pb)[0].mean()
            inter.append(d)
            if d < best[0]:
                best = (d, b)
        closest[a] = best
    print(f"\nIntra-letter spread : mean {np.mean(spreads):.3f}  "
          f"max {np.max(spreads):.3f}")
    print(f"Inter-letter dist   : mean {np.mean(inter):.3f}  "
          f"min {np.min(inter):.3f}")
    tight = sorted(closest.items(), key=lambda t: t[1][0])[:5]
    print("Most-confusable reference pairs:",
          [f"{a}~{b} ({d:.3f})" for a, (d, b) in tight])


def main():
    random.seed(323)
    raw_root = kagglehub.dataset_download(REFERENCE_DATASET)
    class_root = find_class_root(raw_root)
    print(f"Photo dataset: {class_root}")

    detector = make_detector()
    dirs = {d.upper(): os.path.join(class_root, d)
            for d in os.listdir(class_root)
            if os.path.isdir(os.path.join(class_root, d))}

    references = {}
    for letter in CLASS_NAMES:
        if letter not in dirs:
            print(f"{letter}: no photo folder, skipped")
            continue
        poses = collect_poses(detector, dirs[letter])
        if len(poses) < 20:
            print(f"{letter}: only {len(poses)} detections, skipped")
            continue
        pose, spread, kept = robust_average(poses)
        references[letter] = {
            "pose": pose.tolist(),
            "spread": round(spread, 4),
            "detections": len(poses),
            "kept": kept,
        }
        print(f"{letter}: {len(poses)}/{SAMPLES_PER_CLASS} detected, "
              f"kept {kept}, spread {spread:.3f}")

    os.makedirs(os.path.dirname(REFERENCE_POSES_PATH), exist_ok=True)
    with open(REFERENCE_POSES_PATH, "w") as fh:
        json.dump(references, fh)
    print(f"\n{len(references)} reference poses -> {REFERENCE_POSES_PATH}")

    calibration_report(references)
    save_grid_figure(references)


if __name__ == "__main__":
    main()
