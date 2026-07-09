"""Per-letter reference hand poses and geometric similarity scoring.

The tutor scores a learner's hand with two signals: the CNN classifier decides
pass/fail, and the *geometric* distance between the live landmarks and a stored
per-letter reference pose gives graded, per-joint feedback ("your ring finger
is off"). This module owns the geometry half.

Poses are 21 MediaPipe landmarks reduced to 2D and normalized so the
comparison ignores where the hand is and how big it is — but NOT how it is
rotated, because orientation is part of a sign (e.g. G and H point sideways).

Reference poses are built offline by scripts/build_reference_poses.py and
loaded from REFERENCE_POSES_PATH.
"""

import json
import os

import numpy as np

from ..config import REFERENCE_POSES_PATH

# A normalized pose has RMS landmark distance 1.0 from the wrist. Measured on
# the reference dataset (scripts/build_reference_poses.py calibration report +
# scripts/verify_reference_poses.py): same-letter mean per-joint distance is
# ~0.10 (intra spread), different letters average ~0.67, and nearest-reference
# classification hits 81% over 24 classes. The score is therefore 1.0 for any
# distance within intra-class noise (D_PERFECT) falling linearly to 0.0 at
# D_MAX, which puts a correct sign near the top of the bar and a wrong letter
# near the bottom.
D_PERFECT = 0.10
D_MAX = 0.50

# Landmarks whose error matters most for feedback grouping.
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def landmarks_to_array(landmarks):
    """Convert a Tasks-API landmark list (or (21, 2) sequence) to float array."""
    if hasattr(landmarks[0], "x"):
        return np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    return np.asarray(landmarks, dtype=np.float32)


def normalize_pose(points):
    """Wrist-center and scale-normalize a (21, 2) landmark array.

    Translation: subtract the wrist (landmark 0). Scale: divide by the RMS
    distance of all landmarks from the wrist, so every pose has the same
    "size" regardless of hand size or distance to the camera. Rotation is
    deliberately preserved.
    """
    pts = landmarks_to_array(points)
    centered = pts - pts[0]
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if scale < 1e-6:
        return centered
    return centered / scale


def per_joint_distances(live, reference):
    """Per-landmark distances between two poses, mirror-tolerant.

    A left and a right hand make the same sign mirrored, and the webcam feed
    is flipped, so we compare the live pose both as-is and x-mirrored and keep
    whichever fits better. Returns (distances (21,), mirrored: bool) with the
    inputs normalized internally.
    """
    live_n = normalize_pose(live)
    ref_n = normalize_pose(reference)

    direct = np.linalg.norm(live_n - ref_n, axis=1)
    mirrored_live = live_n * np.array([-1.0, 1.0], dtype=np.float32)
    mirrored = np.linalg.norm(mirrored_live - ref_n, axis=1)

    if mirrored.mean() < direct.mean():
        return mirrored, True
    return direct, False


def similarity(live, reference):
    """Graded closeness of a live pose to a reference pose.

    Returns (score, joint_distances, mirrored) where score is 1.0 for any
    mean per-joint distance within intra-class noise (D_PERFECT), falling
    linearly to 0.0 at D_MAX. joint_distances is the (21,) array for
    per-finger feedback.
    """
    dists, mirrored = per_joint_distances(live, reference)
    score = float(np.clip(
        1.0 - (dists.mean() - D_PERFECT) / (D_MAX - D_PERFECT), 0.0, 1.0))
    return score, dists, mirrored


def worst_fingers(joint_distances, top=2):
    """Name the fingers with the largest mean error, worst first."""
    means = {
        name: float(np.mean(joint_distances[list(idx)]))
        for name, idx in FINGER_JOINTS.items()
    }
    return sorted(means, key=means.get, reverse=True)[:top]


class ReferenceLibrary:
    """Loaded set of per-letter reference poses."""

    def __init__(self, poses):
        # letter -> normalized (21, 2) array
        self.poses = {k: normalize_pose(np.asarray(v, dtype=np.float32))
                      for k, v in poses.items()}

    @classmethod
    def load(cls, path=REFERENCE_POSES_PATH):
        """Load the library, or return None if it hasn't been built yet."""
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            data = json.load(fh)
        return cls({letter: entry["pose"] for letter, entry in data.items()})

    @property
    def letters(self):
        return sorted(self.poses)

    def score(self, live_landmarks, letter):
        """(score, joint_distances, mirrored) against one letter's reference."""
        return similarity(live_landmarks, self.poses[letter])

    def best_match(self, live_landmarks):
        """Letter whose reference is geometrically closest, with its score.

        Ranks by raw mean distance rather than the score: the score's
        "perfect" floor flattens ranking between near-identical references
        (e.g. R vs U), which would make ties arbitrary.
        """
        best_letter, best_dist = None, np.inf
        for letter, ref in self.poses.items():
            dist = per_joint_distances(live_landmarks, ref)[0].mean()
            if dist < best_dist:
                best_letter, best_dist = letter, dist
        return best_letter, float(np.clip(
            1.0 - (best_dist - D_PERFECT) / (D_MAX - D_PERFECT), 0.0, 1.0))
