# ASL Alphabet Recognition Using Convolutional Neural Networks

## Overview

This repository started as a static image-classification project for American Sign Language (ASL) alphabet recognition, and now includes a working realtime webcam recognizer.

The realtime path is **skeleton-based** rather than photo-based. Each frame, MediaPipe detects the 21 hand landmarks; we draw them as a colored skeleton on a black canvas (MediaPipe's default style) and classify *that* image with a small CNN. Because the skeleton discards skin tone, lighting, and background, the model generalizes to a live webcam far better than one trained on raw photos.

## Dataset

The realtime classifier is trained on the [ASL Alphabet Wireframes dataset](https://www.kaggle.com/datasets/dylanpallickara129/asl-alphabet-wireframes) — ~99,000 MediaPipe hand-skeleton images across the **24 static ASL letters** (J and Z are motion signs and are excluded). It is downloaded automatically via `kagglehub` (no Kaggle login required for this public dataset).

The original notebooks explore the Sign-MNIST dataset (28×28 grayscale) for static-image CNN baselines; that work is independent of the realtime pipeline.

## Tech Stack

- Deep learning: TensorFlow / Keras
- Realtime vision: OpenCV, MediaPipe
- Data processing: NumPy, Pandas
- Visualization: Matplotlib, Seaborn
- Evaluation: scikit-learn
- Environment: Jupyter notebooks

## Project Structure

```text
CSCI323_ASL_Recognition/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   └── realtime-plan.md
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py          # shared settings (image size, paths, classes)
│   ├── hand_tracking.py   # MediaPipe Tasks HandLandmarker wrapper
│   ├── skeleton.py        # render landmarks as a MediaPipe-style skeleton
│   ├── preprocessing.py   # crop-to-content + resize for the classifier
│   └── live_inference.py  # webcam loop: detect -> render -> classify
├── scripts/
│   ├── prep_wireframes.py # one-time: build the processed training cache
│   ├── train_asl.py       # train the skeleton CNN
│   └── verify_inference.py# held-out accuracy check
└── results/
    ├── figures/
    ├── metrics/
    └── models/            # best_model.h5 + class_names.json (written by training)
```

## Setup

### Prerequisites

- Python 3.11 (MediaPipe/TensorFlow wheels; the repo ships a `.venv311`)
- pip

### Local

```bash
git clone https://github.com/mwlde/CSCI323_ASL_Recognition.git
cd CSCI323_ASL_Recognition
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
```

## Realtime pipeline

Each webcam frame flows through:

1. **OpenCV** captures and displays the frame.
2. **MediaPipe Tasks HandLandmarker** detects the 21 hand landmarks (`src/hand_tracking.py`).
3. **`src/skeleton.py`** renders those landmarks as a colored skeleton on black, normalized (crop-to-content + square-pad) so position and scale don't matter.
4. **A small Keras CNN** classifies the skeleton into one of the 24 letters.
5. A short prediction-history buffer smooths the label across frames.

A right-hand sidebar shows the detected letter in large type, a confidence bar, and the exact skeleton ("model input") being fed to the model, so you can see what it "sees".

> Uses the modern MediaPipe **Tasks** API (`mediapipe >= 0.10`). The legacy `mp.solutions.Hands` API is not used.

## Environment

Use the Python 3.11 virtualenv (`.venv311`), which has TensorFlow, OpenCV, and MediaPipe installed:

```bash
.venv311/bin/pip install -r requirements.txt   # first-time setup
```

The MediaPipe hand landmarker model lives at `models/hand_landmarker.task` (already included).

## Training the model

```bash
# 1. Download + preprocess the wireframe dataset into a local cache (one-time)
.venv311/bin/python -m scripts.prep_wireframes

# 2. Train the skeleton classifier -> results/models/best_model.h5 (+ class_names.json)
.venv311/bin/python -m scripts.train_asl

# 3. (optional) Check held-out accuracy
.venv311/bin/python -m scripts.verify_inference
```

The included model reaches ~97% validation accuracy (~92% on the held-out inference check). The most-confused letters are M/N/T, which differ only by thumb placement.

## Running the webcam demo

```bash
.venv311/bin/python -m src.live_inference
```

Press `q` to quit. If `results/models/best_model.h5` is missing, the window still opens and shows hand tracking with a warning overlay, so you can develop without a trained model.
