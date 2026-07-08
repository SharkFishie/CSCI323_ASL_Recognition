# ASL Alphabet Recognition Using Convolutional Neural Networks

Realtime American Sign Language (ASL) fingerspelling recognition from a webcam,
plus the CNN experiments it grew out of.

The realtime recognizer is **skeleton-based**. Each frame, MediaPipe detects the
21 hand landmarks; those landmarks are drawn as a colored skeleton on a black
canvas (MediaPipe's default style) and a small CNN classifies *that* image.
Because the skeleton throws away skin tone, lighting, and background, the model
generalizes to a live webcam far better than one trained on raw photos.

## Quick start

```bash
# one-time setup (Python 3.11)
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# run the webcam demo (train the model first if it's missing — see Training)
.venv311/bin/python -m src.live_inference
```

Press `q` to quit. Sign a static letter and the sidebar shows the prediction.
If `results/models/best_model.h5` is missing, the window still opens with hand
tracking and a warning overlay, so you can develop without a trained model.

## How it works

Each webcam frame flows through:

1. **OpenCV** captures and displays the frame.
2. **MediaPipe Tasks HandLandmarker** detects the 21 hand landmarks — `src/hand_tracking.py`.
3. **`src/skeleton.py`** renders those landmarks as a colored skeleton on black, then normalizes it (crop-to-content + square-pad) so hand position and scale don't matter.
4. **A small Keras CNN** classifies the skeleton into one of the 24 letters.
5. A short prediction-history buffer smooths the label across frames.

A right-hand **sidebar** shows the detected letter in large type, a confidence
bar, and the exact skeleton ("model input") being fed to the model — so you can
see what it sees.

> Uses the modern MediaPipe **Tasks** API (`mediapipe >= 0.10`); the legacy
> `mp.solutions.Hands` API is not used.

## Dataset

The classifier is trained on the
[ASL Alphabet Wireframes dataset](https://www.kaggle.com/datasets/dylanpallickara129/asl-alphabet-wireframes)
— ~99,000 MediaPipe hand-skeleton images across the **24 static ASL letters**
(J and Z are motion signs and are excluded). It downloads automatically via
`kagglehub`; no Kaggle login is needed for this public dataset.

The `notebooks/` explore the separate Sign-MNIST dataset (28×28 grayscale) for
static-image CNN baselines; that work is independent of the realtime pipeline.

## Project structure

```text
CSCI323_ASL_Recognition/
├── requirements.txt
├── models/                     # hand_landmarker.task (downloaded, gitignored)
├── data/                       # datasets (large files gitignored)
├── docs/
├── notebooks/                  # Sign-MNIST CNN experiments
├── src/
│   ├── config.py               # shared settings (image size, paths, classes)
│   ├── hand_tracking.py        # MediaPipe Tasks HandLandmarker wrapper
│   ├── skeleton.py             # render landmarks as a MediaPipe-style skeleton
│   ├── preprocessing.py        # crop-to-content + resize for the classifier
│   └── live_inference.py       # webcam loop: detect -> render -> classify
├── scripts/
│   ├── prep_wireframes.py      # one-time: build the processed training cache
│   ├── train_asl.py            # train the skeleton CNN
│   └── verify_inference.py     # held-out accuracy check
└── results/
    └── models/                 # best_model.h5 + class_names.json (from training)
```

## Setup

**Prerequisites:** Python 3.11 (required for the MediaPipe/TensorFlow wheels).

```bash
git clone https://github.com/mwlde/CSCI323_ASL_Recognition.git
cd CSCI323_ASL_Recognition
python3.11 -m venv .venv311
.venv311/bin/pip install -r requirements.txt

# download the MediaPipe hand landmarker model (gitignored, ~7 MB)
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Training

The trained model (`results/models/best_model.h5`) is gitignored, so build it
once after cloning:

```bash
# 1. Download + preprocess the wireframe dataset into a local cache (one-time)
.venv311/bin/python -m scripts.prep_wireframes

# 2. Train the classifier -> results/models/best_model.h5 (+ class_names.json)
.venv311/bin/python -m scripts.train_asl

# 3. (optional) Check held-out accuracy through the inference preprocessing
.venv311/bin/python -m scripts.verify_inference
```

The model reaches **~97% validation accuracy** (~92% on the held-out inference
check). The most-confused letters are M/N/T, which differ only by thumb
placement. Key knobs live in `src/config.py` (`IMAGE_SIZE`, `MAX_PER_CLASS`,
`EPOCHS`, `BATCH_SIZE`).

## Tech stack

TensorFlow / Keras · OpenCV · MediaPipe · NumPy · scikit-learn · kagglehub
