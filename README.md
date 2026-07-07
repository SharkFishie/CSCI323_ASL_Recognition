# ASL Alphabet Recognition Using Convolutional Neural Networks

## Overview

This repository started as a static image-classification project for American Sign Language (ASL) alphabet recognition. The initial work focuses on training convolutional neural networks to classify still images of hand poses for the 29 classes in the Kaggle ASL Alphabet dataset.

The project is now being extended with a first-pass realtime webcam inference scaffold. This new code is intentionally simple and modular: it is not a claim that live ASL recognition is fully solved yet. Instead, it provides a beginner-friendly starting point for webcam capture, hand tracking, and live inference with a saved model.

## Dataset

The project uses the [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The dataset contains roughly 87,000 images of size 200x200 across 29 classes: A through Z, plus SPACE, DELETE, and NOTHING.

This dataset is suitable for baseline static-image training. For realtime webcam use, later work may need webcam-style samples or a landmark-based dataset to improve robustness under different lighting, hand positions, and backgrounds.

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
│   ├── config.py
│   ├── hand_tracking.py
│   ├── preprocessing.py
│   └── live_inference.py
└── results/
    ├── figures/
    ├── metrics/
    └── models/
```

## Setup

### Prerequisites

- Python 3.8+
- pip

### Local

```bash
git clone https://github.com/mwlde/CSCI323_ASL_Recognition.git
cd CSCI323_ASL_Recognition
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Approach

The original workflow is still centered on static image classification with CNNs. The repository currently supports training and evaluating models on still images, while the new scaffold adds a simple path toward realtime inference.

The live webcam scaffold uses:

- OpenCV for webcam capture and display
- MediaPipe Hands for hand detection and landmark tracking
- TensorFlow / Keras for loading a trained model later placed at results/models/best_model.h5
- A short prediction-history buffer to smooth outputs across frames

This is a first-pass scaffold, not a finished realtime recognition system.

## Running the webcam scaffold

Once a trained model is available at results/models/best_model.h5, run:

```bash
python -m src.live_inference
```

If the model file is missing, the webcam window will still open and show a warning overlay so the scaffold can be tested and developed further.

MediaPipe Tasks requirement

The realtime scaffold uses the MediaPipe Tasks Hand Landmarker for improved
hand detection when available. Place a Hand Landmarker task file at
`models/hand_landmarker.task` before running live inference. If the file is
missing or MediaPipe Tasks is not available the scaffold will fall back to a
simple OpenCV contour heuristic, but results will be less reliable.

Download the official MediaPipe hand landmarker task file from the MediaPipe
project and place it under `models/hand_landmarker.task` to enable the Tasks
detector.
