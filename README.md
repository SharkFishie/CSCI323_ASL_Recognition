# ASL Alphabet Recognition Using Convolutional Neural Networks

> **CSCI323 Modern Artificial Intelligence**  
> University of Wollongong in Dubai (UOWD)  
> Spring 2026

---

## Overview

This project explores American Sign Language (ASL) alphabet recognition using Convolutional Neural Networks (CNNs). The goal is to classify static hand gesture images corresponding to the 26 letters of the ASL alphabet, with potential applications in assistive communication technology.

The project investigates multiple CNN architectures — from a simple baseline to deeper custom networks and transfer learning approaches (VGG16/ResNet) — to evaluate accuracy and generalisation on ASL hand gesture datasets.

---

## Dataset

The dataset is **not included** in this repository due to its size.

The project uses the [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), which contains 87,000 images (200×200 px) across 29 classes (A–Z + SPACE, DELETE, NOTHING).

---

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** NumPy, Pandas
- **Visualisation:** Matplotlib, Seaborn
- **Evaluation:** scikit-learn (classification report, confusion matrix)
- **Environment:** Jupyter Notebook / Google Colab (GPU)

---

## Project Structure

```
CSCI323_ASL_Recognition/
├── README.md                  # Project documentation
├── .gitignore
├── requirements.txt           # Python dependencies
│
├── data/                      # Dataset directory (not tracked)
│   └── README.md              # Dataset download instructions
│
├── notebooks/                 # Jupyter / Colab notebooks
│
├── src/                       # Source code modules
│   └── __init__.py
│
└── results/                   # Experiment outputs
    ├── figures/               # Plots and confusion matrices
    ├── metrics/               # Evaluation metrics (CSV/JSON)
    └── models/                # Saved model weights (excluded from git)
```

---

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

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/mwlde/CSCI323_ASL_Recognition.git
%cd CSCI323_ASL_Recognition
!pip install -r requirements.txt
```

---

## Approach

Three model configurations were explored:

| Model | Description |
|-------|-------------|
| Baseline CNN | Simple 3-layer CNN for initial benchmarking |
| Custom CNN | Deeper architecture with batch normalisation and dropout |
| Transfer Learning | Fine-tuned VGG16 / ResNet50 with frozen base layers |

Evaluation metrics: accuracy, precision, recall, F1 score, and confusion matrix across all 29 classes.
