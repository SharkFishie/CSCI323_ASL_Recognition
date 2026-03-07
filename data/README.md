# Dataset: Sign Language MNIST

This directory stores the Sign Language MNIST dataset used for training and evaluation. The dataset is **not included** in this repository — download it and place the CSV files directly in this `data/` folder.

## Download Instructions

### Option 1: Kaggle Website

1. Visit the dataset page: **[Sign Language MNIST — Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)**
2. Click the **Download** button (you will need a free Kaggle account)
3. Extract and place `sign_mnist_train.csv` and `sign_mnist_test.csv` into this `data/` directory

### Option 2: Kaggle CLI

```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Set up API credentials (~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/settings → API → Create New Token

# Download the dataset
kaggle datasets download -d datamunge/sign-language-mnist -p data/

# Unzip
unzip data/sign-language-mnist.zip -d data/
```

### Option 3: Google Colab

```python
# Upload your kaggle.json or authenticate
from google.colab import files
files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and extract
!kaggle datasets download -d datamunge/sign-language-mnist -p data/
!unzip data/sign-language-mnist.zip -d data/
```

---

## Expected Files After Extraction

```
data/
├── README.md               # This file
├── sign_mnist_train.csv    # Training set (27,455 rows)
└── sign_mnist_test.csv     # Test set (7,172 rows)
```

Each CSV has the following structure:
- Column 1: `label` — integer class label (0–24, with 9 skipped)
- Columns 2–785: `pixel1` to `pixel784` — flattened 28×28 grayscale image, values 0–255

---

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Classes** | 24 (A–Y, excluding J and Z) |
| **Training Images** | 27,455 |
| **Test Images** | 7,172 |
| **Image Size** | 28 × 28 pixels |
| **Color Format** | Grayscale |
| **File Format** | CSV |
| **Total Size** | ~5 MB |

---

## Why Only 24 Classes?

The letters **J** and **Z** are excluded because their signs require **motion** (a path traced through the air) and cannot be represented as a single static image. All other letters A–Y are included.

## Label Encoding

Labels are integers that map directly to letters, but **label 9 is skipped** (would have been J):

| Label | Letter | Label | Letter |
|-------|--------|-------|--------|
| 0 | A | 13 | N |
| 1 | B | 14 | O |
| 2 | C | 15 | P |
| 3 | D | 16 | Q |
| 4 | E | 17 | R |
| 5 | F | 18 | S |
| 6 | G | 19 | T |
| 7 | H | 20 | U |
| 8 | I | 21 | V |
| ~~9~~ | ~~J~~ | 22 | W |
| 10 | K | 23 | X |
| 11 | L | 24 | Y |
| 12 | M | — | — |

---

## Important Notes

- The dataset files are listed in `.gitignore` and will **not** be tracked by Git
- Each team member must download the dataset independently
- Ensure the CSV files are placed directly in `data/` (not in a subfolder) — notebook paths expect `../data/sign_mnist_train.csv`
- On Google Colab, the dataset must be re-downloaded each session (or stored on Google Drive)
