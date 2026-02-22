# Dataset: ASL Alphabet

This directory is intended to store the ASL Alphabet dataset used for training and evaluation. The dataset is **not included** in this repository due to its large size (~1.1 GB).

## Download Instructions

### Option 1: Kaggle Website

1. Visit the dataset page: **[ASL Alphabet — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)**
2. Click the **Download** button (you will need a free Kaggle account)
3. Extract the downloaded ZIP file into this `data/` directory

### Option 2: Kaggle CLI

```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Set up API credentials (~/.kaggle/kaggle.json)
# Download from: https://www.kaggle.com/settings → API → Create New Token

# Download the dataset
kaggle datasets download -d grassknoted/asl-alphabet -p data/

# Unzip
unzip data/asl-alphabet.zip -d data/
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
!kaggle datasets download -d grassknoted/asl-alphabet -p data/
!unzip data/asl-alphabet.zip -d data/
```

---

## Expected Folder Structure After Extraction

```
data/
├── README.md                    # This file
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/       
│       ├── B/                
│       ├── C/                   
│       ├── ...                 
│       ├── Z/                   
│       ├── space/               
│       ├── del/                 
│       └── nothing/            
└── asl_alphabet_test/
    └── asl_alphabet_test/
        ├── A_test.jpg
        ├── B_test.jpg
        ├── ...                  # (one image per class)
        └── nothing_test.jpg
```

---

## Dataset Specifications

| Property | Value |
|----------|-------|
| **Classes** | 29 (A–Z, space, delete, nothing) |
| **Training Images** | ~87,000 (~3,000 per class) |
| **Test Images** | 29 (1 per class) |
| **Image Size** | 200 × 200 pixels |
| **Color Format** | RGB |
| **File Format** | JPG |
| **Total Size** | ~1.1 GB |

---

## Important Notes

- The dataset files are listed in `.gitignore` and will **not** be tracked by Git
- Each team member must download the dataset independently
- Ensure you have at least **2 GB** of free disk space before downloading
- On Google Colab, the dataset must be re-downloaded each session (or stored on Google Drive)
