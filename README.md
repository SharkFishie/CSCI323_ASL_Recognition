# ASL Alphabet Recognition Using Convolutional Neural Networks


> **CSCI323 Modern Artificial Intelligence**  
> University of Wollongong in Dubai (UOWD)  
> Spring 2026

---

## Team Members (to be added)

| 1 | [Team Member 1] [ID]
| 2 | [Team Member 2] [ID] 
| 3 | [Team Member 3] [ID]
| 4 | [Team Member 4] [ID]
| 5 | [Team Member 5] [ID]

---

## Project Overview (to be added)

### The Problem


### Our Solution


### Target Performance


### Business Value for UAE Organizations (to be added)


---

## Dataset (to be added)

> **Note:** The dataset is **not included** in this repository due to its size. See [`data/README.md`](data/README.md) for download instructions.

---

## Technologies (to be added)


---

## Project Structure

```
CSCI323_ASL_Recognition/
├── README.md                  # Project documentation (this file)
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
│
├── data/                      # Dataset directory
│   └── README.md              # Dataset download instructions
│
├── notebooks/                 # Jupyter / Colab notebooks
│   └── .gitkeep
│
├── src/                       # Source code modules
│   ├── __init__.py
│   └── .gitkeep
│
├── results/                   # Experiment outputs
│   ├── figures/               # Plots, confusion matrices, charts
│   ├── metrics/               # Saved evaluation metrics (CSV/JSON)
│   └── models/                # Saved model weights (excluded from git)
│
├── docs/                      # Additional documentation
│   └── .gitkeep
│
└── presentation/              # Final presentation materials
    └── .gitkeep
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Google Colab account for GPU access

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/SharkFishie/CSCI323_ASL_Recognition.git
cd CSCI323_ASL_Recognition

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (see data/README.md for details)
```

### Google Colab Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repo
!git clone https://github.com/SharkFishie/CSCI323_ASL_Recognition.git
%cd CSCI323_ASL_Recognition

# Install requirements
!pip install -r requirements.txt
```

---

## Usage

### Running Notebooks in Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Select **File → Open Notebook → GitHub**
3. Enter the repository URL: `https://github.com/SharkFishie/CSCI323_ASL_Recognition`
4. Choose the desired notebook from the `notebooks/` folder
5. Ensure the runtime is set to **GPU** (`Runtime → Change runtime type → T4 GPU`)
6. Run all cells sequentially

### Running Locally

```bash
# Launch Jupyter Notebook
jupyter notebook

# Navigate to notebooks/ and open the desired .ipynb file
```

---

## Results (to be added)

> _Results will be updated as experiments are completed._

| Model | Accuracy | Precision | Recall | F1 Score |
| Baseline CNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Custom CNN (Deep) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Transfer Learning (VGG16/ResNet) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Best Model | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## Team Contributions (to be added)

| Role | Responsibilities | Assigned To |
|------|-----------------|-------------|
| **Project Lead + Baseline Model** | Repository setup, baseline CNN architecture, team coordination | _[Member 1]_ |
| **Data Pipeline + Custom CNN** | Data loading, preprocessing, augmentation, custom CNN design | _[Member 2]_ |
| **Transfer Learning + Research** | VGG16/ResNet fine-tuning, literature review, architecture comparison | _[Member 3]_ |
| **Evaluation + Model Comparison** | Metrics computation, confusion matrices, performance analysis | _[Member 4]_ |
| **Demo + Deployment** | Live demo notebook, Gradio/Streamlit interface, presentation slides | _[Member 5]_ |

---

## References (to be added)

1.
---
