# 🤟 ASL Alphabet Recognition Using Convolutional Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Course](https://img.shields.io/badge/Course-CSCI323-purple.svg)](https://www.uowdubai.ac.ae/)

> **CSCI323 — Modern Artificial Intelligence**  
> University of Wollongong in Dubai (UOWD)  
> Spring 2026

---

## 👥 Team Members

| # | Name | Student ID | Role |
|---|------|------------|------|
| 1 | _[Team Member 1]_ | _[ID]_ | Project Lead + Baseline Model |
| 2 | _[Team Member 2]_ | _[ID]_ | Data Pipeline + Custom CNN |
| 3 | _[Team Member 3]_ | _[ID]_ | Transfer Learning + Research |
| 4 | _[Team Member 4]_ | _[ID]_ | Evaluation + Model Comparison |
| 5 | _[Team Member 5]_ | _[ID]_ | Demo + Deployment |

---

## 📋 Project Overview

### The Problem

In the UAE, over **3,000 individuals** rely on sign language as their primary mode of communication. Despite the nation's commitment to supporting **People of Determination**, communication barriers remain significant:

- Professional sign language interpreters cost **AED 200–400 per hour**
- Interpreter availability is limited, especially outside major cities
- Real-time communication remains a daily challenge in healthcare, education, and government services

### Our Solution

We develop a **Convolutional Neural Network (CNN)** system capable of recognizing the **American Sign Language (ASL) alphabet** from static hand gesture images. The system classifies images into **29 classes**:

- **26 letters**: A–Z
- **3 special gestures**: `space`, `delete`, `nothing`

### Target Performance

| Metric | Target |
|--------|--------|
| Overall Accuracy | **> 85%** |
| Per-class Precision | **> 80%** |
| Per-class Recall | **> 80%** |
| Inference Time | **< 100ms** per image |

### Business Value for UAE Organizations

- **Healthcare**: Enable deaf/hard-of-hearing patients to communicate symptoms to medical staff
- **Education**: Support inclusive classrooms at UOWD and other UAE universities
- **Government Services**: Improve accessibility at service centres and smart kiosks
- **Cost Reduction**: Reduce dependence on costly human interpreters for basic alphabet-level communication

---

## 📊 Dataset

| Property | Details |
|----------|--------|
| **Source** | [Kaggle — ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **Total Training Images** | ~87,000 |
| **Total Test Images** | 29 (1 per class) |
| **Classes** | 29 (A–Z + space + delete + nothing) |
| **Image Dimensions** | 200 × 200 pixels |
| **Format** | RGB JPG |
| **Total Size** | ~1.1 GB |

> ⚠️ **Note:** The dataset is **not included** in this repository due to its size. See [`data/README.md`](data/README.md) for download instructions.

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | scikit-learn |
| **Image Processing** | Pillow (PIL) |
| **Development** | Google Colab, Jupyter Notebooks |
| **Version Control** | Git / GitHub |

---

## 📁 Project Structure

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

## 🚀 Installation

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

## 💻 Usage

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

## 📈 Results

> _Results will be updated as experiments are completed._

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Baseline CNN | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Custom CNN (Deep) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Transfer Learning (VGG16/ResNet) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Best Model | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## 📅 Project Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| **Week 1** | Data Exploration + Baseline | Dataset analysis, preprocessing pipeline, baseline CNN model |
| **Week 2** | Advanced Models | Custom deep CNN, transfer learning (VGG16/ResNet), hyperparameter tuning |
| **Week 3** | Documentation + Presentation | Model comparison report, final presentation, code cleanup |

---

## 🤝 Team Contributions

| Role | Responsibilities | Assigned To |
|------|-----------------|-------------|
| **Project Lead + Baseline Model** | Repository setup, baseline CNN architecture, team coordination | _[Member 1]_ |
| **Data Pipeline + Custom CNN** | Data loading, preprocessing, augmentation, custom CNN design | _[Member 2]_ |
| **Transfer Learning + Research** | VGG16/ResNet fine-tuning, literature review, architecture comparison | _[Member 3]_ |
| **Evaluation + Model Comparison** | Metrics computation, confusion matrices, performance analysis | _[Member 4]_ |
| **Demo + Deployment** | Live demo notebook, Gradio/Streamlit interface, presentation slides | _[Member 5]_ |

---

## 📚 References

1. Kaggle ASL Alphabet Dataset — https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. TensorFlow Documentation — https://www.tensorflow.org/api_docs
3. Keras Applications (Pre-trained Models) — https://keras.io/api/applications/
4. Simonyan, K. & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* (VGGNet)
5. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* (ResNet)

---

## 📬 Contact

| Team Member | Email |
|-------------|-------|
| _[Member 1]_ | _[email@uowdubai.ac.ae]_ |
| _[Member 2]_ | _[email@uowdubai.ac.ae]_ |
| _[Member 3]_ | _[email@uowdubai.ac.ae]_ |
| _[Member 4]_ | _[email@uowdubai.ac.ae]_ |
| _[Member 5]_ | _[email@uowdubai.ac.ae]_ |

---

## 🙏 Acknowledgments

We would like to thank our instructors at the **University of Wollongong in Dubai (UOWD)** for their guidance and support throughout this project. Special thanks to the CSCI323 Modern Artificial Intelligence teaching team for providing the framework and resources that made this work possible.

---

_This project is developed as part of the CSCI323 coursework at UOWD. For academic use only._
