# Datasets

This directory holds the datasets used by the project. **No dataset files are
tracked by git** — everything here is downloaded or generated locally.

The project uses two independent datasets:

| Dataset | Used by | How to get it |
|---------|---------|---------------|
| **ASL Alphabet Wireframes** | the realtime pipeline (`src/`, `scripts/`) | automatic — `scripts/prep_wireframes.py` downloads it via `kagglehub` |
| **Sign Language MNIST** | the CNN experiment notebooks (`notebooks/`) | manual download from Kaggle (below) |

---

## 1. ASL Alphabet Wireframes (realtime pipeline)

~99,000 MediaPipe hand-skeleton images across the **24 static ASL letters**
(J and Z are motion signs and are excluded).
Source: [dylanpallickara129/asl-alphabet-wireframes](https://www.kaggle.com/datasets/dylanpallickara129/asl-alphabet-wireframes).

Nothing to download by hand. Running the prep script fetches the raw dataset
via `kagglehub` (no Kaggle login needed for this public dataset) and writes the
processed training cache to `data/wireframes_processed/` (gitignored, ~385 MB):

```bash
.venv311/bin/python -m scripts.prep_wireframes
```

## 2. Sign Language MNIST (notebooks only)

28×28 grayscale images as CSVs; used by the `notebooks/` CNN baselines,
independent of the realtime pipeline.
Source: [datamunge/sign-language-mnist](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

Download from Kaggle (free account required) and place the CSVs directly in
this folder:

```
data/
├── sign_mnist_train.csv    # 27,455 rows
└── sign_mnist_test.csv     # 7,172 rows
```

Or with the Kaggle CLI:

```bash
kaggle datasets download -d datamunge/sign-language-mnist -p data/
unzip data/sign-language-mnist.zip -d data/
```

Each CSV row is `label, pixel1..pixel784` (flattened 28×28, values 0–255).
Labels are 0–24 with **9 (J) skipped**; letters J and Z are excluded because
they require motion.

---

## Reference images

`amer_sign2.png`, `amer_sign3.png`, `american_sign_language.PNG` — ASL alphabet
reference charts used in the notebooks.
