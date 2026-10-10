# music-instrument-recognition

The goal is to build a musical instrument recognition system using the **IRMAS dataset** and compare different audio features, pooling strategies, and classification models.

## Planned Features

- MFCCs
- Spectral centroid
- Zero-crossing rate

Pooling strategies:

- Mean pooling
- Mean + standard deviation pooling

## Models

We plan to compare:

- Logistic Regression
- SVM
- PyTorch MLP

## Evaluation

The main evaluation metrics will be:

- Accuracy
- Macro F1-score
- Precision and recall per class
- Confusion matrix

## Current Repository Structure

```text
.
├── data/
├── notebooks/
│   └── debug.ipynb
├── results/
├── src/
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   └── models.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Download the IRMAS dataset and place it under:

```text
data/
```

The dataset itself is not included in this repository.

## Running

Development and experiments are currently done mainly through:

```text
notebooks/debug.ipynb
```

Reusable code is kept under `src/`.

To start Jupyter:

```bash
jupyter notebook
```