# music-instrument-recognition

The goal is to build a musical instrument recognition system using the **IRMAS dataset** and compare different audio features, pooling strategies, and classification models.

## Features

Frame-level features, computed on 2048-sample windows (46 ms) with a 512-sample hop:

- MFCCs (spectral envelope / timbre)
- Delta MFCCs (how the envelope changes: attack, decay, vibrato)
- Spectral centroid, rolloff, bandwidth and flatness (brightness, extent, spread, noisiness)
- Spectral contrast (peak-to-valley ratio per octave band: harmonic vs. noisy content)
- Zero-crossing rate

Feature sets:

| Set | Content |
|---|---|
| MFCC | MFCCs |
| Low-level | spectral centroid + ZCR |
| Combined | MFCCs + spectral centroid + ZCR |
| Extended | all of the above, with the groups and parameters chosen by the feature study |

The **feature study** (`02_training.ipynb`, evaluated in `03_evaluation.ipynb`) checks the choices on the validation split, one at a time:
- each feature group left out in turn;
- 13 / 20 / 40 MFCCs;
- MFCC 0 (loudness) kept or dropped;
- window length 1024 / 2048 / 4096 samples.

`01_data_exploration.ipynb` explains why each feature suits instrument recognition, and why pitch features (chroma) are left out.

Pooling strategies (frame features to one vector per excerpt):

- Mean pooling
- Mean + standard deviation pooling

## Models

Four kinds of model:
- Logistic Regression (linear)
- k-nearest neighbours (instance-based)
- SVM with RBF kernel
- PyTorch MLP (optional dropout, weight decay and early stopping)

Each model is evaluated with its default settings and after a grid search on the validation split (`PARAM_GRIDS` in `src/experiments.py`).

The evaluation compares and contrasts the models through:
- decision regions in 2D;
- learning curves;
- hyperparameter effects (kNN's k, the SVM's C and γ);
- training and prediction cost, and model size;
- permutation importance of each feature group.

## Evaluation

Metrics:

- Accuracy
- Macro F1-score
- Precision and recall per class
- Confusion matrix, most frequent confusions, and the share of errors within an instrument family

Protocol:

- **Recording-level split.** The ~6700 training excerpts come from ~2000 recordings (about 3 excerpts each). Splitting excerpts at random puts parts of the same recording in train and test and inflates the scores. The experiments use a stratified 70/15/15 split in which every recording is in exactly one split, and they measure how large the inflation of the random split is.
- **Validation for every choice.** Feature set, pooling, model and hyperparameters are all chosen on the validation split by macro F1. The test split is used once, at the end.
- **Variance.** The final comparison is repeated over 5 recording-level splits (tuning and testing on each), reported as mean ± std. All randomness (splits, MLP initialization and batch order) is seeded.
- **Official IRMAS test set.** Multi-label 5-20 s excerpts, cut into 3 s windows (1.5 s hop) whose class probabilities are averaged per excerpt. Reported as micro/macro F1 over a range of thresholds, plus threshold-free top-1 precision, label ranking average precision and macro average precision.

## Repository Structure

```text
.
├── data/                           # IRMAS (not included)
├── docs/
│   └── 02_training.md              # step-by-step explanation of the training notebook
├── notebooks/
│   ├── 01_data_exploration.ipynb   # problem definition, dataset statistics, feature extraction illustrated
│   ├── 02_training.ipynb           # splits, training and tuning, saves all models
│   └── 03_evaluation.ipynb         # loads the models, writes all result tables and figures
├── cache/features/                 # feature cache (git-ignored)
├── models/
│   ├── manifest.csv                # one row per saved model: experiment, split, features, model, params, file
│   ├── splits.csv                  # train/val/test assignment of every excerpt for every split
│   ├── best_config.json            # best feature set + pooling on validation
│   ├── feature_specs.json          # groups and parameters of every feature set used
│   ├── tuning/                     # grid search results
│   └── <experiment>/<split>/*.joblib   # scaler + model (git-ignored)
├── results/
│   ├── figures/
│   └── *.csv                       # result tables
├── src/
│   ├── config.py                   # paths, seeds, feature sets and poolings
│   ├── data.py                     # loading, recording ids, splits, official test set
│   ├── evaluation.py               # metrics, plots, error analysis, multi-label evaluation
│   ├── experiments.py              # hyperparameter search, saving/loading/scoring models
│   ├── features.py                 # feature extraction, pooling, caching, windowing
│   └── models.py                   # Logistic Regression, kNN, SVM, MLP
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

IRMAS is hosted on [Zenodo](https://zenodo.org/records/1290750) (CC BY-NC-SA, non-commercial use only). The first code cells of `01_data_exploration.ipynb` download it, check the MD5 checksums and extract it into `data/`. The same can be done from Python:

```python
from src.config import DATA_DIR
from src.data import download_irmas

download_irmas(DATA_DIR, include_test=True)
```

- **Sizes:** training data about 3.2 GB; the three official test parts about 7.8 GB, needed only for the official test set sections. Plan for roughly 15 GB of free disk space.
- **Re-running:** already extracted parts are skipped, and interrupted downloads resume.

The result (also the layout to use when downloading manually):

```text
data/
├── IRMAS-TrainingData/IRMAS-TrainingData/<cel, cla, ...>/*.wav
├── IRMAS-TestingData-Part1/IRMAS-TestingData-Part1/Part1/*.wav, *.txt
├── IRMAS-TestingData-Part2/IRMAS-TestingData-Part2/IRTestingData-Part2/*.wav, *.txt
└── IRMAS-TestingData-Part3/IRMAS-TestingData-Part3/Part3/*.wav, *.txt
```

The test parts are searched recursively for `.wav` files with a matching `.txt` annotation. Without them, the official test set sections of `01_data_exploration.ipynb` and `03_evaluation.ipynb` are skipped.

The dataset itself is not included in this repository.

## Running

Start Jupyter and run the notebooks in order, each from top to bottom:

```bash
jupyter notebook
```

1. `01_data_exploration.ipynb` defines the problem and explores the data. It extracts the features of all excerpts (in parallel) and caches them in `cache/features/`, so later notebooks load them instantly.
2. `02_training.ipynb` creates the splits, then trains and tunes all models and saves them in `models/`. The experiments are:
   - `feature_study`: choosing the features and their parameters;
   - `leakage`: random vs. recording-level split;
   - `comparison`: feature set × pooling × model, default and tuned;
   - `repeated`: 5 more splits;
   - `learning_curve`: 10% / 25% / 50% of the training data;
   - `illustration`: models on 2 principal components, to draw decision boundaries;
   - `official`: SVM with probability estimates.

   This is the slow step, because of the grid searches.
3. `03_evaluation.ipynb` loads the saved models and evaluates them. It writes every table to `results/*.csv` and every figure to `results/figures/`, and ends with a summary table (`results/summary.csv`).

The evaluation notebook can be re-run on its own as long as `models/` exists. Reusable code is kept under `src/`.
