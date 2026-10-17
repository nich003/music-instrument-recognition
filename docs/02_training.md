# What happens in `02_training.ipynb`

This document walks through `notebooks/02_training.ipynb` cell by cell: what each step does, why it is done that way, and what it writes to disk.

## In one paragraph

The notebook trains **every model the project needs** and saves each one to disk; it does not report final results. It first fixes the data splits, then uses a feature study to decide what the `extended` feature set contains. It then runs six more experiments: a leakage comparison, the main feature/pooling/model comparison, repeated splits, learning curves, 2D models for plotting, and one SVM with probability outputs. Every choice is made on the **validation** split. The test split and the official IRMAS test set are not used here at all. `03_evaluation.ipynb` later loads the saved models and computes all reported numbers.

## Inputs and outputs

**Reads**
- `data/IRMAS-TrainingData/IRMAS-TrainingData/<instrument>/*.wav`: the 6705 training excerpts (3 s each).
- `cache/features/train_nfft*.npz`: cached features, if `01_data_exploration.ipynb` or an earlier run already extracted them. Missing caches are created.

**Writes** (all under `models/`)

| File | Content |
|---|---|
| `splits.csv` | For every excerpt and every split: `train`, `val` or `test` |
| `feature_specs.json` | The exact definition (groups, number of MFCCs, window length, MFCC 0) of every feature set used |
| `best_config.json` | The best feature set + pooling on validation |
| `tuning/*.csv` | Full grid search results for every comparison configuration |
| `<experiment>/<split>/*.joblib` | One file per trained model: the fitted scaler + the fitted model |
| `manifest.csv` | One row per saved model: experiment, split, feature set, pooling, model, variant, hyperparameters, training time, file path |

The evaluation notebook needs nothing else: it rebuilds the features from the cache, reads the splits and specs, and loads the models listed in the manifest.

## Key ideas used throughout

**Train / validation / test.**
- *Training split:* models learn from it.
- *Validation split:* used to **make choices**: hyperparameters, features, which configuration is best.
- *Test split:* used **once**, in `03_evaluation.ipynb`, to report how the chosen models do on unseen data.

If the test split influenced any choice, its score would be optimistic. That is why this notebook never looks at it.

**Recording-level split.** IRMAS has ~2000 recordings, and most contribute 3 excerpts. If excerpts of one recording land in both train and test, a model can score well by recognising the *recording* (its sound, mix, room) rather than the *instrument*. Recording-level splits put every recording in exactly one split.

**Macro F1.** The F1 score is computed per instrument and then averaged, so every instrument counts equally. With imbalanced classes (388 cello vs. 778 voice excerpts), plain accuracy would favour the large classes.

**Scaling.** All features are standardised (zero mean, unit variance) with a `StandardScaler` that is **fit on the training split only** and then applied to validation and test. Fitting it on all data would leak information from validation/test into training.

**Default vs. tuned.**
- *Default:* the settings of the first experiments.
- *Tuned:* the best settings found by a grid search on validation (`tune_model` in `src/experiments.py`). For every combination of hyperparameters, it trains on the training split, computes validation macro F1, and keeps the model with the highest score.

---

## Step by step

### Cell 0: Overview (markdown)
A table of the seven experiments, explained below.

### Cell 1: Setup
- Makes the project root importable (`sys.path.append`), so `from src... import ...` works from inside `notebooks/`.
- Imports the project modules:
  - `src/config.py`: paths, seeds, feature set definitions;
  - `src/data.py`: loading and splitting;
  - `src/features.py`: feature extraction;
  - `src/models.py`: the four models;
  - `src/experiments.py`: tuning, saving, splits table.
- Creates `models/` and `models/tuning/`.
- `load_dataset` builds a table (`df`) with one row per excerpt: file path, instrument label, recording id, and the drum and genre tags parsed from the file name. Files are read in sorted order, so `df` is identical on every machine.

`df` has a plain 0…6704 index. Every feature matrix later has its rows in exactly this order, so `x[train_df.index]` selects the training rows of any feature matrix.

### Cell 3: Splits
Seven splits are created:

| Name | How | Used by |
|---|---|---|
| `random_seed42` | Random 70/15/15 split stratified by instrument (`split_dataset`): the split of the first experiments | `leakage` only |
| `grouped_seed42` | Recording-level 70/15/15 split (`group_split_dataset`) | Everything else |
| `grouped_seed0` … `grouped_seed4` | The same, with other seeds | `repeated` |

How the recording-level split works (`group_split_dataset` in `src/data.py`):
1. `StratifiedGroupKFold` cuts the data into **20 folds**. Each recording is in exactly one fold, and each fold has roughly the same instrument proportions.
2. Folds 0-2 become test (15%), folds 3-5 validation (15%), and the remaining 14 folds training (70%).

The cell prints, for every split, how many test excerpts have a recording that also appears in training. For the random split this is high; for the grouped splits it must be 0%. An `assert` stops the notebook if a grouped split leaks.

All split assignments are saved to `models/splits.csv`.

### Cell 4: Helpers
- `manifest = []` starts the in-memory list of saved models. Every experiment appends one row per saved model; the last cell writes it to `manifest.csv`.
- `split_data(split_name, x)`: for a feature matrix `x`, returns the scaled training and validation features, their labels and the fitted scaler.
- `manifest_params(...)`: looks up the tuned hyperparameters of a model that was saved earlier in this run. Later experiments use it to reuse the `comparison` settings instead of tuning again.

---

### Cells 5-7: Experiment `feature_study` (choosing the features)

**Goal:** decide, on evidence, which feature groups and extraction settings to use, instead of keeping library defaults.

**The feature groups** (computed per 46 ms frame, then summarised per excerpt by mean and std):

| Group | Values per frame | Describes |
|---|---|---|
| `mfcc` | 13 (or 20/40) | Spectral envelope = timbre |
| `delta_mfcc` | same as MFCC | How the envelope changes: attack, decay, vibrato |
| `spectral_contrast` | 7 | Harmonic peaks vs. noise per octave band |
| `spectral_centroid` | 1 | Brightness |
| `zcr` | 1 | Noisiness (time domain) |
| `spectral_rolloff` | 1 | Upper extent of the spectrum |
| `spectral_flatness` | 1 | Tone-like vs. noise-like |
| `spectral_bandwidth` | 1 | Spread of the spectrum |

With mean + std pooling and the default 13 MFCCs, the full extended set has (13 + 13 + 7 + 1 + 1 + 1 + 1 + 1) × 2 = **76 features**.

**Cell 6: the variants.** Starting from "extended with default settings" (`study_extended`: 13 MFCCs, MFCC 0 kept, 2048-sample window), exactly one thing is changed per variant:

| Variant | Change |
|---|---|
| `study_combined` | Reference: the 30-feature set of the first experiments (MFCC + centroid + ZCR) |
| `study_extended` | Reference: all groups, default settings |
| `study_no_delta_mfcc` | Without delta MFCCs |
| `study_no_spectral_contrast` | Without spectral contrast |
| `study_no_spectral_shape` | Without rolloff, flatness and bandwidth (removed together) |
| `study_no_centroid_zcr` | Without centroid and ZCR (removed together) |
| `study_n_mfcc_20`, `study_n_mfcc_40` | 20 or 40 MFCCs instead of 13 |
| `study_drop_c0` | MFCC 0 (overall loudness) removed |
| `study_n_fft_1024`, `study_n_fft_4096` | Window of 1024 (23 ms) or 4096 (93 ms) samples instead of 2048 (46 ms); the hop stays 512 |

**How features are built cheaply** (`build_features` in `src/features.py`):
- **One pass over the audio per window length.** Each pass computes all groups with **40 MFCCs** and stores the mean and std of every group in `cache/features/train_nfft<window>.npz`. It runs in parallel on all CPU cores, and only happens the first time; later runs load the cache.
- **Everything else is a column selection.** "13 MFCCs" is the first 13 of the 40, which is exactly what computing 13 directly gives. "Without MFCC 0" drops one column, and "without group X" drops that group's columns.
- So the 11 variants need only 3 passes over the audio, one per window length.

**Scoring each variant:**
1. Take the variant's features (mean + std pooling) on the `grouped_seed42` split.
2. **Tune an SVM**: 16 combinations of C ∈ {0.1, 1, 10, 100} × γ ∈ {scale, 0.003, 0.01, 0.1}, each trained on the training split and scored on validation.
3. The best validation macro F1 becomes the variant's score (`study_val_f1[name]`), and the winning SVM is saved.

Why the SVM is re-tuned for every variant: the best C and γ depend on how many features there are (30 to 184). A fixed setting would favour whichever variant it happens to suit. Tuning each one compares every variant at its best.

Why an SVM: it was the best model in the first experiments, and it is fast enough to tune 11 times.

**Cell 7: turning scores into a decision.** The reference is `baseline_f1`, the score of `study_extended`.
- `removed_groups`: an ablation's groups are removed **only if leaving them out scored higher** than the reference. Otherwise they stay.
- `n_mfcc`: whichever of 13 / 20 / 40 scored best (13 is the reference score).
- `n_fft`: whichever of 1024 / 2048 / 4096 scored best (2048 is the reference score).
- `drop_c0`: true only if dropping MFCC 0 scored higher than the reference.

The result, `extended_spec`, defines the `extended` feature set for all later experiments.

**Limitations to mention in a report:**
- **One change at a time.** Combinations such as "40 MFCCs *and* a longer window" are never tested, and each removal is judged alone, not together with the others.
- **No minimum improvement.** Any gain, however small, counts. Differences of ~0.01 F1 can be noise on one validation split.
- **The scores are slightly optimistic.** Each one is the best of 16 tries on the same validation data. This affects all variants equally, so the comparison stays fair, but the absolute numbers are a bit high.
- **One model.** The choice is made with the SVM and may not be optimal for the other models.

### Cell 9: Feature sets for the rest of the notebook
Four feature sets are fixed:

| Set | Groups | Settings |
|---|---|---|
| `mfcc` | MFCC | defaults (kept as in the first experiments, so results stay comparable) |
| `low_level` | centroid + ZCR | defaults |
| `combined` | MFCC + centroid + ZCR | defaults |
| `extended` | from the feature study | from the feature study |

All feature definitions, including the study variants, are saved to `models/feature_specs.json`. Then each set is assembled for both poolings:
- **`mean`:** the average of each feature over the excerpt's frames;
- **`mean_std`:** the average and the standard deviation. The std captures how much the sound varies over time.

The result is `features[pooling][feature_set]`, a matrix with one row per excerpt.

---

### Cell 11: Experiment `leakage`
**Question:** how much does the random split inflate the scores?

The same features (combined, mean + std) and the same **default** models (all four) are trained twice: on the random split and on the recording-level split. Only the split differs, so any score difference in `03_evaluation.ipynb` is caused by recordings shared between train and test.

**Saved:** 2 splits × 4 models = 8 models.

### Cells 13-14: Experiment `comparison` (the main experiment)
**Question:** which feature set, pooling and model work best, and how much does tuning help?

For each of the **4 feature sets × 2 poolings = 8 configurations**, and for each of the **4 models**:
1. **Default model:** trained with the default settings and saved.
2. **Tuned model:** grid search on validation (`tune_model`); the best one is saved. The full grid search table goes to `models/tuning/<feature_set>_<pooling>_<model>.csv`.

**The four models** (`src/models.py`):

| Model | Default settings | Grid searched (cell 13 prints it) | Number of settings |
|---|---|---|---|
| Logistic Regression | C = 1 | C ∈ {0.01, 0.1, 1, 10, 100} | 5 |
| kNN | k = 5, uniform votes, Euclidean | k ∈ {1, 3, 5, 11, 21, 41} × uniform/distance-weighted votes × Manhattan/Euclidean | 24 |
| SVM (RBF) | C = 1, γ = scale | C ∈ {0.1, 1, 10, 100} × γ ∈ {scale, 0.003, 0.01, 0.1} | 16 |
| MLP (PyTorch) | 2 hidden layers (64, 32), no dropout, no weight decay, 100 epochs | layers (64, 32) or (256, 128) × dropout 0 / 0.3 × weight decay 0 / 1e-4, up to 300 epochs with early stopping (patience 20) | 8 |

What the hyperparameters do:
- **C** (LR, SVM): how strongly the model is regularised. Small C means a simpler model; large C means it fits the training data more closely.
- **γ** (SVM): the width of the RBF kernel. Large γ means each training point influences only a small area, so the boundary becomes more wiggly.
- **k** (kNN): how many nearest training excerpts vote. Small k is very local and noisy; large k is smoother.
- **Dropout and weight decay** (MLP): regularisation against overfitting.
- **Early stopping** (MLP): training stops when the validation loss has not improved for 20 epochs, and the weights of the best epoch are restored.

The MLP is trained with Adam (learning rate 0.001, batch size 64), seeded so it gives the same result every run. The models run on the CPU.

**Saved:** 8 configurations × 4 models × (default + tuned) = 64 models, plus 32 tuning tables.

### Cell 15: Choosing the best configuration
The configuration (feature set + pooling) with the **highest tuned validation macro F1 of any model** is chosen. It is saved to `models/best_config.json`, and `x_best` holds its feature matrix.

Only the *configuration* is used from here on. All four models are still evaluated on it, so they can be compared on equal terms.

### Cell 17: Experiment `repeated`
**Question:** how much do the results depend on the particular split?

For each of the 5 extra recording-level splits (seeds 0-4), all four models are **tuned again on that split's validation data** and saved. Each split has its own test set, so `03_evaluation.ipynb` can report the mean ± standard deviation over 5 independent train/validation/test splits. If two models differ by less than this spread, the difference is not meaningful.

**Saved:** 5 splits × 4 models = 20 models.

### Cell 19: Experiment `learning_curve`
**Question:** how much training data does each model need?

The tuned settings from `comparison` (best configuration) are reused. Each model is retrained on a random **10%, 25% and 50%** of the training split, stratified by instrument; the scaler is fit on that subset only. The 100% point is the `comparison` model itself. Validation stays the full validation split. (The MLP also uses it for early stopping.)

How to read the curves in `03_evaluation.ipynb`:
- a model that is still rising at 100% would profit from more data;
- a model that is flat early has reached its capacity.

**Saved:** 3 fractions × 4 models = 12 models. The manifest records `train_fraction` and `train_size`.

### Cell 21: Experiment `illustration`
**Question:** what do the decision boundaries of the four models look like?

Boundaries in a 76-dimensional space can't be drawn, so:
1. The best configuration's features are scaled and reduced to their **first two principal components** (PCA, fit on the training split).
2. Each model is trained on these two numbers per excerpt, with its tuned settings.

`03_evaluation.ipynb` colours a 2D grid by each model's prediction:
- logistic regression draws straight boundaries;
- kNN a patchwork around individual points;
- the SVM smooth curves;
- the MLP smooth, learned shapes.

These 2D models score far below the real ones; they exist only for the picture. The scaler + PCA pipeline is saved in place of the scaler, so the evaluation notebook can project new data the same way.

**Saved:** 4 models.

### Cell 23: Experiment `official`
**Why it exists:** the official IRMAS test set has 5-20 s excerpts with possibly several instruments each. `03_evaluation.ipynb` cuts them into 3 s windows, averages each model's **class probabilities** over the windows, and predicts every instrument whose probability is close to the top one. That needs probabilities.

- Logistic Regression, kNN and the MLP output probabilities directly, so their tuned `comparison` models are reused.
- An SVM only outputs decision scores. Here the tuned SVM is wrapped in `CalibratedClassifierCV` (Platt scaling). It fits a sigmoid that maps the SVM's scores to probabilities, using 5-fold cross-validation on the training split, and keeps one SVM trained on the whole training split.

This is the replacement for `SVC(probability=True)`, which scikit-learn 1.9 deprecated.

**Saved:** 1 model.

### Cell 24: Writing the manifest
The in-memory list becomes `models/manifest.csv`, and the cell prints how many models each experiment saved.

**In total: 120 saved models:**

| Experiment | Models |
|---|---|
| `feature_study` | 11 |
| `leakage` | 8 |
| `comparison` | 64 |
| `repeated` | 20 |
| `learning_curve` | 12 |
| `illustration` | 4 |
| `official` | 1 |

Counting every grid search candidate, about **930 fits** are trained (feature study 176, comparison 456, repeated 265, the rest ~30). Most of the time goes into the SVM grids and the MLP grids.

---

## Running and re-running

- **Run order:** `01_data_exploration.ipynb` (optional, but it downloads the data and creates the first feature cache) → `02_training.ipynb` → `03_evaluation.ipynb`.
- **Run it top to bottom in one session.** The manifest is built in memory during the run and only written in the last cell, and later experiments look up settings saved by earlier ones (`manifest_params`). Running only some cells after a kernel restart therefore either fails, or overwrites `manifest.csv` with an incomplete list. To redo a single experiment, restart the kernel and run the whole notebook again; the feature cache makes the reruns skip feature extraction.
- **After changing anything in `src/`,** restart the kernel. A running kernel keeps the old version of the module in memory.
- **Determinism:**
  - all splits, the MLP initialisation and batch order are seeded;
  - the calibration's cross-validation folds are fixed;
  - re-running gives the same models up to small floating-point differences.
- **Hardware:**
  - everything runs on the CPU;
  - feature extraction uses all cores;
  - the grid searches run one fit at a time.
