from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "IRMAS-TrainingData" / "IRMAS-TrainingData"
CACHE_DIR = PROJECT_ROOT / "cache" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42
REPEAT_SEEDS = [0, 1, 2, 3, 4]

# Frame settings of the first experiments: 2048-sample window (46 ms at 44.1 kHz), 512-sample hop, 13 MFCCs
DEFAULT_FEATURE_PARAMS = {"n_mfcc": 13, "n_fft": 2048, "drop_c0": False}
HOP_LENGTH = 512

# Feature groups of each feature set (see FEATURE_GROUPS in src/features.py)
FEATURE_SET_GROUPS = {
    "mfcc": ["mfcc"],
    "low_level": ["spectral_centroid", "zcr"],
    "combined": ["mfcc", "spectral_centroid", "zcr"],
    "extended": ["mfcc", "delta_mfcc", "spectral_centroid", "zcr", "spectral_contrast", "spectral_rolloff", "spectral_flatness", "spectral_bandwidth"]
}
FEATURE_SETS = list(FEATURE_SET_GROUPS)
POOLINGS = ["mean", "mean_std"]
FEATURE_SET_NAMES = {"mfcc": "MFCC", "low_level": "Low-level", "combined": "Combined", "extended": "Extended"}
POOLING_NAMES = {"mean": "Mean", "mean_std": "Mean + Std"}

# Sliding windows for the official test set, as long as the training excerpts
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5

def feature_spec(feature_set, **params):
    return {"groups": FEATURE_SET_GROUPS[feature_set], **DEFAULT_FEATURE_PARAMS, **params}

def official_test_dirs():
    return sorted(DATA_DIR.glob("IRMAS-TestingData*"))
