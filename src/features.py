from pathlib import Path
import librosa
import numpy as np
from joblib import Parallel, delayed

# Frame-level feature groups, in the order they are concatenated
FEATURE_GROUPS = [
    "mfcc",                # spectral envelope (timbre), one row per coefficient
    "delta_mfcc",          # how the envelope changes: attacks, decays, vibrato
    "spectral_centroid",   # brightness
    "zcr",                 # noisiness / brightness in the time domain
    "spectral_contrast",   # peak-to-valley ratio per octave band: harmonic vs. noisy content
    "spectral_rolloff",    # frequency below which 85% of the energy lies
    "spectral_flatness",   # tone-like (0) vs. noise-like (1)
    "spectral_bandwidth"   # spread of the spectrum around the centroid
]

# Groups whose rows are MFCC coefficients, so they can be cut to fewer coefficients or lose coefficient 0
MFCC_GROUPS = {"mfcc", "delta_mfcc"}

# Extracted with the largest number of MFCCs any experiment uses; fewer coefficients are a slice of these
# (the DCT is orthonormal, so the first k of 40 coefficients equal the coefficients computed with n_mfcc=k)
MAX_MFCC = 40

def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr

def frame_features_from_audio(y, sr, n_mfcc=MAX_MFCC, n_fft=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    return {
        "mfcc": mfcc,
        "delta_mfcc": librosa.feature.delta(mfcc, mode="nearest"),
        "spectral_centroid": librosa.feature.spectral_centroid(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length),
        "spectral_contrast": librosa.feature.spectral_contrast(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length),
        "spectral_rolloff": librosa.feature.spectral_rolloff(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length),
        "spectral_flatness": librosa.feature.spectral_flatness(S=spectrum, n_fft=n_fft, hop_length=hop_length),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length)
    }

def extract_frame_features(filepath, n_mfcc=MAX_MFCC, n_fft=2048, hop_length=512):
    y, sr = load_audio(filepath)
    return frame_features_from_audio(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

def temporal_pooling(feature, pooling="mean"):
    if pooling == "mean":
        return np.mean(feature, axis=1)
    elif pooling == "mean_std":
        mean = np.mean(feature, axis=1)
        std = np.std(feature, axis=1)

        return np.concatenate([mean, std])
    else:
        raise ValueError(f"Invalid pooling strategy: {pooling}, select mean or mean_std")

def pool_frame_features(frame_features):
    # Both statistics of every group are stored, so mean and mean + std pooling come from one extraction
    pooled = {}
    for group, values in frame_features.items():
        pooled[f"{group}_mean"] = np.mean(values, axis=1)
        pooled[f"{group}_std"] = np.std(values, axis=1)
    return pooled

def extract_pooled_features(filepath, n_fft=2048, hop_length=512):
    return pool_frame_features(extract_frame_features(filepath, n_fft=n_fft, hop_length=hop_length))

def extract_window_pooled_features(filepath, window_seconds=3.0, hop_seconds=1.5, n_fft=2048, hop_length=512):
    # Long excerpts are cut into windows as long as the 3 s training excerpts, pooled features per window
    y, sr = load_audio(filepath)
    window = int(window_seconds * sr)
    hop = int(hop_seconds * sr)
    starts = range(0, max(len(y) - window, 0) + 1, hop)

    windows = [pool_frame_features(frame_features_from_audio(y[start:start + window], sr, n_fft=n_fft, hop_length=hop_length)) for start in starts]

    return {key: np.vstack([w[key] for w in windows]) for key in windows[0]}

def _cached(cache_path, filepaths, extract):
    # Cache keyed by file names, so a moved data folder still hits the cache but a different file list does not
    names = np.array([Path(filepath).name for filepath in filepaths])

    if cache_path is not None and Path(cache_path).exists():
        cache = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cache["names"], names):
            return {key: cache[key] for key in cache.files if key != "names"}
        print(f"Cache {cache_path} does not match the file list, re-extracting")

    arrays = extract()

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, names=names, **arrays)

    return arrays

def build_pooled_features(df, n_fft=2048, hop_length=512, cache_path=None, n_jobs=-1):
    # Pooled statistics of every feature group for every file (in parallel), rows in the order of df
    def extract():
        per_file = Parallel(n_jobs=n_jobs)(delayed(extract_pooled_features)(filepath, n_fft, hop_length) for filepath in df["filepath"])
        return {key: np.vstack([pooled[key] for pooled in per_file]) for key in per_file[0]}

    return _cached(cache_path, df["filepath"], extract)

def build_window_pooled_features(df, window_seconds=3.0, hop_seconds=1.5, n_fft=2048, hop_length=512, cache_path=None, n_jobs=-1):
    # Pooled statistics of all windows of all files stacked, plus the index of the file each window belongs to
    def extract():
        per_file = Parallel(n_jobs=n_jobs)(
            delayed(extract_window_pooled_features)(filepath, window_seconds, hop_seconds, n_fft, hop_length)
            for filepath in df["filepath"]
        )
        arrays = {key: np.vstack([pooled[key] for pooled in per_file]) for key in per_file[0]}
        arrays["file_index"] = np.concatenate([np.full(len(pooled["mfcc_mean"]), i) for i, pooled in enumerate(per_file)])
        return arrays

    arrays = _cached(cache_path, df["filepath"], extract)
    file_index = arrays.pop("file_index")

    return arrays, file_index

def assemble_features(pooled, spec, pooling="mean", return_names=False):
    # Builds the feature matrix of a feature spec: {"groups": [...], "n_mfcc": 13, "drop_c0": False, ...}.
    # Per group the means come first, then (for mean + std pooling) the standard deviations
    statistics = {"mean": ["mean"], "mean_std": ["mean", "std"]}[pooling]
    first_coefficient = 1 if spec.get("drop_c0", False) else 0

    columns = []
    names = []
    for group in spec["groups"]:
        for statistic in statistics:
            values = pooled[f"{group}_{statistic}"]
            if group in MFCC_GROUPS:
                values = values[:, first_coefficient:spec["n_mfcc"]]
                indices = range(first_coefficient, spec["n_mfcc"])
            else:
                indices = range(values.shape[1])
            columns.append(values)
            names += [(group, f"{group}_{statistic}_{i}") for i in indices]

    x = np.hstack(columns)

    return (x, names) if return_names else x

def build_features(df, specs, poolings, cache_dir, hop_length=512):
    # features[pooling][spec name] for all specs; one cached extraction per window size (n_fft)
    pooled_by_n_fft = {
        n_fft: build_pooled_features(df, n_fft=n_fft, hop_length=hop_length, cache_path=Path(cache_dir) / f"train_nfft{n_fft}.npz")
        for n_fft in sorted({spec["n_fft"] for spec in specs.values()})
    }

    return {
        pooling: {name: assemble_features(pooled_by_n_fft[spec["n_fft"]], spec, pooling) for name, spec in specs.items()}
        for pooling in poolings
    }
