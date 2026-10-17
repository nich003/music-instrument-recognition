import hashlib
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

LABEL_MAP = {
    "cel": "cello",
    "cla": "clarinet",
    "flu": "flute",
    "gac": "acoustic_guitar",
    "gel": "electric_guitar",
    "org": "organ",
    "pia": "piano",
    "sax": "saxophone",
    "tru": "trumpet",
    "vio": "violin",
    "voi": "voice"
}

# IRMAS training excerpts are named e.g. "[cel][cla]0001__2.wav" or "008__[vio][nod][cou_fol]2194__1.wav":
# the number before the final "__N" identifies the source recording, "N" the excerpt within it
RECORDING_ID_PATTERN = re.compile(r"(\d+)__\d+\.wav$")
TAG_PATTERN = re.compile(r"\[([a-z_-]+)\]")

GENRE_MAP = {
    "cla": "classical",
    "cou_fol": "country_folk",
    "jaz_blu": "jazz_blues",
    "lat_sou": "latin_soul",
    "pop_roc": "pop_rock",
    "---": "unknown"
}

# IRMAS on Zenodo (https://zenodo.org/records/1290750), CC BY-NC-SA. Archive name -> MD5 checksum
IRMAS_URL = "https://zenodo.org/records/1290750/files/{name}.zip?download=1"
IRMAS_TRAIN_ARCHIVE = {"IRMAS-TrainingData": "4fd9f5ed5a18d8e2687e6360b5f60afe"}
IRMAS_TEST_ARCHIVES = {
    "IRMAS-TestingData-Part1": "5a2e65520dcedada565dff2050bb2a56",
    "IRMAS-TestingData-Part2": "afb0c8ea92f34ee653693106be95c895",
    "IRMAS-TestingData-Part3": "9b3fb2d0c89cdc98037121c25bd5b556"
}


def recording_id(filepath):
    name = Path(filepath).name
    match = RECORDING_ID_PATTERN.search(name)

    # Grouping only by the number (not by the class tags) is the conservative choice:
    # if it is a global recording id this is exactly right, if not it merges more files than needed, which never leaks
    if match:
        return match.group(1)

    return Path(filepath).stem


def filename_tags(filepath):
    # The first tag is the instrument; the optional others are drums ([dru]/[nod]) and genre (e.g. [cla] = classical)
    tags = TAG_PATTERN.findall(Path(filepath).name)[1:]
    drums = "drums" if "dru" in tags else "no_drums" if "nod" in tags else "unknown"
    genres = [GENRE_MAP.get(tag, tag) for tag in tags if tag not in ("dru", "nod")]
    
    return drums, genres[0] if genres else "unknown"


def load_dataset(data_dir):
    data_dir = Path(data_dir)

    rows = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_short_label = class_dir.name
        if class_short_label not in LABEL_MAP:
            continue

        for filepath in sorted(class_dir.glob("*.wav")):
            drums, genre = filename_tags(filepath)
            rows.append({
                "filepath": str(filepath),
                "label": LABEL_MAP[class_short_label],
                "label_short": class_short_label,
                "recording_id": recording_id(filepath),
                "drums": drums,
                "genre": genre
            })

    return pd.DataFrame(rows)

def split_dataset(df, seed=42):
    # Random 70/15/15 split. Excerpts of the same recording can end up in different splits (leakage)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)

    val_df, test_df =  train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)

    return train_df, val_df, test_df

def group_split_dataset(df, seed=42, n_folds=20, val_folds=3, test_folds=3):
    # Stratified 70/15/15 split where every recording lies in exactly one split.
    # The data is cut into 20 stratified, recording-disjoint folds: 3 go to test, 3 to validation, the rest to training
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_row = np.empty(len(df), dtype=int)

    for fold, (_, fold_idx) in enumerate(splitter.split(df, df["label"], groups=df["recording_id"])):
        fold_of_row[fold_idx] = fold

    test_mask = fold_of_row < test_folds
    val_mask = (fold_of_row >= test_folds) & (fold_of_row < test_folds + val_folds)
    train_mask = ~(test_mask | val_mask)

    return df[train_mask], df[val_mask], df[test_mask]

def load_test_dataset(test_dirs):
    # Official IRMAS test set: variable-length excerpts, each with a .txt listing one or more instruments
    rows = []

    for test_dir in test_dirs:
        for filepath in sorted(Path(test_dir).rglob("*.wav")):
            annotation_path = filepath.with_suffix(".txt")
            if not annotation_path.exists():
                continue

            short_labels = [line.strip() for line in annotation_path.read_text().splitlines()]
            labels = sorted({LABEL_MAP[label] for label in short_labels if label in LABEL_MAP})

            rows.append({
                "filepath": str(filepath),
                "labels": labels
            })

    return pd.DataFrame(rows)


def _md5(path, chunk_size=1 << 24):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def _download(url, path, chunk_size=1 << 20):
    # Resumes from a partial download if the server supports range requests, otherwise starts over
    partial = path.with_name(path.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    
    with urllib.request.urlopen(request, timeout=60) as response:
        if start and response.status != 206:
            start = 0
        total = start + int(response.headers.get("Content-Length", 0))
        
        with open(partial, "ab" if start else "wb") as f:
            done = start
            next_report = done
            while chunk := response.read(chunk_size):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"\r  {path.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
                    next_report = done + 100 * chunk_size
    print()
    
    partial.rename(path)

def download_irmas(data_dir, include_test=True, keep_archives=False):
    # Downloads, verifies and extracts IRMAS into data_dir/<archive name>/<archive name>/...,
    # e.g. data/IRMAS-TrainingData/IRMAS-TrainingData/cel/*.wav. Already extracted parts are skipped
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archives = {**IRMAS_TRAIN_ARCHIVE, **(IRMAS_TEST_ARCHIVES if include_test else {})}
    
    for name, md5 in archives.items():
        target = data_dir / name
        if target.exists() and any(target.iterdir()):
            print(f"{name}: already extracted, skipping")
            continue
        
        archive = data_dir / f"{name}.zip"
        if not archive.exists():
            print(f"{name}: downloading")
            _download(IRMAS_URL.format(name=name), archive)
        
        print(f"{name}: verifying checksum")
        if _md5(archive) != md5:
            archive.unlink()
            raise RuntimeError(f"{archive.name} is corrupt (MD5 mismatch) and was deleted, run again to re-download")
        
        print(f"{name}: extracting")
        partial_target = data_dir / f"{name}.partial"
        shutil.rmtree(partial_target, ignore_errors=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(partial_target)
        partial_target.rename(target)
        
        if not keep_archives:
            archive.unlink()
    
    print("Done")
