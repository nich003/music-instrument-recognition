import json
import time
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.models import fit_scaler, fit_model, MODEL_NAMES

# Searched on the validation split. The MLP grid always uses early stopping, so "epochs" is only an upper bound
PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "kNN": {"n_neighbors": [1, 3, 5, 11, 21, 41], "weights": ["uniform", "distance"], "p": [1, 2]},
    "SVM": {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.003, 0.01, 0.1]},
    "MLP": {
        "hidden_dims": [(64, 32), (256, 128)],
        "dropout": [0.0, 0.3],
        "weight_decay": [0.0, 1e-4],
        "epochs": [300],
        "patience": [20]
    }
}

MODEL_SLUGS = {"Logistic Regression": "lr", "kNN": "knn", "SVM": "svm", "MLP": "mlp"}

def param_combinations(param_grid):
    keys = list(param_grid)
    return [dict(zip(keys, values)) for values in product(*param_grid.values())]

def scaled_split(x, train_df, val_df, test_df):
    # Feature rows are aligned with the full dataframe (RangeIndex), so split indices select rows directly.
    # The scaler is fit on the training split only
    x_train, scaler = fit_scaler(x[train_df.index])

    return x_train, scaler.transform(x[val_df.index]), scaler.transform(x[test_df.index]), scaler

def timed_fit(model_name, x_train, y_train, x_val, y_val, params=None, seed=None, probability=False):
    start = time.perf_counter()
    model = fit_model(model_name, x_train, y_train, x_val, y_val, params=params, seed=seed, probability=probability)
    
    return model, time.perf_counter() - start

def tune_model(model_name, x_train, y_train, x_val, y_val, param_grid=None, seed=None):
    # Grid search on the fixed validation split, selecting by macro F1
    param_grid = param_grid or PARAM_GRIDS[model_name]

    rows = []
    best_model, best_params, best_f1, best_seconds = None, None, -1.0, None

    for params in param_combinations(param_grid):
        model, seconds = timed_fit(model_name, x_train, y_train, x_val, y_val, params=params, seed=seed)
        pred = model.predict(x_val)
        macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)

        rows.append({**params, "Val accuracy": accuracy_score(y_val, pred), "Val macro F1": macro_f1, "Fit seconds": seconds})

        if macro_f1 > best_f1:
            best_model, best_params, best_f1, best_seconds = model, params, macro_f1, seconds

    # Training time of the selected setting, picked up by save_model
    best_model.fit_seconds_ = best_seconds

    return best_model, best_params, pd.DataFrame(rows).sort_values("Val macro F1", ascending=False)

# Splits: one column per split name, value train/val/test per file, rows in the order of the dataset

def splits_table(df, splits):
    table = pd.DataFrame({"file": df["filepath"].map(lambda filepath: Path(filepath).name)})

    for split_name, (train_df, val_df, test_df) in splits.items():
        column = pd.Series("", index=df.index)
        column[train_df.index], column[val_df.index], column[test_df.index] = "train", "val", "test"
        table[split_name] = column

    return table

def get_split(df, table, split_name):
    if not (table["file"].values == df["filepath"].map(lambda filepath: Path(filepath).name).values).all():
        raise ValueError("The saved splits do not match the dataset, re-run the training notebook")

    return tuple(df[table[split_name].values == subset] for subset in ("train", "val", "test"))

# Saved models: scaler + model + how they were made, one joblib file each, listed in a manifest

def save_model(models_dir, experiment, split, feature_set, pooling, model_name, variant, model, scaler, params=None, fit_seconds=None, **info):
    # scaler is anything with transform() applied before the model (a StandardScaler, or a scaler + PCA pipeline).
    # Extra keyword arguments become extra manifest columns
    path = Path(models_dir) / experiment / split / f"{feature_set}_{pooling}_{MODEL_SLUGS[model_name]}_{variant}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)

    return {
        "experiment": experiment,
        "split": split,
        "feature_set": feature_set,
        "pooling": pooling,
        "model": model_name,
        "variant": variant,
        "params": json.dumps(params or {}),
        "fit_seconds": fit_seconds if fit_seconds is not None else getattr(model, "fit_seconds_", None),
        **info,
        "path": str(path.relative_to(models_dir))
    }

def load_model(models_dir, row):
    return joblib.load(Path(models_dir) / row["path"])

def predict(bundle, x):
    return bundle["model"].predict(bundle["scaler"].transform(x))

def predict_proba(bundle, x):
    return bundle["model"].predict_proba(bundle["scaler"].transform(x))

def score_models(manifest, models_dir, df, features, split_table, subsets=("val", "test")):
    # Accuracy and macro F1 of every model in the manifest on its own split
    rows = []

    for _, row in manifest.iterrows():
        bundle = load_model(models_dir, row)
        x = features[row["pooling"]][row["feature_set"]]
        split = dict(zip(("train", "val", "test"), get_split(df, split_table, row["split"])))

        scores = {}
        for subset in subsets:
            y_true = split[subset]["label"].values
            y_pred = predict(bundle, x[split[subset].index])
            scores[f"{subset.capitalize()} accuracy"] = accuracy_score(y_true, y_pred)
            scores[f"{subset.capitalize()} macro F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        rows.append({**row.drop("path").to_dict(), **scores})

    return pd.DataFrame(rows)

def summarize_repeated(results):
    summary = results.groupby("model")[["Test accuracy", "Test macro F1"]].agg(["mean", "std"])

    return summary.reindex([name for name in MODEL_NAMES if name in summary.index])
