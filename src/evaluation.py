import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             average_precision_score, label_ranking_average_precision_score, precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer

INSTRUMENT_FAMILIES = {
    "cello": "strings",
    "violin": "strings",
    "clarinet": "woodwinds",
    "flute": "woodwinds",
    "saxophone": "woodwinds",
    "trumpet": "brass",
    "acoustic_guitar": "guitars",
    "electric_guitar": "guitars",
    "organ": "keyboards",
    "piano": "keyboards",
    "voice": "voice"
}

def evaluate_model(y_true, y_pred, labels=None, verbose=True):
    # Fixed (sorted) label order so the confusion matrix rows/columns are known
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print()
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_per_class": {label: report[label]["precision"] for label in labels},
        "recall_per_class": {label: report[label]["recall"] for label in labels},
        "f1_per_class": {label: report[label]["f1-score"] for label in labels},
        "confusion_matrix": cm,
        "labels": labels
    }

def per_class_metrics(results):
    return pd.DataFrame({
        "Precision": results["precision_per_class"],
        "Recall": results["recall_per_class"],
        "F1": results["f1_per_class"],
        "Support": dict(zip(results["labels"], results["confusion_matrix"].sum(axis=1)))
    })

def plot_confusion_matrix(results, normalize=False, title=None, ax=None, save_path=None):
    cm = results["confusion_matrix"]

    # Row-normalized: each row shows how the true class is distributed over the predictions (diagonal = recall)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(cm, display_labels=results["labels"])
    display.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", xticks_rotation=45, colorbar=False)

    if title:
        ax.set_title(title)

    if save_path:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax

def plot_loss_curves(train_losses, val_losses, title=None, ax=None, save_path=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train")
    ax.plot(epochs, val_losses, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()

    if title:
        ax.set_title(title)

    if save_path:
        ax.figure.tight_layout()
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax

def top_confusions(results, k=10):
    # Largest off-diagonal cells of the confusion matrix
    cm = results["confusion_matrix"]
    labels = results["labels"]
    
    rows = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "True": true_label,
                    "Predicted": predicted_label,
                    "Count": cm[i, j],
                    "Share of true class": cm[i, j] / cm[i].sum(),
                    "Same family": INSTRUMENT_FAMILIES.get(true_label) == INSTRUMENT_FAMILIES.get(predicted_label)
                })
    
    return pd.DataFrame(rows).sort_values("Count", ascending=False).head(k).reset_index(drop=True)

def family_error_share(results):
    # Share of all errors that stay within the instrument family (e.g. violin -> cello).
    # With random errors this would be close to the share of same-family wrong labels, which is printed for reference
    cm = results["confusion_matrix"]
    families = [INSTRUMENT_FAMILIES.get(label) for label in results["labels"]]
    same_family = np.array([[a == b for b in families] for a in families])
    off_diagonal = ~np.eye(len(families), dtype=bool)
    
    errors = cm[off_diagonal].sum()
    within_family = cm[same_family & off_diagonal].sum()
    
    return {
        "errors": int(errors),
        "within_family_share": within_family / max(errors, 1),
        "chance_within_family_share": (same_family & off_diagonal).sum() / off_diagonal.sum()
    }

def aggregate_window_probabilities(window_probabilities, file_index, n_files):
    # Average the class probabilities of all windows of a file
    n_classes = window_probabilities.shape[1]
    sums = np.zeros((n_files, n_classes))
    np.add.at(sums, file_index, window_probabilities)
    counts = np.bincount(file_index, minlength=n_files)[:, None]
    
    return sums / np.maximum(counts, 1)

def evaluate_multilabel(true_label_lists, clip_scores, classes, thresholds=np.round(np.arange(0.1, 1.0, 0.1), 2)):
    # Multi-label evaluation for the official IRMAS test set.
    # An instrument is predicted when its score is at least threshold * the highest score of the clip,
    # so the top instrument is always predicted. Threshold-free ranking metrics are reported as well
    y_true = MultiLabelBinarizer(classes=classes).fit_transform(true_label_lists)
    relative_scores = clip_scores / clip_scores.max(axis=1, keepdims=True)
    
    rows = []
    for threshold in thresholds:
        y_pred = (relative_scores >= threshold).astype(int)
        rows.append({
            "Threshold": threshold,
            "Micro F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Micro precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "Micro recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "Labels per clip": y_pred.sum(axis=1).mean()
        })
    
    summary = {
        "Top-1 precision": y_true[np.arange(len(y_true)), clip_scores.argmax(axis=1)].mean(),
        "LRAP": label_ranking_average_precision_score(y_true, clip_scores),
        "Macro AP": average_precision_score(y_true, clip_scores, average="macro"),
        "True labels per clip": y_true.sum(axis=1).mean()
    }
    
    return summary, pd.DataFrame(rows)


def group_permutation_importance(predict, x, y, column_groups, n_repeats=5, seed=0):
    # Drop in macro F1 when the columns of one feature group are shuffled across excerpts (jointly, so the
    # group's internal structure is kept but its relation to the label is destroyed). Works for any model
    rng = np.random.default_rng(seed)
    column_groups = np.asarray(column_groups)
    baseline = f1_score(y, predict(x), average="macro", zero_division=0)
    
    rows = []
    for group in dict.fromkeys(column_groups):
        columns = np.flatnonzero(column_groups == group)
        drops = []
        for _ in range(n_repeats):
            x_permuted = x.copy()
            x_permuted[:, columns] = x[rng.permutation(len(x))][:, columns]
            drops.append(baseline - f1_score(y, predict(x_permuted), average="macro", zero_division=0))
        rows.append({"Group": group, "Macro F1 drop": np.mean(drops), "Std": np.std(drops)})
    
    return pd.DataFrame(rows)
