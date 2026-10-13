from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm
    }
    