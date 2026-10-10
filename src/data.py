from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

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


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    
    rows = []
    
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_short_label = class_dir.name
        if class_short_label not in LABEL_MAP:
            continue
        
        for filepath in class_dir.glob("*.wav"):
            rows.append({
                "filepath": str(filepath),
                "label": LABEL_MAP[class_short_label],
                "label_short": class_short_label
            })
        
    return pd.DataFrame(rows)

def split_dataset(df, seed=42):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    
    val_df, test_df =  train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)
    
    return train_df, val_df, test_df