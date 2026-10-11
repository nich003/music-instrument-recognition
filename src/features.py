import librosa
import numpy as np

def load_audio(filepath):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    return y, sr

def extract_frame_features(filepath, n_mfcc=13, n_fft=2048, hop_length=512):
    y, sr = load_audio(filepath)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    
    return {
        "mfcc": mfcc,
        "spectral_centroid": spectral_centroid,
        "zcr": zcr
    }
    
def temporal_pooling(feature, pooling="mean"):
    if pooling == "mean":
        return np.mean(feature, axis=1)
    elif pooling == "mean_std":
        mean = np.mean(feature, axis=1)
        std = np.std(feature, axis=1)
        
        return np.concatenate([mean, std])
    else:
        raise ValueError(f"Invalid pooling strategy: {pooling}, select mean or mean_std")

def extract_feature_vector(filepath, selected_feature="combined", pooling="mean", n_mfcc=13, n_fft=2048, hop_length=512):
    features = extract_frame_features(filepath, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = temporal_pooling(features["mfcc"], pooling)
    centroid = temporal_pooling(features["spectral_centroid"], pooling)
    zcr = temporal_pooling(features["zcr"], pooling)
    if selected_feature == "mfcc":
        return mfcc
    elif selected_feature == "audio":
        return np.concatenate([centroid, zcr])
    elif selected_feature == "combined":
        return np.concatenate([mfcc, centroid, zcr])
    else: 
        raise ValueError(f"Select a valid feature set: {selected_feature}")
    
def build_feature_matrix(df, selected_feature="mfcc", pooling="mean", n_mfcc=13, n_fft=2048, hop_length=512):
    
    x = []
    y = []
    
    for _, row in df.iterrows():
        feature_vector = extract_feature_vector(
            row["filepath"],
            selected_feature=selected_feature,
            pooling=pooling,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        x.append(feature_vector)
        y.append(row["label"])
        
    return np.vstack(x), np.array(y)
