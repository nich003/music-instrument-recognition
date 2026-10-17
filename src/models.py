import copy

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def fit_scaler(x_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    return x_train_scaled, scaler

def train_logistic_regression(x_train, y_train, C=1.0):
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(x_train, y_train)

    return model

def train_svm(x_train, y_train, C=1.0, gamma="scale", probability=False, seed=None):
    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)

    # Probability estimates by Platt scaling (sigmoid) fit on 5-fold cross-validated decision values, plus one SVM
    # trained on all data (replaces SVC(probability=True), deprecated in scikit-learn 1.9)
    if probability:
        model = CalibratedClassifierCV(model, method="sigmoid", cv=5, ensemble=False)

    model.fit(x_train, y_train)

    return model

class KNNClassifier:
    # KNeighborsClassifier on integer-encoded labels: scikit-learn 1.8 fails to predict string labels
    # with uniform weights and Manhattan distance (its fast path casts the labels to integers)
    def __init__(self, n_neighbors=5, weights="uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
    
    def fit(self, x_train, y_train):
        self.label_encoder_ = LabelEncoder().fit(y_train)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)
        self.model_.fit(x_train, self.label_encoder_.transform(y_train))
        
        return self
    
    def predict_proba(self, x):
        return self.model_.predict_proba(x)
    
    def predict(self, x):
        return self.classes_[self.model_.predict(x)]

def train_knn(x_train, y_train, n_neighbors=5, weights="uniform", p=2):
    # Instance-based: "training" only stores the (scaled) training set; p=2 is Euclidean, p=1 Manhattan distance
    return KNNClassifier(n_neighbors=n_neighbors, weights=weights, p=p).fit(x_train, y_train)

# Basic MLP
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=11, hidden_dims=(64, 32), dropout=0.0):
        super().__init__()

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_mlp(x_train, y_train, x_val, y_val, input_dim, num_classes=11, epochs=100, batch_size=64, learning_rate=1e-3,
              hidden_dims=(64, 32), dropout=0.0, weight_decay=0.0, patience=None, seed=None):
    # seed makes initialization and batch order reproducible.
    # patience enables early stopping on the validation loss; the weights of the best epoch are restored
    if seed is not None:
        torch.manual_seed(seed)

    model = MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=dropout)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # Training
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            logits = model(x_batch)

            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses

# Prediction using the MLP
def predict_mlp(model, x):
    model.eval()

    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

    return preds.numpy()

def predict_proba_mlp(model, x):
    model.eval()

    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)

    return probabilities.numpy()

# sklearn-style wrapper so the MLP can be tuned and evaluated with the same code as the sklearn models
class TorchMLPClassifier:
    def __init__(self, hidden_dims=(64, 32), dropout=0.0, weight_decay=0.0, learning_rate=1e-3, epochs=100, batch_size=64, patience=None, seed=None):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def fit(self, x_train, y_train, x_val, y_val):
        self.label_encoder_ = LabelEncoder().fit(y_train)
        self.classes_ = self.label_encoder_.classes_

        self.model_, self.train_losses_, self.val_losses_ = train_mlp(
            x_train, self.label_encoder_.transform(y_train),
            x_val, self.label_encoder_.transform(y_val),
            input_dim=x_train.shape[1],
            num_classes=len(self.classes_),
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            patience=self.patience,
            seed=self.seed
        )

        return self

    def predict_proba(self, x):
        return predict_proba_mlp(self.model_, x)

    def predict(self, x):
        return self.classes_[predict_mlp(self.model_, x)]

MODEL_NAMES = ["Logistic Regression", "kNN", "SVM", "MLP"]

# Defaults are the settings of the first experiments, the baseline the tuned models are compared against
DEFAULT_PARAMS = {
    "Logistic Regression": {"C": 1.0},
    "kNN": {"n_neighbors": 5, "weights": "uniform", "p": 2},
    "SVM": {"C": 1.0, "gamma": "scale"},
    "MLP": {"hidden_dims": (64, 32), "dropout": 0.0, "weight_decay": 0.0, "epochs": 100, "patience": None}
}

def fit_model(model_name, x_train, y_train, x_val, y_val, params=None, seed=None, probability=False):
    # Common entry point for the three models; x_val/y_val are only used by the MLP (validation loss, early stopping)
    params = {**DEFAULT_PARAMS[model_name], **(params or {})}

    if model_name == "Logistic Regression":
        return train_logistic_regression(x_train, y_train, **params)
    elif model_name == "kNN":
        return train_knn(x_train, y_train, **params)
    elif model_name == "SVM":
        return train_svm(x_train, y_train, probability=probability, seed=seed, **params)
    elif model_name == "MLP":
        return TorchMLPClassifier(seed=seed, **params).fit(x_train, y_train, x_val, y_val)
    else:
        raise ValueError(f"Invalid model: {model_name}, select one of {MODEL_NAMES}")
