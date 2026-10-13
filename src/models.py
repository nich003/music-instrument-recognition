from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def fit_scaler(x_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    
    return x_train_scaled, scaler

def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    
    return model

def train_svm(x_train, y_train, C=1.0, gamma="scale"):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    model.fit(x_train, y_train)
    
    return model

# Basic MLP
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=11):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)
    
def train_mlp(x_train, y_train, x_val, y_val, input_dim, num_classes=11, epochs=100, batch_size=64, learning_rate=1e-3):
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
        
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    
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
            
    return model, train_losses, val_losses

