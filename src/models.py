from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

# implementing pytorch mlp here later