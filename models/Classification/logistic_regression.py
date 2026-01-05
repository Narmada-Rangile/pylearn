import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # feature Scaling 
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1 # Avoid division by zero
        X = (X - self.mean) / self.std
        
        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient Descent
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict_proba(self, X):
            
        # apply scaling
        X = (X - self.mean) / self.std
        
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i > threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def r2_score_custom(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main():
    # paths
    train_path = 'data/train_binary.csv'
    test_path = 'data/test_binary.csv'
    
    print(f"Loading training data from {train_path}...")
    
    df_train = pd.read_csv(train_path)
    
    # features and labels
    target_col = 'label'
    if target_col not in df_train.columns:
        target_col = df_train.columns[-1]
    
    feature_cols = [c for c in df_train.columns if c != target_col]
    
    X = df_train[feature_cols].values
    y = df_train[target_col].values
    
    # train/Validation Split (80/20)
    indices = np.random.permutation(len(X)) 
    split_idx = int(len(X) * 0.8)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Training Logistic Regression (Samples: {len(X_train)})...")
    clf = LogisticRegression(learning_rate=0.01, n_iterations=2000)
    clf.fit(X_train, y_train)
    
    # evaluate on Validation Set
    y_pred_val = clf.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred_val)
    r2 = r2_score_custom(y_val, y_pred_val)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # process Test Data
    df_test = pd.read_csv(test_path)
    X_test = df_test[feature_cols].values
    
    # predictions
    y_pred_test = clf.predict(X_test)
    
    output_path = 'data/lr__test_pred.csv'
    
    pd.DataFrame({'prediction': y_pred_test}).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
