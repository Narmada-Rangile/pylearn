import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        self.std[self.std == 0] = 1
        X_scaled = (X - self.mean) / self.std
            
        n_samples, n_features = X_scaled.shape
        ones = np.ones((n_samples, 1))
        X_b = np.c_[ones, X_scaled]
        
        # initialize weights (n_features + 1 for bias)
        self.weights = np.zeros(n_features + 1)
        
        # gradient Descent
        for i in range(self.n_iterations):
            gradients = (2/n_samples) * X_b.T @ (X_b @ self.weights - y)
            self.weights -= self.learning_rate * gradients
        
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted.")
            
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # apply scaling using training statistics
        X_scaled = (X - self.mean) / self.std
            
        n_samples = X_scaled.shape[0]
        ones = np.ones((n_samples, 1))
        X_b = np.c_[ones, X_scaled]
        
        return X_b @ self.weights

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_custom(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main():
    train_path = r'C:\Users\nprra\Desktop\HackTech\CP\Python\ML\Data\Linear Regression Test.csv'
    test_path = r'C:\Users\nprra\Desktop\HackTech\CP\Python\ML\Data\Linear Regression Test.csv' 
    
    print(f"Loading data from {train_path}.")
    
    df_train = pd.read_csv(train_path)
    
    target_col = 'target' 
    if target_col not in df_train.columns:
        target_col = df_train.columns[-1]
    
    
    feature_cols = [c for c in df_train.columns if c != target_col]
    
    X = df_train[feature_cols].values
    y = df_train[target_col].values
    
    # train Model
    print("Training Linear Regression Model on full training set.")
    model = LinearRegression()
    model.fit(X, y)
    
    # evaluate on Train Data
    y_pred_train = model.predict(X)
    train_rmse = rmse(y, y_pred_train)
    train_r2 = r2_score_custom(y, y_pred_train)
    
    print(f"\nEvaluation on Training Data ({train_path})")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R2 Score: {train_r2:.4f}")
    

    print(f"\nLoading Test Data from {test_path}...")
    try:
            df_test = pd.read_csv(test_path)
    except FileNotFoundError:
            df_test = pd.read_csv(f'../{test_path}')
            
    # ensure features match
    X_test = df_test[feature_cols].values
    
        
    y_pred_test = model.predict(X_test)
    print("Predictions generated for test set.")
    print("All Predictions:", y_pred_test)
    
    # save predictions
    pd.DataFrame({'prediction': y_pred_test}).to_csv(r'C:\Users\nprra\Desktop\HackTech\CP\Python\ML\Data\linear_regression_test_pred.csv', index=False)
    print("Test predictions saved to Data/linear_regression_test_pred.csv")
    
    # save Train predictions as well
    pd.DataFrame({'prediction': y_pred_train}).to_csv(r'C:\Users\nprra\Desktop\HackTech\CP\Python\ML\Data\linear_regression_train_pred.csv', index=False)
    print("Train predictions saved to Data/linear_regression_train_pred.csv")

if __name__ == "__main__":
    main()
