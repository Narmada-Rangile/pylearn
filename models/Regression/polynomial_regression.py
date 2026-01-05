import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree=1, learning_rate=0.001, iterations=1000):
        # if degree=1, acts as standard multivariate linear regression.
        
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.feature_means = None
        self.feature_stds = None
        
    def create_design_matrix(self, X):
        
           
        # ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # start with bias column ones.
        design_matrix = [np.ones((n_samples, 1))]
        
        # add basic features degree 1
        design_matrix.append(X)
        for d in range(2, self.degree + 1):
            design_matrix.append(X ** d)
            
        return np.column_stack(design_matrix)

    def fit(self, X, y):
       
        X = np.array(X)
        y = np.array(y)
            
        # create design matrix
        X_design = self.create_design_matrix(X)
        
        self.feature_means = np.mean(X_design[:, 1:], axis=0)
        self.feature_stds = np.std(X_design[:, 1:], axis=0)
        
        # avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0
        
        X_design[:, 1:] = (X_design[:, 1:] - self.feature_means) / self.feature_stds
        
        n_samples, n_features = X_design.shape
        
        self.weights = np.zeros(n_features)
        
        # gradient Descent
        for i in range(self.iterations):
            y_pred = X_design @ self.weights
            
            error = y_pred - y
            
            gradient = (1 / n_samples) * (X_design.T @ error)
            
            self.weights -= self.learning_rate * gradient
        
    def predict(self, X):
        X = np.array(X)
        X_design = self.create_design_matrix(X)
        
        if self.feature_means is not None:
            X_design[:, 1:] = (X_design[:, 1:] - self.feature_means) / self.feature_stds
        
        return X_design @ self.weights


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_custom(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main():
    # load Data
    train_path = r'data\poly_train.csv' 
    test_path = r'data\poly_test.csv'
    
    print(f"Loading data from {train_path}...")
    df_train_full = pd.read_csv(train_path)
    df_test_blind = pd.read_csv(test_path) 
    
    # preprocessing
    feature_cols = [c for c in df_train_full.columns if c != 'y']
    target_col = 'y'
    
    # Handle NaNs in training data
    if df_train_full.isnull().values.any():
        print(f"Warning: Found {df_train_full.isnull().sum().sum()} missing values in training data. Dropping rows.")
        df_train_full = df_train_full.dropna()
        
    X = df_train_full[feature_cols].values
    y = df_train_full[target_col].values

    # split Data
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * 0.2)
    
    train_idx = indices[test_size:]
    val_idx = indices[:test_size]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # feature Scaling 
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # avoid division by zero
    std[std == 0] = 1.0
    
    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    
    # scale Target
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std
    
    # train Model
    print("Training Polynomial Regression Model.")
    
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    best_degree = 1
    best_rmse = float('inf')
    best_model = None
    
    for d in degrees:

        model = PolynomialRegression(degree=d)
        model.fit(X_train_scaled, y_train_scaled)
        
        preds_val_scaled = model.predict(X_val_scaled)
        preds_val = preds_val_scaled * y_std + y_mean
        
        error = rmse(y_val, preds_val)
        score = r2_score_custom(y_val, preds_val)
        
        
        if error < best_rmse:
            best_rmse = error
            best_degree = d
            best_model = model
            

    print(f"\nBest Model Degree: {best_degree} with RMSE: {best_rmse:.4f}")
    
    # check on Test Data
    print(f"\nGenerating predictions for {test_path}")
    X_test_blind = df_test_blind[feature_cols].values
    
    # handle NaNs in Test Data, Impute with Mean from Train.
    if np.isnan(X_test_blind).any():
        col_mean = np.nanmean(X_test_blind, axis=0)
        inds = np.where(np.isnan(X_test_blind))
        X_test_blind[inds] = np.take(mean, inds[1]) 
        
        for i in range(X_test_blind.shape[1]):
            col = X_test_blind[:, i]
            col[np.isnan(col)] = mean[i]
            
    # apply same scaling
    X_test_scaled = (X_test_blind - mean) / std
    
    y_test_pred_scaled = best_model.predict(X_test_scaled)
    y_test_pred = y_test_pred_scaled * y_std + y_mean
    
    pd.DataFrame({'prediction': y_test_pred}).to_csv('data/poly_test_pred.csv', index=False)
    print("Test predictions saved to Data/poly_test_pred.csv")

if __name__ == "__main__":
    main()
