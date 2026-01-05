import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'Data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    
    # Preprocessing Training Data
    X = train_df.iloc[:, :-1].values    
    y = train_df.iloc[:, -1].values
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Validate
    val_preds = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_preds)
    print(f"Accuracy: {acc:.4f}")
    
    # Predict on Test Data
    test_df = pd.read_csv(test_path)
    
    X_test = test_df.values
    X_test_scaled = scaler.transform(X_test)
    test_preds = model.predict(X_test_scaled)
    y_test = test_df.iloc[:, -1].values

    output_path = os.path.join(script_dir, 'sub2.csv')
    pd.DataFrame({'id': test_df.iloc[:, 0], 'Target': test_preds}).to_csv(output_path,)
    print(f"Predictions saved to {output_path}")
   
if __name__ == "__main__":
    main()
