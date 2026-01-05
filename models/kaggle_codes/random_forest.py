import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os

def load_data():
    base_path = r'C:\Users\nprra\Desktop\HackTech\CP\Python\ML\Data'
    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def main():
    print("Loading data...")
    try:
        train_df, test_df = load_data()
    except Exception as e:
        print(e)
        return

    # 1. data Cleaning & Preparation
    # check for missing values (though inspection showed none, good practice to handle)
    # split Features and Target
    if 'id' in train_df.columns:
        X = train_df.drop(columns=['id', 'target'])
    else:
        X = train_df.drop(columns=['target'])
    
    y = train_df['target']
    
    # prepare Submission Data
    if 'id' in test_df.columns:
        test_ids = test_df['id']
        X_submit = test_df.drop(columns=['id'])
    else:
        test_ids = test_df.index
        X_submit = test_df

    # align columns just in case
    X_submit = X_submit[X.columns]

    print(f"Data Loaded. Shape: {X.shape}")

    # 2. split Data: Train, Validation, Test (Local Holdout)
    # first split: 80% Train+Val, 20% Local Test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # second split: 80% of Temp -> Train, 20% of Temp -> Validation
    # effective: ~64% Train, ~16% Val, 20% Test
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    print(f"Split sizes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 3. pipeline Construction
    # imputer: Handle any missing values
    # scaler: Normalize features (important for many models, helpful for RF convergence speed/stability)
    # classifier: Random Forest
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        # n_jobs=1 helps avoiding potential deadlocks in some environments during search
        ('rf', RandomForestClassifier(random_state=42, n_jobs=None))
    ])

    # 4. hyperparameter Tuning (RandomizedSearch)
    # to achieve maximum accuracy, we tune key parameters.
    # note: n_iter is set to 2 for quick verification. 
    # IMPORTANT: Increase n_iter to 20 or 50 for better accuracy results!
    param_dist = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['sqrt', 'log2']
    }

    print("Starting Hyperparameter Tuning...")
    search = RandomizedSearchCV(
        pipeline, 
        param_dist, 
        n_iter=2,  # set to 20+ for production
        cv=3, 
        verbose=2, 
        random_state=42, 
        n_jobs=None, # avoid nested parallelism issues
        scoring='accuracy'
    )
    
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best Params: {search.best_params_}")
    print(f"Best CV Score: {search.best_score_:.4f}")

    # 5. evaluation
    print("\n--- Evaluation ---")
    
    # validation set
    val_preds = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # test set (Local Holdout)
    test_preds = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Local Test Accuracy: {test_acc:.4f}")
    
    # detailed Report
    print("\nClassification Report (Local Test):")
    print(classification_report(y_test, test_preds))

    # 6. final Submission
    # retrain on ALL available labeled data with best params
    print("Retraining on full dataset for submission...")
    best_model.fit(X, y)
    
    final_preds = best_model.predict(X_submit)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'target': final_preds
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to 'submission.csv'")

if __name__ == "__main__":
    main()
