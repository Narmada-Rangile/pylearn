import matplotlib
matplotlib.use('Agg') # Set non-interactive backend immediately
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
import traceback
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, classification_report

def load_data():
    """Loads train and test datasets."""
    print("Loading data...")
    train_df = pd.read_csv('Data/train.csv')
    test_df = pd.read_csv('Data/test.csv')
    return train_df, test_df

def get_target_column(df):
    """Infers the target column name."""
    if 'target' in df.columns:
        return 'target'
    elif 'label' in df.columns:
        return 'label'
    elif 'y' in df.columns:
        return 'y'
    else:
        # assume last column
        return df.columns[-1]

def create_pipeline(numeric_features, categorical_features):
    """Creates the preprocessing and model pipeline."""
    
    # numeric transformer: Impute with median
    numeric_transformer = SimpleImputer(strategy='median')
    
    # categorical transformer: Impute with most frequent + Ordinal Encoder
    categoricals_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categoricals_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # XGBoost Classifier
    # set n_jobs=1 because GridSearchCV will handle parallelization
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=42,
        n_jobs=1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf)
    ])
    
    return pipeline

def main():
    try:
        # 1. load Data
        train_df, test_df = load_data()
        
        # 2. prepare Feature/Target split
        target_col = get_target_column(train_df)
        print(f"Inferred target column: {target_col}")
        
        X = train_df.drop(columns=[target_col])
        y = train_df[target_col]
        
        # handle ID columns
        if 'id' in X.columns:
            X = X.drop(columns=['id'])
        
        test_ids = None
        if 'id' in test_df.columns:
            test_ids = test_df['id']
            X_test_predict = test_df.drop(columns=['id'])
        else:
            test_ids = pd.Series(range(len(test_df)), name='id')
            X_test_predict = test_df
            
        # align columns
        X_test_predict = X_test_predict[X.columns]

        # identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        print(f"Numerical features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")

        # 3. create Pipeline
        pipeline = create_pipeline(numeric_features, categorical_features)
        
        # 4. hyperparameter Tuning (Grid Search)
        param_grid = {
            'classifier__max_depth': [3, 6, 10],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__n_estimators': [100, 300],
            'classifier__subsample': [0.8],
            'classifier__colsample_bytree': [0.8]
        }
        
        print(f"Starting Grid Search with scoring='f1'...")
        # use n_jobs=-1 here to parallelize the folds
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            verbose=1, # Reduce verbosity slightly
            n_jobs=-1
        )
        
        # simple split for final holdout check
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
        
        print("Fitting Grid Search on training split...")
        grid_search.fit(X_train, y_train)
        
        print("Best params found:")
        print(grid_search.best_params_)
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        
        # 5. evaluate on Holdout
        print("Evaluating on internal Holdout set...")
        y_pred = best_model.predict(X_holdout)
        final_f1 = f1_score(y_holdout, y_pred)
        print(f"Holdout F1 Score: {final_f1:.4f}")
        print("\nClassification Report:\n", classification_report(y_holdout, y_pred))
        
        # 6. final Training: Refit on ALL data (X, y)
        print("Refitting best model on full training data...")
        best_model.fit(X, y)
        
        # 7. predict on Test Data
        print("Generating predictions for Data/test.csv...")
        predictions = best_model.predict(X_test_predict)
        
        # 8. save Submission
        submission_df = pd.DataFrame({
            'id': test_ids,
            'target': predictions
        })
        submission_path = 'submission_XGBoost.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Predictions saved to {submission_path}")
        
        # 9. feature Importance
        classifier = best_model.named_steps['classifier']
        
        try:
            # ordinal encoder preserves columns 1-to-1, so we can concatenate names
            if len(categorical_features) > 0:
                feature_names = list(numeric_features) + list(categorical_features)
            else:
                feature_names = list(numeric_features)
                
            importance = classifier.feature_importances_
            
            # if lengths don't match, fallback to indices (safeguard)
            if len(importance) != len(feature_names):
                print(f"Warning: Feature names count ({len(feature_names)}) != Importance count ({len(importance)}). Using indices.")
                feature_names = [f"f{i}" for i in range(len(importance))]

            indices = np.argsort(importance)[::-1][:20]
            
            plt.figure(figsize=(10, 8))
            plt.title("Feature Importances (Top 20)")
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("Feature importance plot saved to feature_importance.png")
            
        except Exception as e:
            print(f"Could not generate feature importance plot: {e}")
            traceback.print_exc()

    except Exception:
        print("An error occurred during pipeline execution:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
