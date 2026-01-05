# ML Algorithms - Walkthrough Index

## What's This?
Real talk about the ML algorithms I built. Not just theory - actual implementation insights, bugs I hit, and solutions that worked. Each walkthrough covers what matters: parameters, when stuff breaks, and how to fix it.

## 📚 The Algorithms

### 1. [K-Nearest Neighbors (KNN)](01_KNN_Walkthrough.md)
**Location**: `models/Classification/knn.py`

**Key Topics**:
- Distance metrics (Euclidean vs. Manhattan)
- Feature scaling necessity
- Vectorized distance computation
- K-fold cross-validation implementation
- Curse of dimensionality
- Batch processing for memory management

**Sticking Points Covered**:
- Feature scaling is mandatory
- Division by zero in distance-weighted voting
- Memory explosion with large distance matrices
- Class imbalance issues
- Computational complexity (O(N×D) per query)

**Best For**: Small-medium datasets, baseline comparisons, interpretable results

---

### 2. [Logistic Regression](02_Logistic_Regression_Walkthrough.md)
**Location**: `models/Classification/logistic_regression.py`

**Key Topics**:
- Sigmoid function mathematics
- Gradient descent implementation
- Feature scaling for convergence
- Binary cross-entropy loss
- Threshold selection for imbalanced data

**Sticking Points Covered**:
- Numerical stability (overflow in exponentials)
- Vanishing/exploding gradients
- Convergence monitoring
- Multicollinearity issues
- Perfect separation problem

**Best For**: Binary classification, probability estimates, interpretability needed

---

### 3. [Random Forest](03_Random_Forest_Walkthrough.md)
**Location**: `models/Classification/random_forest.py`

**Key Topics**:
- Decision tree recursive growth
- Information gain (entropy-based splitting)
- Bootstrap sampling and out-of-bag error
- Feature importance calculation
- Ensemble aggregation (majority voting)

**Sticking Points Covered**:
- Infinite recursion in tree building
- Empty cluster handling
- Feature importance biases
- Memory explosion with many trees
- Threshold selection optimization

**Best For**: High accuracy, handles non-linearity, feature importance analysis

---

### 4. [K-Means Clustering](04_KMeans_Walkthrough.md)
**Location**: `models/Clustering/kmeans.py`

**Key Topics**:
- K-means++ initialization
- Vectorized distance computation
- Empty cluster handling
- Elbow method for optimal K
- Silhouette score

**Sticking Points Covered**:
- Feature scaling is mandatory
- Curse of dimensionality
- Non-spherical cluster failures
- Outlier sensitivity
- Convergence issues

**Best For**: Customer segmentation, data compression, unsupervised learning

---

### 5. [Linear Regression](05_Linear_Regression_Walkthrough.md)
**Location**: `models/Regression/linear_regression.py`

**Key Topics**:
- Gradient descent vs. closed-form solution
- Feature scaling for convergence
- Bias trick (augmenting X with 1s)
- Mean Squared Error loss
- Convergence monitoring

**Sticking Points Covered**:
- Division by zero in scaling
- Exploding/vanishing gradients
- Multicollinearity
- Outlier sensitivity
- Overfitting vs. underfitting

**Best For**: Linear relationships, interpretability, fast predictions

---

### 6. [Polynomial Regression](06_Polynomial_Regression_Walkthrough.md)
**Location**: `models/Regression/polynomial_regression.py`

**Key Topics**:
- Polynomial feature transformation
- Design matrix creation
- Degree selection via cross-validation
- Target variable scaling
- Multicollinearity in polynomial features

**Sticking Points Covered**:
- Feature explosion (combinatorial growth)
- Numerical overflow with high powers
- Extrapolation danger
- Memory explosion
- Overfitting with high degrees

**Best For**: Non-linear but smooth relationships, physics/engineering applications

---

### 7. [Neural Network](07_Neural_Network_Walkthrough.md)
**Location**: `models/NeuralNetwork/neural_network.py`

**Key Topics**:
- Multi-task learning (classification + regression)
- Multiple optimizers (SGD, Adam, RMSprop, etc.)
- Batch normalization
- Dropout regularization
- Bayesian hyperparameter optimization (TPE/GP)
- Various initialization strategies
- Custom loss functions (Huber, sparse cross-entropy)

**Sticking Points Covered**:
- Vanishing/exploding gradients
- Dead ReLU neurons
- Training vs. inference modes
- Weight initialization importance
- NaN loss debugging
- Overfitting/underfitting

**Best For**: Complex patterns, large datasets, state-of-the-art accuracy

---

## 🎯 Common Patterns

### Feature Scaling
- **Need it**: KNN, Logistic Regression, Linear Regression, Polynomial Regression, Neural Networks
- **Don't need it**: Random Forest (but won't hurt)
- **Golden rule**: Always use training statistics for test data!

### Overfitting vs. Underfitting
Just remember: Cross-validation is your friend. If train good but test bad = overfitting. If both bad = underfitting.

### Vectorization
All my code uses NumPy properly - no loops! It's 10-1000x faster but yeah, takes a minute to wrap your head around matrix operations.

### Hyperparameter Tuning
Start simple. Get it working. THEN tune. Don't spend hours optimizing before you have a baseline.

