# Logistic Regression - Walkthrough

## Overview
Despite the name, this is for classification, not regression! It uses the sigmoid function to squish predictions between 0 and 1, giving you probabilities. Linear decision boundary but works surprisingly well.

## Key Parameters

### 1. **learning_rate**
- **Default**: 0.01 (good starting point)
- **Too small**: Takes forever to converge
- **Too large**: Bounces around, never settles
- **Pro tip**: Start at 0.01, watch the loss curve. If it's oscillating, lower it. If it's barely moving, increase it.

### 2. **n_iterations**
- **Type**: Integer
- **Default**: 1000
- **Range**: 100 to 10,000+
- **Purpose**: Number of gradient descent update steps
- **When to increase**:
  - Loss still decreasing after max iterations
  - Complex datasets with many features
  - Small learning rate chosen
- **When to decrease**:
  - Loss plateaus early
  - Overfitting concerns
  - Time constraints

---

## How It Works

### The Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

This S-shaped curve maps anything to 0-1, which we interpret as probabilities. That's it!

**Cool property**: The derivative is super clean: `σ'(z) = σ(z) × (1 - σ(z))`
This makes gradient descent really efficient.

---

## What Logistic Regression Can Do

### Primary Use Cases
1. **Binary Classification**: Yes/No, True/False, 0/1 predictions
2. **Probability Estimation**: Get confidence scores (0 to 1)
3. **Linear Decision Boundaries**: Separates classes with hyperplane
4. **Multi-class (with extensions)**: One-vs-Rest, One-vs-One strategies

### Examples
- **Spam Detection**: Email is spam or not
- **Medical Diagnosis**: Disease present or absent
- **Credit Approval**: Approve or reject loan
- **Click Prediction**: User will click ad or not
- **Churn Prediction**: Customer will leave or stay

### Advantages
✅ **Interpretable**: Weights show feature importance
✅ **Probabilistic**: Outputs meaningful probabilities
✅ **Fast Training**: Linear complexity in features
✅ **Low Memory**: Only stores weights (not data)
✅ **Regularization-friendly**: Easy to add L1/L2 penalties
✅ **No Hyperparameter Sensitivity**: Fewer knobs than tree models

### Disadvantages
❌ **Linear Decision Boundary**: Can't capture complex patterns
❌ **Feature Engineering Required**: Need to create polynomial/interaction terms
❌ **Assumes Independence**: Features should be relatively uncorrelated
❌ **Outlier Sensitive**: Can skew the decision boundary
❌ **Binary Only (basic)**: Needs modifications for multi-class

---

## Critical Implementation Insights

### 1. **Feature Scaling is Essential**

**Why Needed**: Gradient descent converges faster when features on similar scales

**Problem Without Scaling**:
```python
# Feature 1: Age (0-100), Feature 2: Income (0-1,000,000)
# Gradients will be vastly different magnitudes
# Weight updates become inefficient
```

**Visual Analogy**:
- Unscaled: Navigating an elongated valley (slow, zigzag path)
- Scaled: Navigating a circular bowl (direct path to minimum)

**Implementation**:
```python
def fit(self, X, y):
    # Critical: Scale during training
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    self.std[self.std == 0] = 1  # Prevent division by zero!
    X = (X - self.mean) / self.std
```

**Key Insight**: Store mean/std during training, apply to test data!

**Common Mistake**:
```python
# WRONG: Scaling train and test separately
X_train_scaled = (X_train - X_train.mean()) / X_train.std()
X_test_scaled = (X_test - X_test.mean()) / X_test.std()  # ❌ Data leakage!

# CORRECT: Use training statistics for both
X_train_scaled = (X_train - X_train.mean()) / X_train.std()
X_test_scaled = (X_test - train_mean) / train_std  # ✅
```

---

### 2. **Gradient Descent Implementation**

**The Update Rule**:
```python
# Forward pass
linear_model = np.dot(X, self.weights) + self.bias
y_predicted = self.sigmoid(linear_model)

# Compute gradients
dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
db = (1 / n_samples) * np.sum(y_predicted - y)

# Update parameters
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```

**Derivation Insight**:
- Loss function: Binary Cross-Entropy L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
- Derivative with respect to weights: ∂L/∂w = (ŷ - y) · x
- The sigmoid's derivative cancels beautifully with the log loss derivative!

**Vectorization Power**:
```python
# Naive loop (SLOW): O(n² × d)
for i in range(n_samples):
    for j in range(n_features):
        dw[j] += (y_pred[i] - y[i]) * X[i, j]

# Vectorized (FAST): O(n × d)
dw = np.dot(X.T, (y_predicted - y)) / n_samples
# 10-100x faster!
```

---

### 3. **Handling Numerical Instability**

**Problem**: Exponential function overflow/underflow

**Scenario**:
```python
z = -1000  # Very negative
np.exp(-z) = np.exp(1000) = inf  # Overflow!
sigmoid(z) = 1 / (1 + inf) = 0  # But this should work!
```

**Solution** (not in current code, but important):
```python
def stable_sigmoid(z):
    # Avoid overflow by using negative values
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),           # For positive z
        np.exp(z) / (1 + np.exp(z))     # For negative z
    )
```

**Alternative**: Clip extreme values
```python
z = np.clip(z, -500, 500)  # Prevent extreme values
```

---

### 4. **Threshold Selection**

**Default Threshold**: 0.5
```python
def predict(self, X, threshold=0.5):
    y_predicted_cls = [1 if i > threshold else 0 
                       for i in self.predict_proba(X)]
    return np.array(y_predicted_cls)
```

**When to Adjust**:
1. **Imbalanced Classes**: Use precision-recall curve to find optimal threshold
2. **Cost-Sensitive**: Higher cost for false negatives? Lower threshold
3. **F1 Optimization**: Maximize F1 score by tuning threshold

**Example**:
```python
# For rare disease detection (false negative = fatal)
predictions = model.predict(X_test, threshold=0.3)  # More sensitive

# For spam detection (false positive = annoying)
predictions = model.predict(X_test, threshold=0.7)  # More specific
```

---

### 5. **Convergence Monitoring**

**Problem**: How do we know when to stop?

**Solutions**:

**1. Loss Tracking** (Best Practice):
```python
for i in range(self.n_iterations):
    # ... gradient descent step ...
    
    # Compute loss
    loss = -np.mean(y * np.log(y_predicted + 1e-15) + 
                    (1-y) * np.log(1-y_predicted + 1e-15))
    
    # Early stopping
    if abs(prev_loss - loss) < 1e-6:
        print(f"Converged at iteration {i}")
        break
    prev_loss = loss
```

**2. Weight Change Tracking**:
```python
weight_change = np.linalg.norm(prev_weights - self.weights)
if weight_change < tolerance:
    break
```

**3. Gradient Magnitude**:
```python
grad_norm = np.linalg.norm(dw)
if grad_norm < 1e-5:
    break
```

**Visualization**:
```python
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Convergence Curve')
# Should show monotonic decrease, then plateau
```

---

## Potential Sticking Points While Coding

### 1. **Vanishing/Exploding Gradients**

**Vanishing** (z very large/small):
```python
# sigmoid(1000) ≈ 1.0
# gradient = sigmoid * (1 - sigmoid) ≈ 1.0 * 0.0 ≈ 0
# Weights stop updating!
```

**Solution**:
- Proper initialization (zero or small random values)
- Feature scaling
- Moderate learning rate
- Batch normalization (for neural networks)

---

### 2. **Perfect Separation**

**Problem**: When data is linearly separable, weights can grow indefinitely

**Example**:
```python
# Data: Class 0 at x < 0, Class 1 at x > 1
# Decision boundary at x = 0.5
# But weights keep increasing to make predictions "more confident"
# w → ∞ to make sigmoid(w*x) → exactly 0 or 1
```

**Solution**: Regularization
```python
# Add L2 penalty to loss
loss += (lambda / 2) * np.sum(self.weights ** 2)

# Update gradients
dw += lambda * self.weights
```

---

### 3. **Non-Convergence**

**Symptom**: Loss oscillates or increases

**Causes**:
1. Learning rate too high
2. Bad initialization
3. Data not scaled
4. Numerical instability

**Debugging**:
```python
# Track loss at each iteration
losses = []
for i in range(n_iterations):
    # ... train ...
    losses.append(current_loss)

# Check pattern
if losses[-1] > losses[0]:
    print("Diverging! Reduce learning rate")
elif np.std(losses[-10:]) > 0.01:
    print("Oscillating! Reduce learning rate")
```

---

### 4. **Initialization Matters**

**Current Implementation**: Zero initialization
```python
self.weights = np.zeros(n_features)
self.bias = 0
```

**Why This Works**: Logistic regression is convex (no local minima)

**Alternative** (neural networks need this):
```python
# Xavier/Glorot initialization
self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
```

---

### 5. **Multicollinearity Issues**

**Problem**: Highly correlated features cause unstable weight estimates

**Example**:
```python
# Feature 1: Height in cm
# Feature 2: Height in inches
# Both measure the same thing!
# Model can't decide how to split weight between them
```

**Detection**:
```python
# Compute correlation matrix
corr_matrix = np.corrcoef(X.T)
# Look for values close to ±1
```

**Solutions**:
1. Remove one of the correlated features
2. PCA (combine into single component)
3. Regularization (L1 for feature selection, L2 for shrinkage)

---

## Performance Optimization

### 1. **Vectorization**

**Before** (Loop - Slow):
```python
for i in range(n_iterations):
    for j in range(n_samples):
        error = y_pred[j] - y[j]
        for k in range(n_features):
            dw[k] += error * X[j, k]
```

**After** (Vectorized - Fast):
```python
for i in range(n_iterations):
    dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
```

**Speedup**: 100-1000x faster!

---

### 2. **Mini-Batch Gradient Descent**

**Current**: Batch gradient descent (uses all data each iteration)

**Alternative**: Mini-batch (subset each iteration)
```python
batch_size = 32
for epoch in range(n_epochs):
    indices = np.random.permutation(n_samples)
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        # ... compute gradients on batch ...
```

**Benefits**:
- Faster iterations
- Can handle datasets larger than memory
- Adds noise → better generalization
- Enables parallelization

---

### 3. **Learning Rate Schedules**

**Step Decay**:
```python
learning_rate = initial_lr * (decay_rate ** (epoch // drop_every))
```

**Exponential Decay**:
```python
learning_rate = initial_lr * np.exp(-decay_rate * epoch)
```

**1/t Decay**:
```python
learning_rate = initial_lr / (1 + decay_rate * epoch)
```

---

## Advanced Extensions

### 1. **Multi-class Logistic Regression**

**One-vs-Rest (OvR)**:
```python
class MultiClassLogistic:
    def __init__(self, n_classes):
        self.classifiers = [LogisticRegression() for _ in range(n_classes)]
    
    def fit(self, X, y):
        for i in range(self.n_classes):
            # Create binary labels: class i vs all others
            y_binary = (y == i).astype(int)
            self.classifiers[i].fit(X, y_binary)
    
    def predict(self, X):
        # Get probabilities from each classifier
        probs = np.array([clf.predict_proba(X) for clf in self.classifiers])
        # Choose class with highest probability
        return np.argmax(probs, axis=0)
```

**Softmax Regression** (Multinomial Logistic):
```python
# Generalization to K classes
# Uses softmax instead of sigmoid
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

---

### 2. **Regularization**

**L2 (Ridge)**:
```python
# Add to loss
loss += (lambda_param / 2) * np.sum(self.weights ** 2)

# Add to gradient
dw += lambda_param * self.weights
```

**L1 (Lasso)**:
```python
# Add to loss
loss += lambda_param * np.sum(np.abs(self.weights))

# Add to gradient
dw += lambda_param * np.sign(self.weights)
```

**Elastic Net** (L1 + L2):
```python
loss += l1_ratio * np.sum(np.abs(w)) + (1-l1_ratio) * np.sum(w**2)
```

**Benefits**:
- Prevents overfitting
- L1 → Feature selection (sparse weights)
- L2 → Weight shrinkage (small but non-zero)

---

### 3. **Polynomial Features**

**Expand feature space** for non-linear boundaries:
```python
from sklearn.preprocessing import PolynomialFeatures

# Create interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Example: [x1, x2] → [x1, x2, x1², x1·x2, x2²]

model.fit(X_poly, y)
```

---

## Debugging Checklist

### Poor Accuracy
- [ ] Features scaled?
- [ ] Learning rate appropriate?
- [ ] Enough iterations?
- [ ] Data linearly separable?
- [ ] Class balance checked?
- [ ] Try polynomial features?

### Non-convergence
- [ ] Reduce learning rate
- [ ] Increase iterations
- [ ] Check for NaN/Inf in data
- [ ] Try different initialization
- [ ] Add regularization

### Overfitting
- [ ] Add regularization
- [ ] Reduce polynomial degree
- [ ] Get more data
- [ ] Cross-validation
- [ ] Feature selection

---

## Evaluation Metrics

### Beyond Accuracy

**1. Confusion Matrix**:
```python
TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
```

**2. Precision**: TP / (TP + FP)
- "Of predicted positives, how many are correct?"

**3. Recall**: TP / (TP + FN)
- "Of actual positives, how many did we find?"

**4. F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean, good for imbalanced data

**5. ROC-AUC**: Area under ROC curve
- Plots TPR vs FPR at various thresholds
- 1.0 = perfect, 0.5 = random

**6. Log Loss**: Measures probability calibration
```python
-np.mean(y * np.log(probs) + (1-y) * np.log(1-probs))
```

---

## Real-World Applications

1. **Medicine**: Disease diagnosis from symptoms/test results
2. **Finance**: Credit scoring, fraud detection
3. **Marketing**: Customer churn prediction, conversion optimization
4. **NLP**: Sentiment analysis, spam filtering
5. **Biology**: Gene expression classification
6. **E-commerce**: Product recommendation (implicit feedback)

---

## Key Takeaways

✅ **Simple & Interpretable**: Understand exactly what model learned
✅ **Probabilistic Output**: Get confidence scores, not just binary
✅ **Fast**: Training and prediction are efficient
✅ **Baseline**: Always try before complex models

❌ **Linear Limitation**: Can't capture complex interactions without feature engineering
❌ **Scaling Required**: Preprocessing is mandatory
❌ **Binary Focus**: Needs extensions for multi-class

**When to Use**:
- Need interpretability
- Linear decision boundary sufficient
- Want probability estimates
- Baseline model
- Small to medium datasets

**When to Avoid**:
- Highly non-linear data
- Complex feature interactions
- No time for feature engineering
- Need automatic feature learning

**Next Steps**:
1. Add regularization
2. Implement learning rate schedules
3. Try mini-batch gradient descent
4. Extend to multi-class
5. Compare with tree-based models
