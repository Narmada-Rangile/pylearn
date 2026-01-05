# Linear Regression - Walkthrough

## Overview
The classic! Fit a straight line (or plane in higher dimensions) through your data. Super interpretable - you can literally see what each feature contributes. Often all you need, and a great baseline even when it's not.

## Parameters

### 1. **learning_rate**
- **Type**: Float
- **Default**: 0.01
- **Range**: 0.0001 to 1.0
- **Purpose**: Controls step size during gradient descent
- **Effect**:
  - **Too small (0.0001)**: Slow convergence, may timeout
  - **Too large (1.0)**: Overshooting, divergence
  - **Optimal (~0.01-0.1)**: Steady convergence to minimum
- **Selection**: Start with 0.01, monitor loss curve

### 2. **n_iterations**
- **Type**: Integer
- **Default**: 1000
- **Range**: 100 to 100,000
- **Purpose**: Number of gradient descent steps
- **When to increase**:
  - Loss still decreasing
  - Large datasets
  - Small learning rate
- **Early stopping**: Monitor convergence, stop when loss plateaus

---

## Mathematical Foundation

### The Linear Model
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
ŷ = β₀ + Xβ  (vector notation)
```

**In Code**:
```python
# With bias trick: add column of 1s to X
X_b = np.c_[np.ones(n_samples), X]
predictions = X_b @ self.weights
```

**Components**:
- **β₀ (bias)**: y-intercept, baseline prediction
- **βᵢ (weights)**: Slope for feature i, feature importance
- **X**: Input features
- **ŷ**: Predicted output

---

### Loss Function: Mean Squared Error

**Formula**:
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Why squared?**
- Penalizes large errors more
- Differentiable everywhere
- Convex (single global minimum)
- Sensitive to outliers

**Alternatives**:
- **MAE** (Mean Absolute Error): Robust to outliers, non-smooth
- **Huber Loss**: Combines MSE and MAE benefits
- **Quantile Loss**: For quantile regression

---

### Gradient Descent

**Intuition**: Navigate downhill on error surface

**Update Rule**:
```python
# Compute gradient
gradients = (2/n) * X_b.T @ (X_b @ weights - y)

# Update weights
weights -= learning_rate * gradients
```

**Derivation**:
```
L = (1/n) Σ(ŷ - y)²
∂L/∂β = (2/n) Σ(ŷ - y) · ∂ŷ/∂β
      = (2/n) Σ(ŷ - y) · x
      = (2/n) X^T(Xβ - y)
```

**Key Insight**: Gradient is matrix multiplication - vectorized!

---

## What Linear Regression Can Do

### Primary Use Cases
1. **Prediction**: Forecast continuous values
2. **Inference**: Understand feature relationships
3. **Feature Importance**: Identify influential variables
4. **Trend Analysis**: Time series forecasting
5. **Causal Analysis**: Measure effect sizes (with cautions)

### Example Applications
- **Real Estate**: Predict house prices from features
- **Finance**: Stock price prediction, risk assessment
- **Marketing**: Sales forecasting, ad spend optimization
- **Healthcare**: Medical costs, treatment outcomes
- **Engineering**: Performance modeling, optimization

### Advantages
✅ **Simple**: Easy to understand and implement
✅ **Fast**: O(nd) training and prediction
✅ **Interpretable**: Weights directly show feature importance
✅ **Low Variance**: Doesn't overfit easily
✅ **Extrapolation**: Can predict beyond training range
✅ **Probabilistic**: Can add confidence intervals
✅ **No Hyperparameters** (closed form): Unique solution

### Limitations
❌ **Linear Only**: Can't capture non-linear relationships
❌ **Feature Engineering**: Need to manually create interactions
❌ **Sensitive to Outliers**: Squared error amplifies outliers
❌ **Multicollinearity**: Correlated features cause instability
❌ **Homoscedasticity Assumption**: Constant error variance
❌ **Normality Assumption**: Errors should be normally distributed

---

## Critical Implementation Insights

### 1. **Feature Scaling - Absolutely Critical**

**Why Needed**: Features with different scales cause slow convergence

**Problem Visualization**:
```
Without scaling: Elongated error surface (slow zigzag path)
With scaling: Circular error surface (direct path)
```

**Example**:
```python
# Feature 1: Age (0-100), Feature 2: Income (0-1M)
# Gradient for income will be 10,000x larger!
# Weight updates become inefficient
```

**Implementation**:
```python
def fit(self, X, y):
    # Compute statistics on training data
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    
    # Handle constant features
    self.std[self.std == 0] = 1  # Critical: Avoid division by zero
    
    # Apply scaling
    X_scaled = (X - self.mean) / self.std
```

**Common Mistake**:
```python
# WRONG: Scale test data independently
X_test_scaled = (X_test - X_test.mean()) / X_test.std()  # ❌

# CORRECT: Use training statistics
X_test_scaled = (X_test - self.train_mean) / self.train_std  # ✅
```

**Key Insight**: Scaling parameters are part of the model!

---

### 2. **The Bias Trick - Elegant Simplification**

**Problem**: Handling bias separately is tedious
```python
# Without bias trick (clunky)
prediction = self.bias + np.dot(X, self.weights)
```

**Solution**: Add column of 1s to X
```python
# With bias trick (elegant)
X_b = np.c_[np.ones((n_samples, 1)), X]
prediction = np.dot(X_b, self.weights)
# Now weights[0] is the bias!
```

**Benefits**:
- Unified matrix operations
- Simpler code
- Vectorized gradient computation

**Trade-off**: Slightly more memory (one extra column)

---

### 3. **Closed-Form Solution vs. Gradient Descent**

**Normal Equation** (closed-form):
```python
# One-step solution
beta = np.linalg.inv(X^T X) @ X^T @ y
```

**Pros**:
- Exact solution
- No hyperparameters (learning rate, iterations)
- No iterations needed

**Cons**:
- O(n³) time complexity (matrix inversion)
- Requires X^T X to be invertible
- Infeasible for large datasets (>10K features)
- No regularization built-in

**When to Use**:
- Small datasets (< 10K samples, < 100 features)
- Need exact solution
- Computational resources available

**Gradient Descent** (iterative):
```python
# Multiple iterations
for i in range(n_iterations):
    gradients = compute_gradients(X, y, weights)
    weights -= learning_rate * gradients
```

**Pros**:
- Scales to large datasets
- O(nd) per iteration
- Easy to add regularization
- Handles streaming data

**Cons**:
- Requires hyperparameter tuning
- May not converge if badly configured
- Approximate solution

**When to Use**:
- Large datasets (>10K samples)
- Online learning
- Need regularization

---

### 4. **Gradient Computation - Vectorization is Key**

**Naive Loop** (slow):
```python
gradients = np.zeros(n_features + 1)
for i in range(n_samples):
    error = predictions[i] - y[i]
    for j in range(n_features + 1):
        gradients[j] += error * X_b[i, j]
gradients *= (2 / n_samples)
```

**Vectorized** (fast):
```python
gradients = (2 / n_samples) * X_b.T @ (X_b @ weights - y)
```

**Speedup**: 100-1000x faster!

**Matrix Dimensions**:
```
X_b: (n, d+1)
weights: (d+1, 1)
predictions: (n, 1)
error: (n, 1)
gradients: (d+1, 1)
```

**Key Insight**: Matrix multiplication leverages BLAS (optimized linear algebra)

---

### 5. **Convergence Monitoring**

**Problem**: How do we know when to stop?

**Solution**: Track loss over iterations
```python
losses = []
for i in range(n_iterations):
    # Compute predictions
    predictions = X_b @ weights
    
    # Compute loss
    loss = np.mean((predictions - y) ** 2)
    losses.append(loss)
    
    # Early stopping
    if i > 0 and abs(losses[-1] - losses[-2]) < 1e-6:
        print(f"Converged at iteration {i}")
        break
    
    # Gradient descent step
    gradients = (2 / n_samples) * X_b.T @ (predictions - y)
    weights -= learning_rate * gradients
```

**Visualize Convergence**:
```python
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Convergence Curve')
plt.yscale('log')  # Log scale to see plateau
plt.show()
```

**Good Convergence**:
```
Loss
  |●
  | ●
  |  ●
  |   ●___________  ← Smooth decrease, then plateau
  |_________________ Iteration
```

**Bad Convergence (high LR)**:
```
Loss
  |    ●     ●
  |  ●   ● ●    ●   ← Oscillation or increase
  | ●          ●
  |_________________ Iteration
```

---

## Common Bugs I Hit

### 1. **Forgot to Scale**
Gradient descent is SLOW without scaling. Feature with big range dominates, zigzag path to minimum.

### 2. **Learning Rate Drama**
- Too small: Takes forever
- Too large: Bounces around, never converges, might even diverge
- Just right: Smooth loss curve

**Fix**: Start at 0.01, plot loss vs iterations. If oscillating, lower it.

### 3. **Multicollinearity**
Highly correlated features mess up weight estimates. Like having height in cm AND inches - model can't decide how to split the weight.

**Fix**: Remove one, or use regularization (Ridge/Lasso).

---

### 4. **Outliers Dominate Loss**

**Problem**: Squared error amplifies outliers

**Example**:
```python
# Most points: y ∈ [0, 100]
# One outlier: y = 10,000
# MSE dominated by outlier error: (10,000 - ŷ)²
```

**Visual**:
```
y
|                            ●  ← Outlier pulls regression line
|  ●   ●
|    ●   ●  ●
|  ●  ●   ●
|______________________ x
```

**Solutions**:
1. **Outlier Removal**: Preprocess data
   ```python
   # Remove points beyond 3 standard deviations
   z_scores = np.abs((y - y.mean()) / y.std())
   X_clean = X[z_scores < 3]
   y_clean = y[z_scores < 3]
   ```

2. **Robust Loss**: Use MAE or Huber loss
   ```python
   # MAE: Less sensitive to outliers
   loss = np.mean(np.abs(predictions - y))
   gradients = (1/n) * X_b.T @ np.sign(predictions - y)
   ```

3. **Robust Regression**: RANSAC, Theil-Sen estimator

---

### 5. **Underfitting (High Bias)**

**Symptom**: Poor performance on both training and test data

**Causes**:
- Linear model too simple for data
- Missing important features
- Over-regularization

**Diagnosis**:
```python
train_score = r2_score(y_train, model.predict(X_train))
test_score = r2_score(y_test, model.predict(X_test))

if train_score < 0.6 and test_score < 0.6:
    print("Underfitting!")
```

**Solutions**:
1. **Polynomial Features**: Add non-linearity
   ```python
   from sklearn.preprocessing import PolynomialFeatures
   poly = PolynomialFeatures(degree=2)
   X_poly = poly.fit_transform(X)
   ```

2. **Interaction Terms**: x₁ × x₂
3. **More Features**: Domain-specific engineering
4. **Non-linear Model**: Try tree-based methods

---

### 6. **Overfitting (High Variance)**

**Symptom**: Good training, poor test performance

**Causes**:
- Too many features (curse of dimensionality)
- Polynomial features without regularization
- Small dataset

**Diagnosis**:
```python
if train_score > 0.95 and test_score < 0.7:
    print("Overfitting!")
```

**Solutions**:
1. **Regularization**: Ridge (L2) or Lasso (L1)
   ```python
   # Ridge: Add penalty for large weights
   loss += (lambda / 2) * np.sum(weights ** 2)
   gradients += lambda * weights
   ```

2. **Feature Selection**: Remove irrelevant features
3. **More Data**: Collect additional samples
4. **Cross-Validation**: Tune regularization strength

---

## Advanced Extensions

### 1. **Ridge Regression (L2 Regularization)**

**Loss Function**:
```
L = MSE + λ Σ βᵢ²
```

**Effect**: Shrinks weights toward zero

**Implementation**:
```python
def fit(self, X, y, lambda_reg=0.1):
    for i in range(n_iterations):
        # Standard gradient
        gradients = (2/n) * X_b.T @ (X_b @ weights - y)
        
        # Add regularization term
        gradients[1:] += lambda_reg * weights[1:]  # Don't regularize bias
        
        weights -= learning_rate * gradients
```

**Closed Form**:
```python
# Adds λI to X^T X to ensure invertibility
weights = np.linalg.inv(X.T @ X + lambda_reg * np.eye(d)) @ X.T @ y
```

**When to Use**:
- Multicollinearity present
- Overfitting
- More features than samples

---

### 2. **Lasso Regression (L1 Regularization)**

**Loss Function**:
```
L = MSE + λ Σ |βᵢ|
```

**Effect**: Shrinks some weights to exactly zero (feature selection)

**Implementation** (requires subgradient):
```python
gradients += lambda_reg * np.sign(weights)
```

**When to Use**:
- Many irrelevant features
- Want sparse model (feature selection)
- Interpretability important

---

### 3. **Polynomial Regression**

**Idea**: Transform features to capture non-linearity

**Example**:
```python
# Original: y = β₀ + β₁x
# Polynomial (degree 2): y = β₀ + β₁x + β₂x²

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# [x₁, x₂] → [x₁, x₂, x₁², x₁x₂, x₂²]
```

**Trade-off**: More features → more overfitting risk (use regularization!)

---

### 4. **Mini-Batch Gradient Descent**

**Problem**: Full-batch slow on large datasets

**Solution**: Update weights using subsets

**Implementation**:
```python
batch_size = 32
for epoch in range(n_epochs):
    indices = np.random.permutation(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # Compute gradients on batch
        predictions = X_batch @ weights
        gradients = (2/batch_size) * X_batch.T @ (predictions - y_batch)
        weights -= learning_rate * gradients
```

**Benefits**:
- Faster iterations
- Stochastic noise helps escape local minima (not relevant for linear regression)
- Enables online learning

---

## Evaluation Metrics

### 1. **Mean Squared Error (MSE)**
```python
mse = np.mean((y_true - y_pred) ** 2)
```
- Units: Squared target units
- Sensitive to outliers

### 2. **Root Mean Squared Error (RMSE)**
```python
rmse = np.sqrt(mse)
```
- Units: Same as target
- More interpretable than MSE

### 3. **Mean Absolute Error (MAE)**
```python
mae = np.mean(np.abs(y_true - y_pred))
```
- Robust to outliers
- Linear penalty

### 4. **R² Score (Coefficient of Determination)**
```python
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
```
- Range: (-∞, 1]
- 1.0: Perfect predictions
- 0.0: Model = mean baseline
- Negative: Model worse than mean!

**Interpretation**:
- R² = 0.8 → Model explains 80% of variance

### 5. **Adjusted R²**
```python
n = len(y_true)
p = n_features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```
- Penalizes adding features
- Better for model comparison

---

## Assumptions of Linear Regression

### 1. **Linearity**: Relationship between X and y is linear
**Check**: Residual plot (should be random)

### 2. **Independence**: Observations are independent
**Check**: Durbin-Watson test (time series)

### 3. **Homoscedasticity**: Constant error variance
**Check**: Residuals vs. fitted plot (should be uniform)

### 4. **Normality**: Errors are normally distributed
**Check**: QQ-plot, Shapiro-Wilk test

### 5. **No Multicollinearity**: Features not highly correlated
**Check**: VIF, correlation matrix

**What if violated?**
- Linearity → Non-linear models
- Independence → Time series models (ARIMA)
- Homoscedasticity → Weighted least squares
- Normality → Robust regression (less critical for large n)
- Multicollinearity → Ridge/Lasso, PCA

---

## Debugging Checklist

### Poor Performance
- [ ] Features scaled?
- [ ] Learning rate appropriate?
- [ ] Enough iterations?
- [ ] Check for multicollinearity
- [ ] Visualize residuals
- [ ] Try polynomial features
- [ ] Check assumptions

### Non-Convergence
- [ ] Reduce learning rate
- [ ] Increase iterations
- [ ] Check for NaN/Inf in data
- [ ] Feature scaling applied?
- [ ] Try closed-form solution

### Overfitting
- [ ] Add regularization (Ridge/Lasso)
- [ ] Remove correlated features
- [ ] Get more data
- [ ] Cross-validation
- [ ] Reduce polynomial degree

---

## Key Takeaways

✅ **Interpretable**: Weights directly show feature importance
✅ **Fast**: Training and prediction are efficient
✅ **Simple**: Easy to implement and understand
✅ **Probabilistic**: Can provide confidence intervals
✅ **Baseline**: Always try before complex models

❌ **Linear Only**: Can't capture non-linear patterns without feature engineering
❌ **Sensitive to Outliers**: Squared error amplifies large errors
❌ **Assumptions**: Requires several assumptions to be valid
❌ **Feature Engineering**: Manual interaction/polynomial terms needed

**When to Use**:
- Relationships are linear
- Need interpretability
- Small to medium datasets
- Baseline model
- Inference (not just prediction)

**When to Avoid**:
- Highly non-linear data
- Many outliers (use robust methods)
- Assumptions violated
- Black-box model acceptable (try tree-based)

**Next Steps**:
1. Add regularization (Ridge/Lasso)
2. Implement confidence intervals
3. Try polynomial features
4. Compare with closed-form solution
5. Implement learning rate schedules
