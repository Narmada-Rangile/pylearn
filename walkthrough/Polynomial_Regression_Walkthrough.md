# Polynomial Regression - Walkthrough

## Overview
Linear regression but make it curvy! Transform your features into x, x², x³, etc., then use regular linear regression. Still linear in the weights (which is why it works), but now you can fit curves. Just don't go crazy with high degrees or you'll overfit like crazy.

## Parameters

### 1. **degree**
- **Type**: Integer
- **Default**: 1 (equivalent to linear regression)
- **Range**: 1 to 10+ (practical limit ~5-6)
- **Purpose**: Highest power of polynomial terms
- **Effect**:
  - **degree=1**: Linear (straight line)
  - **degree=2**: Quadratic (parabola)
  - **degree=3**: Cubic (S-curve)
  - **degree≥4**: Complex curves, high overfitting risk
- **Selection**: Cross-validation, watch for overfitting

### 2. **learning_rate**
- **Type**: Float
- **Default**: 0.001
- **Range**: 0.0001 to 0.1
- **Purpose**: Gradient descent step size
- **Note**: Lower than linear regression due to feature explosion
- **Why lower?**: More features → larger gradients → need smaller steps

### 3. **iterations**
- **Type**: Integer
- **Default**: 1000
- **Range**: 100 to 10,000
- **Purpose**: Number of optimization steps
- **Note**: May need more iterations than linear regression due to complex loss surface

---

## Mathematical Foundation

### Polynomial Transformation

**Degree 2 Example** (1 feature):
```
Original: [x]
Transformed: [1, x, x²]

Model: y = β₀ + β₁x + β₂x²
```

**Degree 2 Example** (2 features):
```
Original: [x₁, x₂]
Transformed: [1, x₁, x₂, x₁², x₁x₂, x₂²]

Model: y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₁x₂ + β₅x₂²
```

**General Formula**:
```
Number of terms = (n + d)! / (n! × d!)
Where n = number of features, d = degree
```

**Feature Explosion**:
```
Features  Degree  Total Terms
   1        3         4
   2        3         10
   5        3         56
   10       3         286
   10       5         3,003  ← Becomes unwieldy!
```

---

### Design Matrix Creation

**Implementation**:
```python
def create_design_matrix(self, X):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Start with bias column (all ones)
    design_matrix = [np.ones((n_samples, 1))]
    
    # Add polynomial terms degree 1 to degree
    for d in range(1, self.degree + 1):
        design_matrix.append(X ** d)
    
    return np.column_stack(design_matrix)
```

**Example Output** (degree=3, 1 feature):
```python
X = [[2], [3], [4]]

# After transformation:
[[1, 2, 4, 8],    # [1, x, x², x³]
 [1, 3, 9, 27],
 [1, 4, 16, 64]]
```

**Key Insight**: This is a **simplified implementation** for independent features. Full polynomial features would include cross-terms (x₁x₂, x₁²x₂, etc.).

---

## What Polynomial Regression Can Do

### Capture Non-Linear Patterns
1. **Quadratic Trends**: U-shaped or inverted U-shaped curves
   - Economics: Diminishing returns
   - Physics: Projectile motion
2. **Cubic Trends**: S-shaped curves
   - Biology: Growth curves
   - Marketing: Product lifecycle
3. **Higher-Order**: Complex oscillations
   - Time series: Seasonal patterns

### Visual Examples

**Linear (degree=1)**:
```
y
|      ●
|    ●
|  ●
|●________ x
```

**Quadratic (degree=2)**:
```
y
|   ●●●
| ●     ●
|●       ●___ x
```

**Cubic (degree=3)**:
```
y
|    ●●
|  ●   ●
|●      ●
|        ●__ x
```

**Over-fitted (degree=10)**:
```
y
|  ●●●●
| ●    ●●
|●       ●
|         ●_ x
← Oscillates wildly through noise!
```

---

### Advantages
✅ **Captures Non-Linearity**: No manual feature engineering
✅ **Still Linear Model**: Standard linear regression applies
✅ **Interpretable** (low degrees): Coefficients have meaning
✅ **Closed-Form Solution**: Can use normal equation
✅ **Fast Prediction**: Just matrix multiplication

### Disadvantages
❌ **Feature Explosion**: Number of features grows combinatorially
❌ **Overfitting Risk**: High degrees fit noise
❌ **Extrapolation Issues**: Unreliable beyond training range
❌ **Multicollinearity**: x, x², x³ are highly correlated
❌ **Scaling Critical**: Polynomial features have vastly different scales
❌ **Interpretability Lost** (high degrees): Complex interactions

---

## Critical Implementation Insights

### 1. **Feature Scaling is MANDATORY**

**Problem**: Polynomial features have wildly different scales

**Example**:
```python
x = 100  # Already large
x² = 10,000
x³ = 1,000,000
x⁴ = 100,000,000  # Astronomical!
```

**Consequence**: Numerical instability, poor convergence

**Solution**: Scale AFTER polynomial transformation
```python
def fit(self, X, y):
    # Create polynomial features
    X_design = self.create_design_matrix(X)
    
    # Scale everything except bias column
    self.feature_means = np.mean(X_design[:, 1:], axis=0)
    self.feature_stds = np.std(X_design[:, 1:], axis=0)
    
    # Handle zero variance
    self.feature_stds[self.feature_stds == 0] = 1.0
    
    # Apply scaling
    X_design[:, 1:] = (X_design[:, 1:] - self.feature_means) / self.feature_stds
```

**Key Insight**: Don't scale the bias column (always 1)!

---

### 2. **Target Scaling - Often Overlooked**

**Problem**: If y has large values, gradients can explode

**Example**:
```python
y = [100000, 200000, 150000]  # House prices
# Errors are large → gradients are huge → unstable training
```

**Solution**: Scale both X and y
```python
# In training
y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

# Train on scaled data
model.fit(X_scaled, y_scaled)

# Predict and unscale
predictions_scaled = model.predict(X_test_scaled)
predictions = predictions_scaled * y_std + y_mean
```

**Key Insight**: Remember to inverse transform predictions!

---

### 3. **Degree Selection - The Goldilocks Problem**

**Too Low** (underfitting):
```python
degree = 1  # Linear
# Can't capture curvature
# High bias, low variance
```

**Too High** (overfitting):
```python
degree = 10  # Complex
# Fits noise
# Low bias, high variance
```

**Just Right**:
```python
# Use cross-validation
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
best_degree = 1
best_rmse = float('inf')

for d in degrees:
    model = PolynomialRegression(degree=d)
    
    # K-fold cross-validation
    scores = []
    for train_idx, val_idx in kfold_splits:
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[val_idx], y[val_idx])
        scores.append(score)
    
    avg_score = np.mean(scores)
    
    if avg_score < best_rmse:
        best_rmse = avg_score
        best_degree = d
```

**Visualization**: Plot validation error vs. degree
```
Error
  |●
  |  ●
  |    ●          ← Optimum
  |      ●
  |        ●●●●  ← Overfitting
  |_______________ Degree
   1  2  3  4  5  6  7
```

---

### 4. **Multicollinearity is Guaranteed**

**Problem**: x, x², x³, ... are inherently correlated

**Example**:
```python
x = np.array([1, 2, 3, 4, 5])
corr(x, x²) ≈ 0.98  # Very high!
corr(x², x³) ≈ 0.99  # Even higher!
```

**Consequences**:
- Unstable weight estimates
- Weights change dramatically with small data changes
- High condition number of X^T X matrix
- Closed-form solution may fail (singular matrix)

**Solutions**:

**1. Ridge Regularization** (L2):
```python
# Add penalty for large weights
loss = MSE + λ Σ βᵢ²

# In gradient descent
gradients += lambda_reg * weights
```

**2. Orthogonal Polynomials**:
```python
# Use Legendre or Chebyshev polynomials instead of raw powers
# These are mathematically orthogonal → no correlation
from numpy.polynomial import legendre
poly_features = legendre.legval(X, coeffs)
```

**3. Standardization**: Already applied (helps but doesn't eliminate)

---

### 5. **Extrapolation Danger**

**Problem**: Polynomials behave badly outside training range

**Example**:
```python
# Training data: x ∈ [0, 10]
# Learned: y = x - 0.1x² + 0.001x³

# Prediction at x=20 (outside range):
y = 20 - 0.1(400) + 0.001(8000) = 20 - 40 + 8 = -12
# Nonsensical! Physical values may be negative, but likely wrong
```

**Visual**:
```
y
 |        Training
 |        Region
 |      ●●●●●
 |    ●       ●
 |  ●           ●
 | ●             ↓ Extrapolation
 |                ●
 |_____________________ x
-2  0   5   10   15   20

← May curve back down unexpectedly!
```

**Key Insight**: Polynomial regression is **interpolation**, not extrapolation!

**Solution**: Only predict within training range, or use domain knowledge to constrain

---

### 6. **Gradient Computation with Polynomial Features**

**Same as Linear Regression**:
```python
gradients = (2 / n_samples) * X_design.T @ (X_design @ weights - y)
```

**Key Insight**: Because we transform features, the model is still linear in weights!
- This is why it's called "polynomial regression" but uses linear regression machinery
- Non-linear in original features, linear in polynomial features

---

## Where It Goes Wrong

### 1. **Feature Explosion**
With 10 features and degree 5, you get 3,003 features. Memory and speed problems incoming!

### 2. **Overfitting is Easy**
degree=10 will fit noise perfectly. Always use cross-validation to pick degree.

### 3. **Scale BEFORE Transforming**
```python
# WRONG
X_poly = X ** 5  # 100^5 = 10 billion!
X_scaled = scale(X_poly)

# RIGHT  
X_scaled = scale(X)
X_poly = X_scaled ** 5  # Now it's reasonable
```

### 4. **Don't Extrapolate!**
Polynomials go crazy outside training range. Train on x∈[0,10], predict at x=20? Good luck!

---

## Advanced Techniques

### 1. **Interaction Terms** (True Polynomial Features)

**Current Implementation**: Independent features only
```python
# [x₁, x₂] → [x₁, x₂, x₁², x₂²]
```

**Full Polynomial Features**: Include interactions
```python
# [x₁, x₂] → [x₁, x₂, x₁², x₁x₂, x₂²]

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

**When Useful**: Features interact (e.g., height × weight for BMI)

---

### 2. **Regularization is Essential**

**Ridge** (L2):
```python
# Shrinks all coefficients
loss = MSE + λ Σ βᵢ²
```

**Lasso** (L1):
```python
# Some coefficients become exactly zero (feature selection)
loss = MSE + λ Σ |βᵢ|
```

**Elastic Net** (L1 + L2):
```python
# Combination of both
loss = MSE + λ₁ Σ |βᵢ| + λ₂ Σ βᵢ²
```

**Why Critical**: High-degree polynomials have many correlated features → regularization stabilizes weights

---

### 3. **Piecewise Polynomial Regression** (Splines)

**Idea**: Fit different polynomials in different regions

**Example**:
```python
# Split domain into segments
# Fit cubic polynomial in each segment
# Ensure continuity at boundaries (knots)

from scipy.interpolate import UnivariateSpline
spline = UnivariateSpline(X, y, k=3, s=0.1)
predictions = spline(X_test)
```

**Benefits**:
- Lower degree per segment (less overfitting)
- More flexible than single polynomial
- Better extrapolation

---

### 4. **Bayesian Model Comparison**

**Use Bayesian Information Criterion (BIC)** to select degree:
```python
def bic_score(y_true, y_pred, n_features):
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    bic = n * np.log(mse) + n_features * np.log(n)
    return bic

# Lower BIC is better
bic_scores = []
for degree in [1, 2, 3, 4, 5]:
    model = PolynomialRegression(degree=degree)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    n_features = model.weights.shape[0]
    bic = bic_score(y_val, predictions, n_features)
    bic_scores.append(bic)

best_degree = np.argmin(bic_scores) + 1
```

---

## Comparison: Linear vs. Polynomial Regression

| Aspect | Linear | Polynomial |
|--------|--------|------------|
| **Complexity** | Simple | Complex |
| **Features** | n | O(n^d) |
| **Interpretability** | High | Low (degree > 2) |
| **Overfitting Risk** | Low | High |
| **Extrapolation** | Reliable | Unreliable |
| **Training Speed** | Fast | Slower (more features) |
| **Prediction Speed** | Fast | Slower (more computation) |
| **Memory** | Low | High |
| **Use Case** | Linear trends | Non-linear trends |

---

## Debugging Checklist

### Poor Training Performance
- [ ] Degree too low? (Underfitting)
- [ ] Features scaled?
- [ ] Target scaled?
- [ ] Learning rate appropriate?
- [ ] Enough iterations?
- [ ] Check for NaN/Inf

### Poor Test Performance
- [ ] Degree too high? (Overfitting)
- [ ] Add regularization
- [ ] Cross-validate degree selection
- [ ] Visualize predictions
- [ ] Check extrapolation range

### Numerical Issues
- [ ] Scale features BEFORE polynomial transformation
- [ ] Check for overflow (limit degree)
- [ ] Handle zero variance features
- [ ] Use regularization (improves conditioning)

---

## Real-World Applications

1. **Economics**: Demand curves (U-shaped relationships)
2. **Engineering**: Stress-strain curves, calibration
3. **Biology**: Growth models (S-curves)
4. **Physics**: Trajectory prediction, motion under friction
5. **Finance**: Option pricing (non-linear payoffs)
6. **Chemistry**: Reaction rates vs. temperature
7. **Climate Science**: Temperature trends over time

---

## Key Takeaways

✅ **Captures Non-Linearity**: Simple extension of linear regression
✅ **Still Linear Model**: Standard techniques apply
✅ **No Black Box** (low degrees): Interpretable relationships
✅ **Fast**: Closed-form solution available
✅ **Flexible**: Adjust degree to match data complexity

❌ **Feature Explosion**: Combinatorial growth with degree
❌ **Overfitting**: High degrees fit noise
❌ **Multicollinearity**: Polynomial features highly correlated
❌ **Extrapolation**: Unreliable beyond training range
❌ **Scaling Critical**: Numerical stability requires careful scaling

**When to Use**:
- Clear non-linear trend
- Smoothly curved relationships
- Interpolation (not extrapolation)
- Need interpretability (degree ≤ 3)
- Domain suggests polynomial form

**When to Avoid**:
- Need extrapolation
- High-dimensional data (feature explosion)
- No clear polynomial relationship
- Black-box models acceptable (use trees/neural networks)

**Best Practices**:
1. **Always scale** features before polynomial transformation
2. **Cross-validate** degree selection
3. **Use regularization** (Ridge/Lasso)
4. **Visualize** predictions vs. actual
5. **Limit degree** to ≤ 5 unless strong reason
6. **Check residuals** for patterns
7. **Avoid extrapolation** or use with extreme caution
8. **Consider splines** for complex curves
