# K-Nearest Neighbors (KNN) - Walkthrough

## Overview
KNN is super simple: to predict something, just look at the K closest examples and let them vote. No training phase needed - it just memorizes the data!

## Parameters That Actually Matter

### 1. **k (Number of Neighbors)**
- **Default**: 5 (use odd numbers to avoid ties)
- **Small k (1-3)**: Overfits - too sensitive to noise
- **Large k (50+)**: Underfits - too smooth, misses patterns
- **How to pick**: Use cross-validation! Don't just guess.

### 2. **task**
- **Type**: String
- **Options**: `'classification'` or `'regression'`
- **Purpose**: Determines prediction methodology
  - Classification: Uses majority voting
  - Regression: Uses averaging of neighbor values
- **When to Use**:
  - Classification: Discrete target variables (categories)
  - Regression: Continuous target variables (numerical values)

### 3. **distance_metric**
- **Type**: String
- **Options**: `'euclidean'` or `'manhattan'`
- **Purpose**: Defines how distance between points is calculated
- **Euclidean Distance**: √(Σ(xi - yi)²)
  - Best for: Continuous features with similar scales
  - Sensitive to: Feature scaling differences
- **Manhattan Distance**: Σ|xi - yi|
  - Best for: High-dimensional data, grid-like paths
  - More robust to: Outliers
- **Implementation**: Vectorized computation for efficiency

### 4. **weights**
- **Type**: String
- **Options**: `'uniform'` or `'distance'`
- **Purpose**: Determines how neighbor votes are weighted
- **Uniform**: All neighbors have equal voting power
- **Distance-weighted**: Closer neighbors have more influence
  - Formula: weight = 1 / (distance + epsilon)
  - Better for: Non-uniform distributions, reducing influence of distant neighbors
  - Handles edge cases with epsilon (1e-10) to avoid division by zero

### 5. **batch_size** (in predict method)
- **Type**: Integer
- **Default**: 500
- **Purpose**: Controls memory usage during prediction
- **Why needed**: Distance matrices can be huge (N_test × N_train)
- **Memory consideration**: For 10K test × 100K train, that's 1 billion floats ≈ 4GB!

---

## What KNN Can Do

### Classification Tasks
1. **Binary Classification**: Spam detection, disease diagnosis
2. **Multi-class Classification**: Image recognition, character recognition
3. **Probability Estimation**: Returns confidence scores via vote proportions

### Regression Tasks
1. **Numeric Prediction**: House price estimation, temperature forecasting
2. **Weighted Averaging**: Distance-weighted predictions for smoother estimates

### Why KNN Rocks
✅ No training needed - just store the data
✅ Simple to understand
✅ Works for both classification and regression
✅ Naturally handles multi-class problems

### Why KNN Sucks
❌ Predictions are slow (gotta check every point!)
❌ Memory hog (stores everything)
❌ Terrible in high dimensions (curse of dimensionality)
❌ MUST scale features or it breaks
❌ Sensitive to noisy data

---

## The Big Gotchas

### 1. **Feature Scaling - Not Optional!**

Listen, if you don't scale your features, KNN will completely ignore some of them.

**Why?** Simple - if age is 0-100 and salary is 20,000-200,000, the distance calculation will be dominated by salary. Age might as well not exist.

```python
# The fix:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training stats!
```

**The mistake everyone makes**: Scaling test data with its own mean/std. That's data leakage!

---

### 2. **Vectorized Distance Computation**

**Naive Approach (Slow)**:
```python
# O(N_test × N_train × D)
for test_point in X_test:
    for train_point in X_train:
        dist = np.sum((test_point - train_point)**2)**0.5
```

**Optimized Approach (Fast)**:
```python
# Euclidean: Use algebraic trick (x-y)² = x² + y² - 2xy
X_sq = np.sum(X**2, axis=1, keepdims=True)  # Shape: (m, 1)
Train_sq = np.sum(self.X_train**2, axis=1)   # Shape: (n,)
dists = np.sqrt(X_sq + Train_sq.T - 2 * np.dot(X, self.X_train.T))  # (m, n)
```

**Why This Works**:
- Avoids explicit broadcasting subtraction
- Uses optimized BLAS matrix multiplication
- ~10-100x faster for large datasets

**Tricky Part**: Must use `np.maximum(..., 0)` because floating-point errors can make very small values negative inside sqrt!

---

### 3. **Handling Edge Cases in Classification**

**Problem**: Tie-breaking in majority voting

**Scenario**:
```python
# k=4, classes [0, 0, 1, 1] - a tie!
```

**Solution Implemented**:
```python
vals, counts = np.unique(labels[i], return_counts=True)
# np.unique sorts, so we get consistent tie-breaking
predictions.append(vals[np.argmax(counts)])
```

**Key Insight**: `np.unique` returns sorted values, ensuring deterministic behavior. For better tie-breaking, use distance weights!

---

### 4. **Batch Processing for Memory Management**

**Problem**: Distance matrix can exceed available RAM

**Example**:
- 100,000 test samples × 1,000,000 training samples × 8 bytes = 800GB!

**Solution**:
```python
def predict(self, X, batch_size=500):
    predictions = []
    for i in range(0, n_queries, batch_size):
        X_batch = X[i:i+batch_size]
        dists = self._compute_distances(X_batch)  # Only batch_size × N_train
        # ... process batch ...
        predictions.extend(batch_preds)
    return np.array(predictions)
```

**Key Insight**: Process in chunks to keep memory usage bounded. Default batch_size=500 keeps memory under ~4GB for typical datasets.

---

### 5. **K-fold Cross-Validation Implementation**

**Common Mistake**: Not shuffling data before splitting

**Problem**:
```python
# If data is sorted by class: [all_0s, all_1s]
# First fold gets only 0s, last fold gets only 1s!
```

**Correct Approach**:
```python
def k_fold_cv(X, y, k_folds=5):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Critical!
    
    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[:n_samples % k_folds] += 1  # Handle uneven division
```

**Key Insight**: Shuffling ensures balanced class distribution across folds. Handle remainder samples carefully to avoid index errors.

---

### 6. **Distance-Weighted Voting**

**Problem**: Division by zero when distance = 0 (duplicate points)

**Naive Approach**:
```python
weights = 1.0 / dists  # Breaks when dists = 0!
```

**Robust Solution**:
```python
with np.errstate(divide='ignore', invalid='ignore'):
    weights = 1.0 / (dists + 1e-10)
```

**Key Insight**: Small epsilon prevents division by zero while minimally affecting non-zero distances.

---

## Potential Sticking Points While Coding

### 1. **Curse of Dimensionality**
**Problem**: As dimensions increase, all points become equidistant.

**Why It Happens**: Volume of hypersphere concentrates at surface

**Example**:
```python
# In 100 dimensions, even random points have similar distances!
np.random.seed(42)
X = np.random.randn(1000, 100)
dists = cdist(X[:10], X, 'euclidean')
print(f"Mean: {dists.mean():.2f}, Std: {dists.std():.2f}")
# Output: Mean: 10.00, Std: 0.32  <- Very small relative variance!
```

**Solution**: 
- Dimensionality reduction (PCA)
- Feature selection
- Use distance-weighted voting

---

### 2. **Class Imbalance**
**Problem**: Rare classes never get predicted with uniform voting

**Example**:
```python
# Dataset: 95% class 0, 5% class 1
# With k=10, likely 9-10 neighbors are class 0
# Class 1 rarely predicted!
```

**Solutions**:
- Use smaller k
- Distance-weighted voting (gives rare but close points more weight)
- Stratified sampling
- Cost-sensitive learning

---

### 3. **Computational Complexity**

**Time Complexity**:
- Training: O(1) - just stores data
- Prediction: O(N × D) per query
  - N = training samples
  - D = dimensions

**Space Complexity**: O(N × D)

**Real-World Impact**:
```python
# Dataset: 1M samples, 100 features
# Single prediction: 100M operations
# 10K predictions: 1 trillion operations!
```

**Mitigation Strategies**:
1. **Approximate Nearest Neighbors (ANN)**
   - KD-Trees (works well up to ~20 dimensions)
   - Ball Trees (better for higher dimensions)
   - LSH (Locality Sensitive Hashing)
   
2. **Parallelization**
   - Distance computation is embarrassingly parallel
   - Use NumPy's vectorization (already exploits BLAS)

---

### 4. **Choosing Optimal K**

**Approach Used**: Cross-Validation with Grid Search

**Implementation**:
```python
def select_best_k(X, y, k_values, task='classification', cv=5):
    # Scale first!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    mean_scores = []
    for k in k_values:
        params = {'k': k, 'task': task, 'weights': 'distance'}
        score = cross_val_score(KNN, params, X_scaled, y, cv=cv)
        mean_scores.append(score)
    
    best_idx = np.argmax(mean_scores)
    return k_values[best_idx]
```

**Visualization**: Elbow method
```python
plt.plot(k_values, mean_scores, marker='o')
# Look for the "elbow" where improvement plateaus
```

**Key Insight**: Don't just pick k=5 arbitrarily! Data characteristics matter. Use odd k to avoid ties.

---

## Performance Optimization Tricks

### 1. **Manhattan vs. Euclidean**
```python
# Manhattan is faster (no square root!)
dists = np.sum(np.abs(X[:, np.newaxis, :] - X_train), axis=2)

# But Euclidean uses optimized BLAS (faster overall!)
dists = np.sqrt(X_sq + Train_sq.T - 2 * np.dot(X, X_train.T))
```

**Takeaway**: Euclidean with algebraic trick beats Manhattan despite sqrt!

---

### 2. **Partial Sorting**
```python
# Full sort: O(N log N)
knn_indices = np.argsort(dists, axis=1)[:, :k]

# Partial sort: O(N + k log k) - faster when k << N
knn_indices = np.argpartition(dists, k, axis=1)[:, :k]
# But elements within k aren't sorted!

# If need sorted: combine approaches
indices = np.argpartition(dists, k, axis=1)[:, :k]
sorted_within = np.take_along_axis(dists, indices, axis=1).argsort(axis=1)
knn_indices = np.take_along_axis(indices, sorted_within, axis=1)
```

**When Used**: Not in current implementation (clarity over micro-optimization), but valuable for very large N.

---

### 3. **Memory Layout**
```python
# Row-major (C-contiguous) is faster for row operations
assert X.flags['C_CONTIGUOUS']

# If not, convert:
X = np.ascontiguousarray(X)
```

**Why**: Cache locality improves performance by ~2x for distance calculations.

---

## Common Debugging Scenarios

### 1. **Poor Accuracy Despite "Good" Implementation**
**Checklist**:
- [ ] Did you scale features?
- [ ] Is k too large or too small?
- [ ] Are features relevant?
- [ ] Is there class imbalance?
- [ ] Are there outliers/noise?

### 2. **Memory Error**
**Symptoms**: `numpy.core._exceptions.MemoryError`

**Fix**:
```python
# Reduce batch_size
predictions = model.predict(X_test, batch_size=100)
```

### 3. **Slow Predictions**
**Symptoms**: Minutes per prediction

**Fixes**:
1. Check if using vectorized operations (not loops)
2. Ensure NumPy linked to optimized BLAS (Intel MKL, OpenBLAS)
3. Consider approximate methods (KD-Tree, etc.)

---

## Best Practices Summary

1. **Always scale features** before fitting or predicting
2. **Use cross-validation** to select k, don't guess
3. **Prefer distance weights** over uniform for better performance
4. **Monitor memory usage** with large datasets (use batching)
5. **Benchmark both metrics** (Euclidean vs. Manhattan) for your data
6. **Consider dimensionality reduction** for high-D data
7. **Use KNN as a baseline** before trying complex models
8. **Document data assumptions** (scaling params, k selection rationale)

---

## Advanced Extensions

### 1. **KD-Tree Acceleration**
```python
from scipy.spatial import KDTree
tree = KDTree(X_train)
distances, indices = tree.query(X_test, k=k)
```
**Works well**: Up to ~20 dimensions
**Breaks down**: High dimensions (curse of dimensionality)

### 2. **Ball Tree**
```python
from sklearn.neighbors import BallTree
tree = BallTree(X_train)
distances, indices = tree.query(X_test, k=k)
```
**Better for**: Higher dimensions than KD-Tree

### 3. **Locality Sensitive Hashing (LSH)**
**Use case**: Very high dimensions, approximate results acceptable
**Trade-off**: Speed vs. accuracy

---

## Real-World Applications

1. **Recommendation Systems**: Find similar users/items
2. **Anomaly Detection**: Points with distant neighbors are outliers
3. **Missing Value Imputation**: Use neighbors' values
4. **Image Classification**: Simple yet effective for small datasets
5. **Spell Checkers**: Find nearest valid word
6. **Medical Diagnosis**: Match patient symptoms to known cases

---

## Key Takeaways

✅ **Simplicity**: No training, easy to implement
✅ **Versatility**: Works for classification and regression
✅ **Non-parametric**: No assumptions about data distribution
✅ **Interpretability**: Predictions easily explainable

❌ **Computational Cost**: Slow for large datasets
❌ **Memory**: Stores entire training set
❌ **Feature Scaling**: Critical preprocessing requirement
❌ **High Dimensions**: Performance degrades

**When to Use**: 
- Small to medium datasets
- Need interpretable model
- Data has local structure
- Baseline comparison

**When to Avoid**:
- Real-time prediction required
- Very large datasets (>100K samples)
- Very high dimensions (>50 features)
- Memory constrained environments
