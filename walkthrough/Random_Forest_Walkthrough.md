# Random Forest - Walkthrough

## Overview
Think of it as asking a bunch of decision trees for their opinion, then taking a vote. Each tree is slightly different (trained on different data subsets with random features), so they don't all make the same mistakes. Super powerful and surprisingly hard to mess up!

## Parameters

### Decision Tree Parameters

#### 1. **min_samples_split**
- **Type**: Integer
- **Default**: 2
- **Range**: 2 to N (any value ≥ 2 is valid)
- **Purpose**: Minimum samples required to split an internal node
- **Effect**:
  - **Small values (2-5)**: Deep trees, more splits, overfitting risk
  - **Large values (100+)**: Shallow trees, fewer splits, underfitting risk
- **Use case**: Increase to regularize (reduce overfitting)

#### 2. **max_depth**
- **Type**: Integer
- **Default**: 100 (effectively unlimited)
- **Range**: 1 to ∞
- **Purpose**: Maximum depth of the tree
- **Effect**:
  - **Shallow (1-5)**: Simple rules, high bias, low variance, underfitting
  - **Deep (20+)**: Complex rules, low bias, high variance, overfitting
- **Sweet spot**: Usually 5-20 for most datasets
- **Stopping criterion**: Most important for controlling complexity

#### 3. **n_features** (per tree)
- **Type**: Integer or None
- **Default**: None (uses all features)
- **Purpose**: Number of features to consider when finding best split
- **Feature subsampling**: Creates diversity among trees
- **Implementation**: Random selection without replacement per split

---

### Random Forest Parameters

#### 4. **n_estimators**
- **Default**: 100 (usually good enough)
- **The cool thing**: More trees = better results, won't overfit! It just takes longer.
- **Diminishing returns** kick in around 200-500 trees
- **My approach**: Start with 100, bump to 200 if you need that extra edge

#### 5. **max_features**
- **Type**: String, int, or float
- **Options**:
  - `'sqrt'`: √n_features (default for classification)
  - `'log2'`: log₂(n_features)
  - Integer: Exact number
  - Float: Fraction of n_features
  - `None`: All features (no randomness)
- **Purpose**: Creates diversity by limiting features per split
- **Effect**:
  - **Fewer features**: More diversity, less correlated trees, better generalization
  - **All features**: Less diversity, correlated trees, may overfit
- **Recommendation**: Use 'sqrt' for classification, 'n_features/3' for regression

#### 6. **bootstrap**
- **Type**: Boolean
- **Default**: True
- **Purpose**: Whether to use bootstrap samples when building trees
- **Bootstrap sampling**: Draw N samples WITH replacement from training set
- **Effect**:
  - **True**: Different trees see different data, more diversity
  - **False**: All trees see same data, only feature randomness provides diversity
- **Out-of-bag samples**: When True, ~37% samples not used per tree (useful for OOB error estimation)

#### 7. **random_state**
- **Type**: Integer or None
- **Default**: 42 (or None)
- **Purpose**: Seed for reproducibility
- **Controls**: Bootstrap sampling, feature selection, tie-breaking
- **Important for**: Debugging, comparing experiments, sharing results

---

## What Random Forest Can Do

### Classification
1. **Binary Classification**: Two-class problems
2. **Multi-class**: Naturally handles many classes
3. **Probability Estimation**: Aggregated vote proportions
4. **Feature Importance**: Identify most predictive features

### Regression
1. **Continuous Prediction**: Average of tree predictions
2. **Uncertainty Estimation**: Variance across trees
3. **Non-linear Relationships**: Automatically captures complex patterns

## What It's Good At

✅ High accuracy out of the box
✅ Handles non-linear relationships (no feature engineering!)
✅ Built-in feature importance
✅ Hard to overfit (unlike single trees)
✅ Works with messy data
✅ No scaling needed

## What It Sucks At

❌ Slow predictions (gotta ask all the trees)
❌ Memory hungry (stores all trees)
❌ Black box (can't visualize like single tree)
❌ Biased toward majority class if imbalanced

---

## Critical Implementation Insights

### 1. **Decision Tree Building - Recursive Growth**

**The Core Algorithm**:
```python
def _grow_tree(self, X, y, depth=0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))
    
    # Stopping criteria
    if (depth >= self.max_depth or 
        n_labels == 1 or 
        n_samples < self.min_samples_split):
        return Node(value=self._most_common_label(y))
    
    # Feature subsampling
    feat_idxs = self.rng.choice(n_feats, self.n_features, replace=False)
    
    # Find best split
    best_feat, best_thresh, best_gain = self._best_split(X, y, feat_idxs)
    
    if best_feat is None:  # No valid split found
        return Node(value=self._most_common_label(y))
    
    # Recursively build left and right subtrees
    left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
    left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
    right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
    
    return Node(best_feat, best_thresh, left, right)
```

**Key Insight**: Recursion naturally creates tree structure. Base cases prevent infinite loops.

---

### 2. **Information Gain - The Split Criterion**

**Entropy Calculation**:
```python
def _entropy(self, y):
    hist = np.bincount(y.astype(int))
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
```

**Why p > 0 check?**
```python
# log2(0) = -infinity → NaN in computation!
# Skip zero probabilities to avoid this
```

**Information Gain Formula**:
```
IG = H(parent) - [p(left)·H(left) + p(right)·H(right)]
```

**Intuition**: How much does this split reduce uncertainty?
- High IG: Split creates pure children
- Low IG: Split doesn't separate classes well

**Alternative Metrics** (not implemented, but common):
- **Gini Impurity**: Faster to compute, similar results
  - G = 1 - Σ(pᵢ²)
  - Measures probability of incorrect classification
- **Variance Reduction**: For regression
  - Minimize MSE in child nodes

---

### 3. **Threshold Selection - Greedy Search**

**Current Implementation**:
```python
def _best_split(self, X, y, feat_idxs):
    best_gain = -1
    split_idx, split_thresh = None, None
    
    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx]
        thresholds = np.unique(X_column)
        
        # Optimization: Limit thresholds if too many
        if len(thresholds) > 100:
            thresholds = np.percentile(X_column, np.linspace(0, 100, 50))
        
        for threshold in thresholds:
            gain = self._information_gain(y, X_column, threshold)
            
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_thresh = threshold
    
    return split_idx, split_thresh, best_gain
```

**Critical Optimization**: 
- **Problem**: For continuous features with many unique values, testing all thresholds is O(n²)
- **Solution**: Sample percentiles when >100 unique values
- **Trade-off**: Slight accuracy loss for major speed gain

**Alternative Strategies**:
1. **Random thresholds**: Pick k random values (faster, more random)
2. **Binning**: Pre-discretize continuous features
3. **Histogram-based**: Used in LightGBM, XGBoost

---

### 4. **Feature Importance Calculation**

**Implementation**:
```python
# In DecisionTree
self.feature_importances_ = np.zeros(self.n_features_in_)

# During tree growth
self.feature_importances_[best_feat] += best_gain * n_samples

# After tree building
if sum_imp > 0:
    self.feature_importances_ /= sum_imp  # Normalize
```

**Why weight by n_samples?**
- Splits at root affect more samples → more important
- Splits near leaves affect fewer samples → less important

**In Random Forest**:
```python
# Aggregate across all trees
self.feature_importances_ = np.zeros(self.n_features_)
for tree in self.estimators_:
    self.feature_importances_ += tree.feature_importances_
self.feature_importances_ /= self.n_estimators
```

**Interpretation**:
- Higher value → feature used more often in informative splits
- Relative importance, not absolute
- Can be biased toward high-cardinality features

---

### 5. **Bootstrap Sampling - Creating Diversity**

**Implementation**:
```python
if self.bootstrap:
    idxs = self.rng.choice(n_samples, n_samples, replace=True)
    X_sample, y_sample = X[idxs], y[idxs]
else:
    X_sample, y_sample = X, y
```

**Key Property**: With replacement!
```python
# Example: 5 samples
idxs = [0, 2, 2, 4, 0]  # Sample 0 twice, sample 2 twice, sample 4 once
# Samples 1 and 3 not included (Out-of-Bag)
```

**Out-of-Bag (OOB) Error**:
- Each tree trained on ~63% of data
- Remaining ~37% can be used for validation
- No need for separate validation set!

```python
# OOB error estimation (not in current code, but valuable)
def oob_score(self):
    oob_predictions = np.zeros((n_samples, n_classes))
    oob_counts = np.zeros(n_samples)
    
    for tree, bootstrap_indices in zip(self.trees, self.bootstrap_samples):
        # Identify OOB samples
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bootstrap_indices] = False
        oob_indices = np.where(oob_mask)[0]
        
        # Predict on OOB samples
        if len(oob_indices) > 0:
            preds = tree.predict(X[oob_indices])
            oob_predictions[oob_indices] += preds
            oob_counts[oob_indices] += 1
    
    # Average predictions
    oob_predictions /= np.maximum(oob_counts[:, np.newaxis], 1)
    return accuracy(y, np.argmax(oob_predictions, axis=1))
```

---

### 6. **Ensemble Aggregation - Wisdom of Crowds**

**Classification (Majority Vote)**:
```python
def predict(self, X):
    # Get predictions from all trees
    tree_preds = np.array([tree.predict(X) for tree in self.estimators_])
    # Shape: (n_trees, n_samples)
    
    # Transpose for per-sample voting
    tree_preds = np.swapaxes(tree_preds, 0, 1)
    # Shape: (n_samples, n_trees)
    
    # Majority vote per sample
    predictions = []
    for preds in tree_preds:
        counts = np.bincount(preds.astype(int))
        predictions.append(np.argmax(counts))
    
    return np.array(predictions)
```

**Probability Estimation**:
```python
def predict_proba(self, X):
    # Average probabilities across trees
    tree_probs = np.array([tree.predict_proba(X) for tree in self.estimators_])
    # Shape: (n_trees, n_samples, n_classes)
    
    return np.mean(tree_probs, axis=0)
    # Shape: (n_samples, n_classes)
```

**Regression (Not shown, but similar)**:
```python
def predict(self, X):
    tree_preds = np.array([tree.predict(X) for tree in self.estimators_])
    return np.mean(tree_preds, axis=0)  # Simple average
```

---

## Potential Sticking Points While Coding

### 1. **Infinite Recursion**

**Problem**: Tree growth never stops

**Causes**:
- No stopping criteria
- Criteria never met (e.g., depth check missing)
- All same values but algorithm keeps trying to split

**Solution**: Multiple stopping conditions
```python
if (depth >= self.max_depth or          # Depth limit
    n_labels == 1 or                      # Pure node
    n_samples < self.min_samples_split):  # Too few samples
    return Node(value=leaf_value)
```

**Edge Case**: All features have same value but different labels
```python
# Can't split! Return majority class
if best_feat is None:
    return Node(value=self._most_common_label(y))
```

---

### 2. **Division by Zero in Entropy**

**Problem**: log₂(0) is undefined

**Scenario**:
```python
# Class distribution: [5, 0, 0]
# Probabilities: [1.0, 0.0, 0.0]
# log₂(0) = -∞ → NaN
```

**Solution**: Filter zero probabilities
```python
return -np.sum([p * np.log2(p) for p in ps if p > 0])
```

---

### 3. **Empty Splits**

**Problem**: Split produces empty left or right child

**Causes**:
- Threshold equals min or max value
- All samples go to one side

**Solution**: Check before recursing
```python
if len(left_idxs) == 0 or len(right_idxs) == 0:
    return 0  # Zero information gain
```

---

### 4. **Memory Explosion**

**Problem**: Storing many deep trees uses tons of memory

**Example**:
```python
# 1000 trees × 1000 nodes/tree × 100 bytes/node = 100 MB
# For large forests (10,000 trees), can be GBs!
```

**Solutions**:
1. Limit max_depth
2. Increase min_samples_split
3. Use model compression
4. Prune trees post-training

---

### 5. **Slow Training**

**Bottleneck**: Finding best split is O(n × d × n log n) per node

**Optimization Strategies**:

**1. Threshold Sampling** (implemented):
```python
if len(thresholds) > 100:
    thresholds = np.percentile(X_column, np.linspace(0, 100, 50))
```

**2. Feature Subsampling**:
```python
feat_idxs = self.rng.choice(n_feats, self.n_features, replace=False)
# Only consider subset of features per split
```

**3. Parallel Tree Building**:
```python
from multiprocessing import Pool

def train_tree(args):
    X, y, params = args
    tree = DecisionTree(**params)
    tree.fit(X, y)
    return tree

with Pool(n_jobs) as pool:
    self.estimators_ = pool.map(train_tree, tree_args)
```

**4. Histogram-based Splits** (advanced):
- Pre-bin continuous features
- Used in LightGBM, XGBoost
- Major speedup with minimal accuracy loss

---

### 6. **Feature Importance Biases**

**Problem 1**: High-cardinality features get inflated importance

**Reason**: More unique values → more potential splits → higher chance of appearing

**Example**:
```python
# ID column: 10,000 unique values
# Age column: 100 unique values
# ID will have higher importance even if meaningless!
```

**Solution**: Permutation importance (post-hoc)
```python
def permutation_importance(model, X, y):
    baseline_score = model.score(X, y)
    importances = []
    
    for feature_idx in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, feature_idx])  # Destroy feature
        permuted_score = model.score(X_permuted, y)
        importance = baseline_score - permuted_score
        importances.append(importance)
    
    return np.array(importances)
```

**Problem 2**: Correlated features split importance

**Reason**: Model randomly picks between correlated features

**Solution**: Cluster features, pick representatives

---

## Advanced Techniques

### 1. **Class Weight Balancing**

**Problem**: Imbalanced classes (e.g., 95% class 0, 5% class 1)

**Solution**: Weight samples inversely to class frequency
```python
def fit(self, X, y, sample_weight=None):
    if sample_weight is None:
        # Compute balanced weights
        class_counts = np.bincount(y)
        sample_weight = len(y) / (len(class_counts) * class_counts[y])
    
    # Use weights in information gain calculation
    # Modify entropy to weighted entropy
```

---

### 2. **Extremely Randomized Trees (Extra Trees)**

**Difference**: Random thresholds instead of optimal

**Implementation**:
```python
# Instead of testing all thresholds
for threshold in thresholds:
    gain = self._information_gain(y, X_column, threshold)

# Pick random threshold
min_val, max_val = X_column.min(), X_column.max()
threshold = np.random.uniform(min_val, max_val)
gain = self._information_gain(y, X_column, threshold)
```

**Benefits**:
- Faster training (no optimal search)
- More randomness → less overfitting
- Often similar accuracy

---

### 3. **Weighted Random Forest**

**Idea**: Weight trees by performance

**Implementation**:
```python
# During training, track OOB error per tree
tree_weights = []
for tree in self.estimators_:
    oob_acc = compute_oob_accuracy(tree)
    tree_weights.append(oob_acc)

# Prediction with weights
def predict(self, X):
    tree_preds = [tree.predict(X) for tree in self.estimators_]
    # Weighted vote instead of majority
    weighted_votes = np.average(tree_preds, axis=0, weights=self.tree_weights)
    return np.argmax(weighted_votes, axis=1)
```

---

### 4. **Confidence Estimation**

**Variance across trees**:
```python
def predict_with_uncertainty(self, X):
    tree_preds = np.array([tree.predict(X) for tree in self.estimators_])
    
    predictions = np.mean(tree_preds, axis=0)
    uncertainty = np.std(tree_preds, axis=0)
    
    return predictions, uncertainty
```

**High uncertainty** = Trees disagree = Low confidence

---

## Debugging Checklist

### Poor Accuracy
- [ ] Enough trees? (Try 200-500)
- [ ] Trees too shallow? (Increase max_depth)
- [ ] Too much regularization? (Decrease min_samples_split)
- [ ] Feature importance makes sense?
- [ ] Try increasing max_features
- [ ] Check for data leakage

### Overfitting
- [ ] Decrease max_depth (try 5-15)
- [ ] Increase min_samples_split (try 10-50)
- [ ] Reduce max_features
- [ ] Cross-validation score?
- [ ] Learning curve shows high variance?

### Slow Training
- [ ] Too many trees? (Start with 100)
- [ ] Trees too deep? (Limit to 20)
- [ ] Too many features? (Feature selection)
- [ ] Enable parallel training
- [ ] Threshold sampling working?

### High Memory Usage
- [ ] Reduce n_estimators
- [ ] Reduce max_depth
- [ ] Increase min_samples_split
- [ ] Consider model compression

---

## Comparison: Single Tree vs. Random Forest

| Aspect | Single Decision Tree | Random Forest |
|--------|---------------------|---------------|
| **Overfitting** | High (fits noise) | Low (averaging smooths) |
| **Interpretability** | High (visual tree) | Low (many trees) |
| **Training Time** | Fast | Slow (many trees) |
| **Prediction Time** | Fast | Slow (query all) |
| **Accuracy** | Moderate | High |
| **Variance** | High | Low |
| **Bias** | Low (can be complex) | Slightly higher |
| **Stability** | Unstable (small data change → big tree change) | Stable |

---

## Real-World Applications

1. **Finance**: Credit scoring, fraud detection
2. **Healthcare**: Disease diagnosis, patient risk stratification
3. **E-commerce**: Customer churn, product recommendation
4. **Marketing**: Click-through rate prediction
5. **Manufacturing**: Quality control, predictive maintenance
6. **Ecology**: Species classification, habitat modeling
7. **Computer Vision**: Object detection (as feature classifier)
8. **Kaggle**: Ensemble methods dominate competitions

---

## Key Takeaways

✅ **Ensemble Power**: Multiple weak learners → strong learner
✅ **Robustness**: Handles various data types, missing values (with modifications), outliers
✅ **Feature Importance**: Automatic feature selection/ranking
✅ **No Scaling**: Tree-based methods invariant to feature scales
✅ **Parallelizable**: Trees independent, train in parallel

❌ **Black Box**: Hard to interpret compared to single tree
❌ **Memory**: Stores all trees
❌ **Slow Prediction**: Must query all trees
❌ **No Extrapolation**: Can't predict beyond training range

**When to Use**:
- Need high accuracy without much tuning
- Mixed feature types
- Non-linear relationships
- Feature importance needed
- Interpretability less critical

**When to Avoid**:
- Need full interpretability
- Extremely large datasets (consider gradient boosting)
- Real-time prediction with strict latency
- Memory constrained

**Recommended Workflow**:
1. Start with default parameters (100 trees, max_depth=10)
2. Check feature importance (remove noise)
3. Tune max_depth and min_samples_split with cross-validation
4. Increase n_estimators until performance plateaus
5. Adjust max_features for variance-bias trade-off
6. Consider class weights for imbalanced data
7. Use OOB score to avoid separate validation set
