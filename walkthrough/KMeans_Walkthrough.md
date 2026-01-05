# K-Means Clustering - Walkthrough

## Overview
Want to group similar things together? K-Means does exactly that. Pick K random starting points, assign each data point to its nearest center, recalculate centers, repeat until nothing changes. Simple but effective!

## Parameters

### 1. **n_clusters (k)**
- **Type**: Integer
- **Default**: 8
- **Range**: 1 to N (number of samples)
- **Purpose**: Number of clusters to form
- **Selection Methods**:
  - **Elbow Method**: Plot inertia vs. k, find "elbow" point
  - **Silhouette Score**: Measure cluster quality
  - **Domain Knowledge**: Business requirements
  - **Gap Statistic**: Compare with random data
- **Effect**:
  - **Too few**: Under-segmentation, high inertia
  - **Too many**: Over-segmentation, low interpretability

### 2. **max_iter**
- **Type**: Integer
- **Default**: 300
- **Range**: 10 to 1000+
- **Purpose**: Maximum number of iterations to run
- **When needed**:
  - Large datasets: May need more iterations
  - Random initialization: Some runs converge slower
  - Complex shapes: Non-spherical clusters need more iterations
- **Convergence**: Algorithm usually converges before max_iter

### 3. **tol (tolerance)**
- **Type**: Float
- **Default**: 1e-4
- **Range**: 1e-6 to 1e-2
- **Purpose**: Convergence threshold for centroid movement
- **Formula**: `shift = Σ(old_center - new_center)²`
- **Effect**:
  - **Small (1e-6)**: More precise, slower
  - **Large (1e-2)**: Less precise, faster
- **Use case**: Adjust based on speed vs. accuracy trade-off

### 4. **init (initialization method)**
- **Type**: String
- **Options**: `'k-means++'` or `'random'`
- **Purpose**: How to choose initial centroids
- **k-means++ (recommended)**:
  - Smart initialization
  - Spreads out initial centroids
  - Faster convergence
  - Better final results
- **random**:
  - Uniform random selection
  - Fast but suboptimal
  - May get stuck in local minima
- **Impact**: Can drastically affect final clustering!

### 5. **n_init**
- **Type**: Integer
- **Default**: 10
- **Range**: 1 to 100
- **Purpose**: Number of times to run algorithm with different initializations
- **Why needed**: K-means finds local minima, not global
- **Best result selected by lowest inertia**
- **Trade-off**: Higher n_init = better results but slower

---

## Algorithm Attributes

### **cluster_centers_**
- **Type**: ndarray of shape (n_clusters, n_features)
- **Content**: Coordinates of cluster centers (centroids)
- **Use**: Assign new points, visualize clusters

### **labels_**
- **Type**: ndarray of shape (n_samples,)
- **Content**: Cluster assignment for each sample
- **Values**: 0 to k-1

### **inertia_**
- **Type**: Float
- **Formula**: Σ(distance from point to its assigned center)²
- **Purpose**: Measure of cluster compactness
- **Lower is better** (but decreases with more clusters)
- **Used for**: Elbow method, comparing runs

### **n_iter_**
- **Type**: Integer
- **Content**: Actual iterations run before convergence
- **Use**: Diagnose convergence issues

---

## What K-Means Can Do

### Applications
1. **Customer Segmentation**: Group customers by behavior
2. **Image Compression**: Reduce colors in image
3. **Document Clustering**: Group similar documents
4. **Anomaly Detection**: Points far from centroids are anomalies
5. **Feature Engineering**: Cluster membership as new feature
6. **Data Preprocessing**: Stratified sampling by clusters
7. **Vector Quantization**: Compress high-dimensional data

### Advantages
✅ **Simple**: Easy to understand and implement
✅ **Fast**: O(n × k × d × iterations) - linear in n
✅ **Scalable**: Works on large datasets
✅ **Guaranteed Convergence**: Always converges (to local minimum)
✅ **Interpretable**: Centroids represent cluster "prototypes"

### The Gotchas

**1. Choosing K is hard**
Use the elbow method: plot inertia vs. K, look for the bend. Or use silhouette score.

**2. Feature scaling is mandatory**
Same as KNN - large-scale features will dominate.

**3. Only finds spherical clusters**
If your clusters are moon-shaped or weird, K-Means will fail. Try DBSCAN instead.

**4. Sensitive to outliers**
One crazy outlier will pull a centroid way off.

**5. Gets stuck in local minima**
That's why we run it multiple times (n_init parameter).

---

## Critical Implementation Insights

### 1. **K-Means++ Initialization - The Game Changer**

**Standard Random Initialization** (problematic):
```python
# Naive approach
indices = np.random.choice(n_samples, k, replace=False)
centers = X[indices]
```

**Problems**:
- Centers may be close together
- Poor coverage of data space
- Slow convergence
- Local minima

**K-Means++ Algorithm** (smart):
```python
def _init_centroids(self, X):
    n_samples = X.shape[0]
    centers = []
    
    # 1. Choose first center uniformly at random
    first_idx = np.random.randint(n_samples)
    centers.append(X[first_idx])
    
    # 2. For each remaining center
    for _ in range(1, self.n_clusters):
        # Compute distance to nearest existing center
        C_current = np.array(centers)
        curr_dists = np.sum((X[:, np.newaxis, :] - C_current[np.newaxis, :, :])**2, axis=2)
        min_dists_sq = np.min(curr_dists, axis=1)
        
        # Probability proportional to squared distance
        probs = min_dists_sq / np.sum(min_dists_sq)
        
        # Select next center with these probabilities
        next_idx = np.random.choice(n_samples, p=probs)
        centers.append(X[next_idx])
    
    return np.array(centers)
```

**Why It Works**:
- Points far from existing centers more likely to be chosen
- Spreads centroids across data space
- Provably O(log k) approximation to optimal
- Empirically: 2-3x fewer iterations, better results

**Key Insight**: Probability proportional to D(x)² ensures good spread

---

### 2. **Vectorized Distance Computation**

**Naive Approach** (slow):
```python
dists = np.zeros((n_samples, n_clusters))
for i in range(n_samples):
    for j in range(n_clusters):
        dists[i, j] = np.sqrt(np.sum((X[i] - centers[j])**2))
```

**Optimized Approach** (fast):
```python
def _compute_distances(self, X, centers):
    # Euclidean distance: ||x - c||² = ||x||² + ||c||² - 2·x·c
    X_sq = np.sum(X**2, axis=1, keepdims=True)      # (n, 1)
    C_sq = np.sum(centers**2, axis=1)                # (k,)
    dists = np.sqrt(np.maximum(
        X_sq + C_sq - 2 * np.dot(X, centers.T),
        0
    ))
    return dists
```

**Why Maximum?**: Floating-point errors can make very small values negative
```python
# Example: Should be 0 but floating-point gives -1e-15
np.sqrt(-1e-15)  # NaN!
np.sqrt(max(-1e-15, 0))  # 0.0 ✓
```

**Performance Gain**: 100-1000x faster using BLAS-optimized matrix multiplication

---

### 3. **The Main Algorithm Loop**

**Conceptual Flow**:
1. **Assign** samples to nearest centroid
2. **Update** centroids to mean of assigned samples
3. **Repeat** until convergence

**Implementation**:
```python
def _run_kmeans(self, X):
    centers = self._init_centroids(X)
    labels = None
    
    for iteration in range(self.max_iter):
        # E-step: Assign to nearest centroid
        dists = self._compute_distances(X, centers)
        new_labels = np.argmin(dists, axis=1)
        
        # Check convergence
        if labels is not None and np.all(labels == new_labels):
            # No changes, converged!
            inertia = np.sum(np.min(dists, axis=1) ** 2)
            return centers, new_labels, inertia, iteration + 1
        
        labels = new_labels
        
        # M-step: Update centroids
        new_centers = np.zeros_like(centers)
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centers[k] = np.mean(X[mask], axis=0)
            else:
                # Empty cluster! Reinitialize
                new_centers[k] = X[np.random.choice(len(X))]
        
        # Check tolerance
        shift = np.sum((centers - new_centers) ** 2)
        centers = new_centers
        
        if shift < self.tol:
            # Converged by tolerance
            dists = self._compute_distances(X, centers)
            inertia = np.sum(np.min(dists, axis=1) ** 2)
            return centers, labels, inertia, iteration + 1
    
    # Max iterations reached
    dists = self._compute_distances(X, centers)
    inertia = np.sum(np.min(dists, axis=1) ** 2)
    return centers, labels, inertia, self.max_iter
```

**Key Points**:
- Two convergence criteria: labels unchanged OR shift < tolerance
- Handle empty clusters explicitly
- Recompute inertia for final result

---

### 4. **Empty Cluster Handling - Critical Edge Case**

**Problem**: Sometimes all points assigned away from a centroid

**Causes**:
- Bad initialization
- Outlier centroid
- Cluster absorbed by neighbors

**Naive Approach** (breaks):
```python
new_centers[k] = np.mean(X[mask], axis=0)
# When mask is all False: mean of empty array = NaN!
```

**Robust Solution**:
```python
if np.any(mask):
    new_centers[k] = np.mean(X[mask], axis=0)
else:
    # Reinitialize with random point
    new_centers[k] = X[np.random.choice(n_samples)]
    print(f"Warning: Cluster {k} became empty, reinitializing")
```

**Alternative Strategies**:
1. **Split largest cluster**: Find cluster with highest inertia, split in two
2. **Furthest point**: Place at point furthest from all centers
3. **Remove cluster**: Continue with k-1 clusters
4. **Restart**: Abort run, try different initialization

**Best Practice**: Use k-means++ to minimize occurrence

---

### 5. **Multiple Runs with n_init**

**Why Needed**: K-means finds local minima

**Example**: Same data, different initializations → different results

**Implementation**:
```python
def fit(self, X):
    best_inertia = np.inf
    
    for run in range(self.n_init):
        centers, labels, inertia, n_iter = self._run_kmeans(X)
        
        if inertia < best_inertia:
            best_inertia = inertia
            self.cluster_centers_ = centers
            self.labels_ = labels
            self.inertia_ = inertia
            self.n_iter_ = n_iter
    
    return self
```

**Trade-off**: More runs = better results but slower

**Recommendation**:
- n_init=10 for small datasets (<10K samples)
- n_init=3-5 for medium datasets (10K-100K)
- n_init=1 with k-means++ for large datasets (>100K)

---

## Potential Sticking Points While Coding

### 1. **Curse of Dimensionality**

**Problem**: In high dimensions, distances become meaningless

**Example**:
```python
# 100-dimensional random data
X = np.random.randn(1000, 100)
dists = cdist(X[:10], X, 'euclidean')

print(f"Min: {dists.min():.2f}, Max: {dists.max():.2f}, "
      f"Mean: {dists.mean():.2f}, Std: {dists.std():.2f}")
# Output: Min: 8.5, Max: 11.5, Mean: 10.0, Std: 0.3
# All points are nearly equidistant!
```

**Why It Happens**: Volume concentrates at surface of hypersphere

**Solutions**:
1. **Dimensionality Reduction**: PCA before clustering
2. **Feature Selection**: Keep only informative features
3. **Alternative Algorithms**: DBSCAN, Hierarchical Clustering
4. **Metric Learning**: Learn better distance metric

---

### 2. **Feature Scaling is Mandatory**

**Problem**: Features with larger ranges dominate distance

**Example**:
```python
# Feature 1: Age (20-80), Feature 2: Income (20,000-200,000)
# Distance dominated by income!
point_a = [25, 50000]
point_b = [30, 51000]
distance = np.sqrt((30-25)**2 + (51000-50000)**2) ≈ 1000
# Age difference contributes only 25 out of 1000!
```

**Solution**: Always scale before clustering
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_scaled)
```

**Common Mistake**: Forgetting to scale test data with training scaler!

---

### 3. **Non-Spherical Clusters**

**Problem**: K-means assumes spherical (Gaussian) clusters

**Fails On**:
- **Elongated clusters**: Ellipses, lines
- **Concentric circles**: Rings
- **Moon shapes**: Crescents
- **Arbitrary shapes**: S-curves, spirals

**Example**:
```python
# Two moons dataset
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
# Result: Splits each moon in half instead of separating them!
```

**Solutions**:
1. **Kernel K-means**: Project to higher dimensions
2. **Spectral Clustering**: Use graph-based similarity
3. **DBSCAN**: Density-based, finds arbitrary shapes
4. **GMM**: Gaussian Mixture Models (soft clustering, ellipses)

---

### 4. **Choosing Optimal K - The Elbow Method**

**Implementation**:
```python
def elbow_method(X, k_range=range(1, 11)):
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()
    
    return inertias
```

**Visual Interpretation**:
```
Inertia
  |
  |●
  |  ●
  |    ●
  |      ●___●___●___●  ← Elbow here (k=4)
  |__________________________ k
   1  2  3  4  5  6  7  8
```

**Programmatic Elbow Detection**:
```python
def find_elbow(inertias):
    # Method 1: Maximum curvature (2nd derivative)
    deltas = np.diff(inertias)
    diffs = np.diff(deltas)
    elbow_idx = np.argmax(diffs) + 2  # +2 for indexing offset
    
    return elbow_idx

# Method 2: Kneedle algorithm (more sophisticated)
from kneed import KneeLocator
kl = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
optimal_k = kl.elbow
```

**Key Insight**: Look for point where adding more clusters gives diminishing returns

---

### 5. **Silhouette Score - Better than Elbow**

**Formula**: For each point i:
```
a(i) = average distance to points in same cluster
b(i) = average distance to points in nearest other cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Silhouette Score**: Average s(i) over all points
- Range: [-1, 1]
- +1: Perfect clustering
- 0: Overlapping clusters
- -1: Incorrect assignments

**Implementation**:
```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

best_k = np.argmax(scores) + 2  # +2 for range offset
```

**Advantages over Inertia**:
- Considers both cohesion and separation
- Accounts for cluster shape
- Peak value easier to identify than elbow

---

### 6. **Outlier Sensitivity**

**Problem**: Outliers pull centroids away from true cluster centers

**Example**:
```python
# Cluster with outlier
cluster = np.random.randn(100, 2)  # Mean at (0, 0)
outlier = np.array([[50, 50]])
X = np.vstack([cluster, outlier])

kmeans = KMeans(n_clusters=1)
kmeans.fit(X)
print(kmeans.cluster_centers_)
# Center pulled toward outlier: [[0.5, 0.5]] instead of [[0, 0]]
```

**Solutions**:
1. **Outlier Removal**: Preprocess to remove outliers
2. **Robust K-means**: Use median instead of mean
3. **DBSCAN**: Density-based, naturally handles outliers
4. **GMM with Full Covariance**: More flexible

**Robust K-means** (not standard):
```python
# Replace mean with median
new_centers[k] = np.median(X[mask], axis=0)
```

---

### 7. **Convergence Issues**

**Symptom**: Reaches max_iter without converging

**Causes**:
1. Tolerance too strict
2. max_iter too small
3. Pathological data
4. Poor initialization

**Debugging**:
```python
kmeans = KMeans(n_clusters=k, verbose=True)
kmeans.fit(X)

if kmeans.n_iter_ == kmeans.max_iter:
    print("Warning: Did not converge!")
    print(f"Final shift: {final_shift}")
```

**Solutions**:
- Increase max_iter
- Loosen tolerance
- Try different initialization
- Check for degenerate data (e.g., duplicate points)

---

## Advanced Techniques

### 1. **Mini-Batch K-Means**

**Problem**: Standard K-means slow on huge datasets

**Solution**: Update centroids using mini-batches
```python
class MiniBatchKMeans:
    def fit(self, X, batch_size=100):
        centers = self._init_centroids(X)
        
        for epoch in range(max_epochs):
            for batch in get_batches(X, batch_size):
                # Assign batch to nearest centroids
                labels = self.predict(batch)
                
                # Update centroids incrementally
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if np.any(mask):
                        # Moving average update
                        centers[k] = (1-lr) * centers[k] + lr * np.mean(batch[mask], axis=0)
        
        return centers
```

**Trade-offs**:
- 10-100x faster
- Slightly worse clustering quality
- Good for streaming data

---

### 2. **Fuzzy C-Means (Soft Clustering)**

**Difference**: Points can belong to multiple clusters with probabilities

**Update Rule**:
```python
# Membership matrix: U[i, k] = probability point i in cluster k
U = compute_memberships(X, centers, fuzziness=2)

# Update centers using weighted mean
for k in range(n_clusters):
    weights = U[:, k] ** fuzziness
    centers[k] = np.sum(weights[:, np.newaxis] * X, axis=0) / np.sum(weights)
```

**Use Case**: Overlapping clusters, uncertainty quantification

---

### 3. **K-Means for Image Compression**

**Idea**: Reduce color palette by clustering pixel colors

**Implementation**:
```python
from PIL import Image

# Load image
img = np.array(Image.open('photo.jpg'))
h, w, c = img.shape

# Reshape to (n_pixels, 3)
X = img.reshape(-1, 3) / 255.0

# Cluster colors
kmeans = KMeans(n_clusters=16)  # 16-color palette
kmeans.fit(X)

# Replace pixels with nearest centroid
compressed = kmeans.cluster_centers_[kmeans.labels_]
compressed_img = (compressed * 255).astype(np.uint8).reshape(h, w, c)

# Save
Image.fromarray(compressed_img).save('compressed.jpg')
```

**Result**: Reduces file size while preserving visual quality

---

### 4. **Hierarchical Initialization**

**Idea**: Use hierarchical clustering to initialize K-means

**Benefit**: Better initial centroids than k-means++

**Implementation**:
```python
from scipy.cluster.hierarchy import linkage, fcluster

# Hierarchical clustering on subset (faster)
sample_indices = np.random.choice(len(X), min(1000, len(X)), replace=False)
X_sample = X[sample_indices]

Z = linkage(X_sample, method='ward')
labels = fcluster(Z, k, criterion='maxclust')

# Use cluster means as initial centroids
initial_centers = np.array([X_sample[labels == i].mean(axis=0) 
                             for i in range(1, k+1)])

kmeans = KMeans(n_clusters=k, init=initial_centers, n_init=1)
```

---

## Comparison: K-Means vs. Other Clustering

| Algorithm | Shape | Outliers | K Selection | Speed |
|-----------|-------|----------|-------------|-------|
| **K-Means** | Spherical | Sensitive | Manual | Fast O(nkd) |
| **DBSCAN** | Arbitrary | Robust | Automatic | Medium O(n log n) |
| **Hierarchical** | Flexible | Moderate | Dendrogram | Slow O(n²) |
| **GMM** | Ellipsoidal | Sensitive | Manual/BIC | Slow O(nk²d²) |
| **Mean Shift** | Arbitrary | Robust | Automatic | Very Slow O(n²) |

---

## Real-World Considerations

### 1. **Interpretability**
```python
# Cluster centers are interpretable prototypes
for k, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {k}: {feature_names}")
    print(f"  Values: {center}")
    # Example: Cluster 0 - "High spenders, young age"
```

### 2. **Actionability**
```python
# Assign new customers to segments
new_customer = [[25, 50000, 5]]  # age, income, years
cluster_id = kmeans.predict(new_customer)
print(f"Assign to segment {cluster_id}")
# Apply targeted marketing strategy for this segment
```

### 3. **Stability**
```python
# Bootstrap to assess stability
from sklearn.utils import resample

cluster_assignments = []
for _ in range(100):
    X_boot = resample(X)
    kmeans_boot = KMeans(n_clusters=k)
    labels = kmeans_boot.fit_predict(X_boot)
    cluster_assignments.append(labels)

# High variance in assignments = unstable clustering
```

---

## Key Takeaways

✅ **Simple & Fast**: Easy to implement, scales well
✅ **Interpretable**: Centroids represent cluster prototypes
✅ **Versatile**: Customer segmentation, compression, preprocessing
✅ **Guaranteed Convergence**: Always reaches a solution

❌ **Requires K**: Must specify number of clusters
❌ **Local Minima**: Sensitive to initialization (use k-means++)
❌ **Spherical Assumption**: Struggles with irregular shapes
❌ **Scaling Required**: Features must be normalized
❌ **Outlier Sensitive**: Outliers distort centroids

**When to Use**:
- Need fast clustering on large datasets
- Clusters are roughly spherical
- K is known or can be estimated
- Interpretability important

**When to Avoid**:
- Unknown number of clusters
- Non-convex cluster shapes
- Noisy data with many outliers
- Need hierarchical structure

**Best Practices**:
1. Always scale features
2. Use k-means++ initialization
3. Run multiple times (n_init=10)
4. Use elbow method + silhouette score for K
5. Visualize clusters (PCA for high-D)
6. Validate with domain knowledge
7. Check cluster sizes (very unbalanced = problem)
8. Consider alternatives for non-spherical data
