import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, init='k-means++', n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        
        # attributes
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X):
        X = np.array(X)
        best_inertia = np.inf
        
        for i in range(self.n_init):
            centers, labels, inertia, n_iter = self._run_kmeans(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter
                
        return self

    def predict(self, X):
        X = np.array(X)
        dists = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _run_kmeans(self, X):
        n_samples, n_features = X.shape
        centers = self._init_centroids(X)
        labels = None
        inertia = None
        
        for i in range(self.max_iter):
            # assignment
            dists = self._compute_distances(X, centers)
            new_labels = np.argmin(dists, axis=1)
            
            # convergence check
            if labels is not None and np.all(labels == new_labels):
                # recalculate inertia one last time
                min_dists = np.min(dists, axis=1)
                inertia = np.sum(min_dists ** 2)
                return centers, new_labels, inertia, i + 1
            
            labels = new_labels
            
            # update Centers
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    new_centers[k] = np.mean(X[mask], axis=0)
                else:
                    # robust empty cluster handling: Reinitialize random
                    new_centers[k] = X[np.random.choice(n_samples)]
            
            # check shift tolerance
            shift = np.sum((centers - new_centers) ** 2)
            centers = new_centers
            
            if shift < self.tol:
                # Recalculate inertia
                dists = self._compute_distances(X, centers)
                min_dists = np.min(dists, axis=1)
                inertia = np.sum(min_dists ** 2)
                return centers, labels, inertia, i + 1
        
        # max iter reached
        dists = self._compute_distances(X, centers)
        min_dists = np.min(dists, axis=1)
        inertia = np.sum(min_dists ** 2)
        return centers, labels, inertia, self.max_iter

    def _compute_distances(self, X, centers):
        # vectorized Euclidean Distance: (x-c)^2 = x^2 + c^2 - 2xc
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        C_sq = np.sum(centers**2, axis=1)
        dists = np.sqrt(np.maximum(X_sq + C_sq - 2 * np.dot(X, centers.T), 0))
        return dists

    def _init_centroids(self, X):
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices]
            
        elif self.init == 'k-means++':
            centers = []
            # 1. choose first center uniformly
            first_idx = np.random.randint(n_samples)
            centers.append(X[first_idx])
            
            # 2. choose remaining k-1 centers
            for _ in range(1, self.n_clusters):
                # calculate squared distance to nearest existing center
                # use simple loop here since k is usually small during init
                dists_sq = np.array([np.min(np.sum((X - c)**2, axis=1)) for c in centers])
                # wait, this loop logic is slightly wrong for multiple centers.
                # correct way: D(x) = min distance to ANY center
                
                # let's vectorize init dists properly
                # current centers matrix
                C_current = np.array(centers)
                # compute distances from all X to all C_current
                # reuse vectorized function but handle shape mismatch manually or just use norm
                # since init loop runs k times, let's keep it robust but optimized enough
                
                # dists from X to all current centers
                # just use broadcasting for correctness in init
                # (N, 1, D) - (1, K_curr, D)
                curr_dists = np.sum((X[:, np.newaxis, :] - C_current[np.newaxis, :, :])**2, axis=2)
                min_dists_sq = np.min(curr_dists, axis=1)
                
                # Probabilities
                probs = min_dists_sq / np.sum(min_dists_sq)
                
                # select next center
                next_idx = np.random.choice(n_samples, p=probs)
                centers.append(X[next_idx])
                
            return np.array(centers)
        else:
            raise ValueError("Init must be 'random' or 'k-means++'")

if __name__ == "__main__":
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, '../data', 'unsupervised_data.csv')

    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        # Load data (assuming headers exist based on file inspection: ID, Feature_1...)
        df = pd.read_csv(data_path)
        
        # drop ID if it exists and is not a feature
        if 'ID' in df.columns:
            X = df.drop('ID', axis=1).values
        else:
            X = df.values
            
        print(f"Data shape: {X.shape}")

        # 1. Elbow Method to find optimal K
        print("Running Elbow Method (k=1 to 10)...")
        inertias = []
        K_range = range(1, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, max_iter=100)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}")
            
        # Plot Elbow Curve
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertias, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.grid(True)
        # plt.show()
        plt.savefig('kmeans_elbow_plot.png')
        print("Elbow plot saved to kmeans_elbow_plot.png")

        # determine "elbow" programmatically (simple heuristic: max 2nd derivative)
        # or just pick a reasonable default based on typical tasks if unclear
        # Let's assume k=3 or 4 is likely given the synthetic look, but I'll pick the one 
        # with the most significant drop.
        # for automation, let's just pick k=5 as a safe bet if not interactive, 
        # or better yet, analyze the diffs.
        
        deltas = np.diff(inertias)
        diffs = np.diff(deltas)
        # the index with the max acceleration (curvature) is often the elbow
        # k_range is 1..10. diffs corresponds to k=2..9.
        # acceleration[i] corresponds to k[i+1] -> k=2 is index 0
        elbow_idx = np.argmax(diffs) + 2 
        best_k = elbow_idx 
        
        # soft-limit scaling if the heuristic fails (e.g. k=2 is often simple)
        if best_k < 2: best_k = 3
        
        print(f"Estimated optimal k: {best_k}")

        # 2. Train Final Model
        print(f"Training Final KMeans with k={best_k}...")
        final_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, max_iter=300)
        final_kmeans.fit(X)
        
        print("Cluster Centers:")
        print(final_kmeans.cluster_centers_)
        print(f"Final Inertia: {final_kmeans.inertia_:.2f}")
        
        # save results with labels 
        # (Optional: append labels to dataframe and save)
        # df['Cluster'] = final_kmeans.labels_
        # df.to_csv('unsupervised_data_clustered.csv', index=False)
        # print("Saved clustered data.")

    else:
        # Fallback to synthetic
        print("Data file not found. Running synthetic demonstration.")
        np.random.seed(42)
        X = np.r_[
            np.random.randn(100, 2) + [2, 2],
            np.random.randn(100, 2) + [-2, -2],
            np.random.randn(100, 2) + [2, -2]
        ]
        
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=5)
        kmeans.fit(X)
        print(f"Inertia: {kmeans.inertia_:.4f}")
