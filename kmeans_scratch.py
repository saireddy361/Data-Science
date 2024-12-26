import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=100, tolerance=1e-4):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tolerance
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign clusters
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i] for i in range(self.k)])

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            # Calculate the Euclidean distance between each point and each centroid
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return distances

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

# Example Usage
# Assuming you have a dataset loaded into numpy array `X`
X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8], [8, 9]])

model = KMeansScratch(n_clusters=3, max_iters=100)
model.fit(X)  # Fit the model to the dataset
labels = model.predict(X)  # Predict the cluster labels for the dataset

print(f"Cluster labels: {labels}")
print(f"Centroids: {model.centroids}")
