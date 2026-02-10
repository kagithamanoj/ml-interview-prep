"""
Python ML Coding Patterns — Common Interview Coding Tasks
Run this file to see output from all examples.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPLEMENT LINEAR REGRESSION FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════════

class LinearRegression:
    """Linear regression using gradient descent."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Gradients
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IMPLEMENT LOGISTIC REGRESSION FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticRegression:
    """Binary logistic regression using gradient descent."""

    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IMPLEMENT K-NEAREST NEIGHBORS FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════════

class KNN:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            # Compute distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Get k nearest indices
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self.y_train[k_indices]

            # Majority vote
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])

        return np.array(predictions)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IMPLEMENT K-MEANS CLUSTERING FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════════

class KMeans:
    """K-Means clustering."""

    def __init__(self, k: int = 3, max_iters: int = 100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]

        # Random initialization
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)

            # Update centroids
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0
                else self.centroids[i]
                for i in range(self.k)
            ])

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return labels

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        distances = np.array([
            np.sqrt(np.sum((X - c) ** 2, axis=1)) for c in self.centroids
        ])
        return np.argmin(distances, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(X)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMMON NUMPY/PANDAS PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def common_patterns():
    """Common data manipulation patterns asked in interviews."""

    # --- Softmax ---
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return exp_x / exp_x.sum()

    # --- Cross-entropy loss ---
    def cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # --- Cosine similarity ---
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # --- Confusion matrix ---
    def confusion_matrix(y_true, y_pred):
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1
        return cm

    # --- Train/test split ---
    def train_test_split(X, y, test_size=0.2, seed=42):
        np.random.seed(seed)
        indices = np.random.permutation(len(X))
        split = int(len(X) * (1 - test_size))
        return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

    return {
        "softmax": softmax,
        "cross_entropy": cross_entropy,
        "cosine_similarity": cosine_similarity,
        "confusion_matrix": confusion_matrix,
        "train_test_split": train_test_split,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    # --- Linear Regression ---
    print("=" * 50)
    print("1. Linear Regression")
    X = np.random.randn(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5
    model = LinearRegression(lr=0.1, epochs=500)
    model.fit(X, y)
    print(f"   Learned: w={model.weights[0]:.2f} (true=3), b={model.bias:.2f} (true=2)")
    print(f"   MSE: {model.mse(X, y):.4f}")

    # --- Logistic Regression ---
    print("\n2. Logistic Regression")
    X_cls = np.vstack([np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) - [2, 2]])
    y_cls = np.array([1] * 50 + [0] * 50)
    lr = LogisticRegression(lr=0.1, epochs=500)
    lr.fit(X_cls, y_cls)
    print(f"   Accuracy: {lr.accuracy(X_cls, y_cls):.1%}")

    # --- KNN ---
    print("\n3. K-Nearest Neighbors")
    knn = KNN(k=5)
    knn.fit(X_cls, y_cls)
    preds = knn.predict(X_cls[:10])
    print(f"   First 10 predictions: {preds}")
    print(f"   First 10 actual:      {y_cls[:10]}")

    # --- K-Means ---
    print("\n4. K-Means Clustering")
    X_km = np.vstack([np.random.randn(30, 2) + [4, 4], np.random.randn(30, 2) - [4, 4], np.random.randn(30, 2)])
    km = KMeans(k=3)
    labels = km.fit(X_km)
    print(f"   Cluster sizes: {[np.sum(labels == i) for i in range(3)]}")

    # --- Common Patterns ---
    print("\n5. Common Patterns")
    patterns = common_patterns()
    print(f"   softmax([1,2,3]): {patterns['softmax'](np.array([1.0, 2.0, 3.0]))}")
    print(f"   cosine_sim([1,0], [1,1]): {patterns['cosine_similarity'](np.array([1,0]), np.array([1,1])):.4f}")

    print("\n✅ All examples complete!")
