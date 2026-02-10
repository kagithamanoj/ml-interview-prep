# Unsupervised Learning — Interview Reference

## Clustering Algorithms

### K-Means
**How it works**: Partition data into K clusters by iteratively updating centroids.

```
Algorithm:
1. Initialize K random centroids
2. Assign each point to nearest centroid
3. Update centroids = mean of assigned points
4. Repeat 2-3 until convergence
```

**Interview Questions:**
- *"Time complexity?"* → O(n × K × d × iterations)
- *"How to choose K?"* → Elbow method, silhouette score, domain knowledge
- *"Limitations?"* → Assumes spherical clusters, sensitive to initialization, must specify K
- *"How to handle initialization?"* → K-Means++ (smart initialization)

### DBSCAN
**How it works**: Density-based clustering — finds core points with ≥ minPts in ε radius.

```
Key concepts:
- Core point: ≥ minPts neighbors within ε
- Border point: within ε of a core point but < minPts neighbors
- Noise point: neither core nor border
```

**Interview Questions:**
- *"When to use over K-Means?"* → Non-spherical clusters, unknown K, noisy data
- *"Parameters?"* → ε (neighborhood radius), minPts (density threshold)
- *"Limitations?"* → Struggles with varying density, sensitive to ε and minPts

### Hierarchical Clustering
**How it works**: Build a tree (dendrogram) of clusters.

```
Agglomerative (bottom-up):
1. Start: each point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains

Linkage methods:
- Single: min distance between clusters
- Complete: max distance
- Average: mean distance
- Ward: minimizes variance
```

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)
**How it works**: Find directions of maximum variance (eigenvectors of covariance matrix).

```
Algorithm:
1. Center data (subtract mean)
2. Compute covariance matrix
3. Eigendecomposition → eigenvalues + eigenvectors
4. Keep top-k eigenvectors (components)
5. Project data onto these components
```

**Interview Questions:**
- *"When to use?"* → Feature reduction, visualization, denoising
- *"How many components?"* → Explained variance ratio (e.g., keep 95%)
- *"Assumptions?"* → Linear relationships, mean-centered data
- *"PCA vs t-SNE?"* → PCA: global structure, linear. t-SNE: local structure, non-linear

### t-SNE
**How it works**: Non-linear dimensionality reduction that preserves local neighborhoods.

```
Key concepts:
- Computes pairwise similarity in high-D (Gaussian)
- Computes pairwise similarity in low-D (t-distribution)
- Minimizes KL divergence between the two
```

**Interview Questions:**
- *"Why not use for ML features?"* → Non-deterministic, expensive, only for visualization
- *"Perplexity parameter?"* → Balances local vs global structure (~5-50)
- *"Why t-distribution?"* → Heavy tails prevent crowding in low-D

### UMAP
**How it works**: Similar to t-SNE but faster, preserves more global structure.

**Interview Questions:**
- *"UMAP vs t-SNE?"* → UMAP: faster, better global structure, scalable, can be used for feature engineering
- *"When to use?"* → Large datasets, need speed, want global + local structure

---

## Anomaly Detection

### Isolation Forest
```
How it works:
1. Build random trees by randomly selecting features and split points
2. Anomalies are isolated with fewer splits (shorter path length)
3. Score = avg path length across all trees (shorter = more anomalous)
```

### One-Class SVM
```
How it works:
1. Map data to high-D space via kernel
2. Find hyperplane that separates data from origin
3. Points on wrong side = anomalies
```

### LOF (Local Outlier Factor)
```
How it works:
1. Compare local density of a point to its neighbors
2. LOF > 1 → less dense than neighbors → anomaly
3. LOF ≈ 1 → similar density → normal
```

---

## Quick Reference Table

| Algorithm | Type | Pros | Cons |
|-----------|------|------|------|
| K-Means | Clustering | Fast, simple | Must choose K, spherical only |
| DBSCAN | Clustering | No K needed, finds noise | Density-dependent |
| Hierarchical | Clustering | Dendrogram visualization | O(n³) memory |
| PCA | Dim. Reduction | Fast, interpretable | Linear only |
| t-SNE | Dim. Reduction | Great visualizations | Slow, non-deterministic |
| UMAP | Dim. Reduction | Fast, global+local | Newer, less studied |
| Isolation Forest | Anomaly | Fast, scalable | Tuning contamination % |
| One-Class SVM | Anomaly | Works in high-D | Kernel choice matters |
| LOF | Anomaly | Local density aware | Slow for large N |
