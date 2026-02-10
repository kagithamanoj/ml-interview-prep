# Linear Algebra for ML — Interview Questions & Answers

## Vectors & Matrices

### Q1: What is a dot product and why is it important in ML?
**Answer**: The dot product of two vectors **a** and **b** is `a · b = Σ(aᵢ × bᵢ)`. It measures similarity between vectors — when normalized, it gives cosine similarity. In ML:
- **Neural networks**: Every layer computes `Wx + b` (matrix-vector dot products)
- **Embeddings**: Similarity search uses dot products / cosine similarity
- **SVMs**: Kernel functions are generalized dot products
- **Attention**: Scaled dot-product attention in transformers: `Attention(Q,K,V) = softmax(QKᵀ/√d)V`

### Q2: What is an eigenvalue/eigenvector? Where is it used in ML?
**Answer**: For matrix A, if `Av = λv`, then **v** is an eigenvector and **λ** is its eigenvalue. The eigenvector doesn't change direction under transformation A — only scales by λ.

**ML applications**:
- **PCA**: Eigenvectors of the covariance matrix = principal components
- **Spectral clustering**: Uses eigenvectors of the graph Laplacian
- **Google PageRank**: Dominant eigenvector of the link matrix
- **Stability analysis**: Eigenvalues of the Hessian determine optimization landscape

### Q3: What is the difference between L1 and L2 norms?
**Answer**:
| Norm | Formula | Effect in ML |
|------|---------|-------------|
| **L1 (Manhattan)** | `Σ|xᵢ|` | Promotes sparsity (many weights → 0), feature selection |
| **L2 (Euclidean)** | `√Σxᵢ²` | Promotes small weights (shrinkage), prevents overfitting |

**Regularization**: L1 → Lasso regression, L2 → Ridge regression, L1+L2 → Elastic Net

### Q4: What is a positive definite matrix? Why does it matter?
**Answer**: Matrix M is positive definite if `xᵀMx > 0` for all non-zero x. 

**Why it matters**:
- Covariance matrices must be positive semi-definite
- Guarantees a unique minimum in optimization (convexity)
- Kernel matrices in SVMs must be positive semi-definite (Mercer's theorem)

## Matrix Decompositions

### Q5: Explain SVD and its use in ML.
**Answer**: Singular Value Decomposition factors any m×n matrix as `A = UΣVᵀ`:
- **U** (m×m): Left singular vectors (orthonormal)
- **Σ** (m×n): Diagonal matrix of singular values (sorted descending)
- **Vᵀ** (n×n): Right singular vectors (orthonormal)

**ML uses**:
- **Dimensionality reduction**: Keep top-k singular values (truncated SVD)
- **Recommendation systems**: Matrix factorization for collaborative filtering
- **NLP**: Latent Semantic Analysis (LSA) applies SVD to term-document matrices
- **Image compression**: Approximate images with fewer singular values
- **Pseudoinverse**: Used in least-squares solutions

### Q6: What is the relationship between PCA and SVD?
**Answer**: PCA can be computed via SVD. Given centered data matrix X:
1. Covariance matrix: `C = XᵀX / (n-1)`
2. SVD of X: `X = UΣVᵀ`
3. Principal components = columns of V
4. Eigenvalues of C = `σ²/(n-1)` where σ are singular values

SVD is numerically more stable than eigendecomposition of the covariance matrix.
