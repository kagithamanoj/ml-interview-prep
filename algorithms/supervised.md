# Supervised Learning — Interview Questions & Answers

## Linear Models

### Q1: How does logistic regression work? Why not use linear regression for classification?
**Answer**: Logistic regression applies a sigmoid function to a linear combination:
```
P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx+b)))
```

**Why not linear regression?**
- Linear regression outputs unbounded values — not valid probabilities
- MSE loss on 0/1 labels creates non-convex optimization
- Logistic regression uses cross-entropy loss → convex optimization + probabilistic outputs

### Q2: What is the difference between L1 and L2 regularization in practice?
**Answer**:
| | L1 (Lasso) | L2 (Ridge) |
|--|-----------|-----------|
| **Penalty** | `λΣ|wᵢ|` | `λΣwᵢ²` |
| **Effect** | Drives weights to exactly 0 | Shrinks weights toward 0 |
| **Use** | Feature selection (sparse models) | Prevent overfitting (all features kept) |
| **Solution** | Not differentiable at 0 | Closed-form solution exists |
| **When to use** | Many irrelevant features | All features somewhat useful |

### Q3: Explain gradient descent variants.
**Answer**:
| Variant | Batch Size | Pros | Cons |
|---------|-----------|------|------|
| **Batch GD** | Full dataset | Stable convergence | Slow, memory-heavy |
| **SGD** | 1 sample | Fast, can escape local minima | Noisy, unstable |
| **Mini-batch** | 32-256 | Balance of speed & stability | Hyperparameter (batch size) |

**Popular optimizers**:
- **Adam**: Adaptive learning rates + momentum. Default choice for most tasks.
- **AdamW**: Adam with decoupled weight decay. Better generalization.
- **SGD + Momentum**: Often better final performance for CNNs (with careful tuning).

## Tree-Based Methods

### Q4: How does a Random Forest handle overfitting compared to a single Decision Tree?
**Answer**: Decision trees overfit by memorizing training data. Random Forest reduces variance through:
1. **Bagging**: Each tree trained on a bootstrap sample (random subset with replacement)
2. **Feature randomness**: Each split considers only √p random features (classification) or p/3 (regression)
3. **Averaging**: Final prediction = average (regression) or majority vote (classification)

**Result**: Individual trees may overfit, but their **averaged predictions** cancel out the noise → lower variance, similar bias.

### Q5: XGBoost vs. Random Forest — when to use which?
**Answer**:
| | XGBoost | Random Forest |
|--|---------|--------------|
| **Strategy** | Sequential boosting (trees correct previous errors) | Parallel bagging (trees vote independently) |
| **Bias/Variance** | Reduces bias primarily | Reduces variance primarily |
| **Performance** | Usually higher accuracy | More robust, less tuning needed |
| **Overfitting** | Can overfit if not tuned (regularize!) | Naturally resistant |
| **Speed** | Faster training (early stopping) | Slower (all trees fully grown) |
| **Tabular data** | Often SOTA on competitions | Great baseline, harder to overfit |

## SVMs

### Q6: Explain the kernel trick in SVMs.
**Answer**: The kernel trick maps data to a higher-dimensional space **without explicitly computing the transformation**. It works because SVMs only need dot products between data points:

```
K(x, z) = φ(x) · φ(z)
```

**Common kernels**:
| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | `xᵀz` | Linearly separable data |
| **RBF/Gaussian** | `exp(-γ‖x-z‖²)` | Most common default, non-linear |
| **Polynomial** | `(xᵀz + c)^d` | Feature interactions |

**RBF kernel** maps to infinite-dimensional space — can learn any decision boundary with enough data.
