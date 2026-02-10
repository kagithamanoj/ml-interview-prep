# ML Cheat Sheet — Quick Reference

## Model Selection Guide

```
Is your target continuous?
├── Yes → REGRESSION
│   ├── Linear relationship? → Linear/Ridge/Lasso Regression
│   ├── Non-linear? → Random Forest, XGBoost, SVR
│   └── Very complex? → Neural Network
└── No → CLASSIFICATION
    ├── Binary? → Logistic Regression, SVM, XGBoost
    ├── Multi-class? → Softmax, Random Forest, XGBoost
    └── Multi-label? → Binary Relevance, Classifier Chains

No labels available?
├── Find groups? → K-Means, DBSCAN, GMM
├── Reduce dimensions? → PCA, t-SNE, UMAP
└── Detect anomalies? → Isolation Forest, Autoencoder
```

## Metrics Quick Reference

| Task | Metric | When to Use |
|------|--------|-------------|
| **Classification** | Accuracy | Balanced classes |
| | F1 Score | Imbalanced classes |
| | AUC-ROC | Ranking/threshold tuning |
| | Log Loss | Probabilistic predictions |
| **Regression** | MSE/RMSE | Penalize large errors |
| | MAE | Robust to outliers |
| | R² | Explained variance |
| **Ranking** | NDCG | Search/recommendation |
| | MAP | Information retrieval |

## Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Default for hidden layers |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary output |
| Softmax | eˣⁱ/Σeˣ | (0, 1) | Multi-class output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | LSTM gates |
| GELU | x×Φ(x) | (-0.17, ∞) | Transformers |

## Optimizer Cheat Sheet

| Optimizer | Learning Rate | When to Use |
|-----------|--------------|-------------|
| **Adam** | 1e-3 to 1e-4 | Default for most tasks |
| **AdamW** | 1e-3 to 1e-5 | Transformers, better generalization |
| **SGD+Momentum** | 1e-1 to 1e-2 | CNNs (with scheduler) |
| **RMSprop** | 1e-3 | RNNs |

## Common Hyperparameters

| Model | Key Params | Typical Range |
|-------|-----------|---------------|
| **Random Forest** | n_estimators, max_depth | 100-500, 10-30 |
| **XGBoost** | max_depth, learning_rate, n_estimators | 3-8, 0.01-0.3, 100-1000 |
| **SVM** | C, gamma, kernel | 0.1-100, auto/scale, rbf |
| **Neural Net** | lr, batch_size, layers, dropout | 1e-4, 32-256, 2-10, 0.1-0.5 |

## Data Preprocessing Checklist

- [ ] Handle missing values (imputation, removal)
- [ ] Encode categoricals (one-hot, label, target encoding)
- [ ] Scale features (StandardScaler for SVM/NN, not needed for trees)
- [ ] Handle class imbalance (SMOTE, class weights, oversampling)
- [ ] Feature engineering (interactions, polynomial, domain-specific)
- [ ] Train/validation/test split (or k-fold CV)
- [ ] Check for data leakage (fit scaler on train only!)
