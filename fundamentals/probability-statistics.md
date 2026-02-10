# Probability & Statistics for ML — Interview Questions & Answers

## Probability Fundamentals

### Q1: Explain Bayes' Theorem and its use in ML.
**Answer**: `P(A|B) = P(B|A) × P(A) / P(B)`

- **P(A|B)**: Posterior — probability of A given we observed B
- **P(B|A)**: Likelihood — probability of B given A is true
- **P(A)**: Prior — initial belief about A
- **P(B)**: Evidence — normalizing constant

**ML uses**:
- **Naive Bayes classifier**: Spam detection, text classification
- **Bayesian optimization**: Hyperparameter tuning (prior over function space)
- **Bayesian neural networks**: Uncertainty estimation over weights
- **A/B testing**: Bayesian approach gives probability of improvement, not just p-values

### Q2: What is the difference between MLE and MAP estimation?
**Answer**:
| Method | Formula | Intuition |
|--------|---------|-----------|
| **MLE** | `argmax P(data\|θ)` | Find θ that makes the data most probable |
| **MAP** | `argmax P(θ\|data) = argmax P(data\|θ)P(θ)` | Like MLE but includes a prior on θ |

**Connection to regularization**:
- MLE = unregularized training
- MAP with Gaussian prior on θ = L2 regularization
- MAP with Laplace prior on θ = L1 regularization

### Q3: Explain the bias-variance tradeoff.
**Answer**:
```
Total Error = Bias² + Variance + Irreducible Noise
```
- **Bias**: Error from wrong assumptions (underfitting) — model is too simple
- **Variance**: Error from sensitivity to training data (overfitting) — model is too complex
- **Tradeoff**: Decreasing one typically increases the other

| Model Complexity | Bias | Variance | Result |
|-----------------|------|----------|--------|
| Low (linear)    | High | Low      | Underfitting |
| Medium          | Med  | Med      | Good fit |
| High (deep net) | Low  | High     | Overfitting |

**Solutions**: Cross-validation, regularization, ensemble methods, more training data.

### Q4: What are Type I and Type II errors?
**Answer**:
| Error | Name | Description | Example (spam filter) |
|-------|------|-------------|----------------------|
| **Type I** | False Positive | Reject true null hypothesis | Legitimate email → spam |
| **Type II** | False Negative | Fail to reject false null | Spam email → inbox |

- **Precision** penalizes Type I errors: `TP / (TP + FP)`
- **Recall** penalizes Type II errors: `TP / (TP + FN)`
- **F1 Score** balances both: `2 × Precision × Recall / (Precision + Recall)`

## Distributions

### Q5: When do you use Gaussian vs. Bernoulli vs. Poisson distributions?
**Answer**:
| Distribution | Use Case | Parameters | Example |
|-------------|----------|------------|---------|
| **Gaussian** | Continuous data, errors, noise | μ (mean), σ² (variance) | Heights, test scores |
| **Bernoulli** | Binary outcomes | p (probability of success) | Coin flip, click/no-click |
| **Binomial** | Count of successes in n trials | n, p | Emails clicked out of 100 |
| **Poisson** | Count of rare events in fixed interval | λ (rate) | Server requests per minute |
| **Exponential** | Time between Poisson events | λ | Time between failures |

### Q6: What is the Central Limit Theorem and why does it matter?
**Answer**: The CLT states that the **mean of a large number of i.i.d. random variables** approaches a normal distribution, regardless of the underlying distribution.

**Why it matters for ML**:
- Justifies using Gaussian assumptions in many models
- Enables confidence intervals and hypothesis testing
- Explains why batch means (mini-batch gradient descent) are more stable
- Foundation for bootstrap methods
