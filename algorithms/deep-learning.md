# Deep Learning — Interview Reference

## Neural Network Fundamentals

### Activation Functions
| Function | Formula | Range | When to Use |
|----------|---------|-------|-------------|
| ReLU | max(0, x) | [0, ∞) | Default for hidden layers |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | When dying ReLU is a problem |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary classification output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | RNNs, normalized output |
| Softmax | eˣⁱ/Σeˣʲ | (0, 1), sum=1 | Multi-class output |
| GELU | x·Φ(x) | ~ (-0.17, ∞) | Transformers (BERT, GPT) |

### Loss Functions
| Function | Task | Formula |
|----------|------|---------|
| MSE | Regression | (1/n)Σ(y-ŷ)² |
| MAE | Regression (robust) | (1/n)Σ\|y-ŷ\| |
| BCE | Binary Classification | -[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| Cross-Entropy | Multi-class | -Σ yᵢ·log(ŷᵢ) |
| Focal Loss | Imbalanced | -αₜ(1-pₜ)ᵧ·log(pₜ) |

### Optimizers
| Optimizer | Key Idea | When to Use |
|-----------|----------|-------------|
| SGD | Vanilla gradient descent | Simple, with momentum |
| SGD + Momentum | Accelerated gradient | When stuck in local minima |
| Adam | Adaptive learning rate | Default choice |
| AdamW | Adam + weight decay fix | Transformers, large models |
| LAMB | Layer-wise adaptive rates | Very large batch training |

---

## Architectures

### CNNs (Convolutional Neural Networks)
```
Key concepts:
- Convolution: sliding filter extracts local features
- Pooling: reduces spatial dimensions (max/avg)
- Feature hierarchy: edges → textures → objects

Architecture pattern:
Conv → BatchNorm → ReLU → Pool → ... → Flatten → Dense → Softmax
```

**Interview Q:** *"Why convolutions instead of fully-connected?"*
→ Parameter sharing, translation invariance, spatial hierarchy

### RNNs / LSTMs / GRUs
```
RNN problem: vanishing/exploding gradients for long sequences

LSTM gates:
- Forget gate: what to discard from cell state
- Input gate: what new info to store
- Output gate: what to output

GRU (simplified LSTM):
- Update gate: combines forget + input gates
- Reset gate: how much past to forget
```

**Interview Q:** *"LSTM vs GRU?"*
→ GRU: fewer parameters, faster. LSTM: more expressive for long sequences.

### Transformers
```
Key innovation: Self-attention (parallelizable, long-range dependencies)

Architecture:
- Encoder: Bidirectional attention (BERT)
- Decoder: Causal attention (GPT)
- Encoder-Decoder: Both (T5, BART)

Key formulas:
- Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
- Multi-head: concat(head₁,...,headₕ)Wᴼ
```

**Interview Q:** *"Why scale by √dₖ?"*
→ Prevents softmax saturation when dot products are large

---

## Regularization Techniques

| Technique | How | When |
|-----------|-----|------|
| Dropout | Randomly zero out neurons (p=0.1-0.5) | Overfitting, after dense layers |
| Batch Normalization | Normalize activations per batch | Faster training, higher LR |
| Layer Normalization | Normalize per sample | Transformers, RNNs |
| Weight Decay (L2) | Penalize large weights | Always (small λ) |
| Early Stopping | Stop when val loss increases | Training any model |
| Data Augmentation | Transform training data | Limited data |
| Label Smoothing | Soften hard labels (0.1) | Classification, transformers |

---

## Training Tricks

### Learning Rate Schedules
```
- Warmup: Start low, increase linearly (first ~5% of training)
- Cosine decay: High → Low following cosine curve
- Step decay: Reduce by factor every N epochs
- OneCycleLR: Ramp up → ramp down (fast training)
```

### Batch Size Guidelines
```
- Small (8-32): Better generalization, noisy gradients
- Large (256-8192): Better GPU utilization, needs LR scaling
- Rule: Scale LR linearly with batch size (LR × BS/base_BS)
```

### Mixed Precision Training
```
- Use FP16 for forward/backward pass (faster, less memory)
- Keep FP32 master weights for stability
- 2x speedup with minimal accuracy loss
- Enable: torch.cuda.amp.autocast()
```

---

## Common Interview Questions

1. **"Explain backpropagation"** → Chain rule applied layer by layer. Compute loss gradient, propagate backward, update weights via SGD/Adam.

2. **"Vanishing gradient problem"** → Sigmoid squashes gradients < 1, compounding over layers. Fix: ReLU, skip connections, LSTM gates, batch norm.

3. **"Bias-variance tradeoff"** → High bias = underfitting (model too simple). High variance = overfitting (model too complex). Balance with regularization + model complexity.

4. **"Why is ResNet effective?"** → Skip connections allow gradient flow through identity paths. Enables training 100+ layers without degradation.

5. **"Explain attention mechanism"** → Each token computes relevance scores with all others via Q·K similarity, then takes weighted sum of V. Allows capturing long-range dependencies.

6. **"Transfer learning strategy?"** → Pretrain on large dataset (ImageNet/Wikipedia), freeze base layers, fine-tune top layers on your task. Progressive unfreezing for best results.
