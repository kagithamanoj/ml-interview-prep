# Deep Learning — Interview Questions & Answers

## Neural Network Fundamentals

### Q1: Explain backpropagation in simple terms.
**Answer**: Backpropagation computes how much each weight contributed to the error, so we can update it.

1. **Forward pass**: Input flows through layers to produce output
2. **Loss computation**: Compare output to ground truth
3. **Backward pass**: Chain rule propagates gradients from loss back through each layer
4. **Weight update**: `w = w - lr × ∂loss/∂w`

**Key insight**: Chain rule allows computing gradients layer-by-layer without recalculating from scratch:
```
∂loss/∂w₁ = ∂loss/∂output × ∂output/∂hidden × ∂hidden/∂w₁
```

### Q2: What is the vanishing gradient problem? How is it solved?
**Answer**: In deep networks, gradients shrink exponentially as they propagate backward through many layers. With sigmoid/tanh activations (gradients in [0, 0.25] or [0, 1]), multiplying many small numbers → near-zero gradients → early layers don't learn.

**Solutions**:
| Solution | How It Helps |
|----------|-------------|
| **ReLU activation** | Gradient = 1 for positive inputs (no shrinking) |
| **Residual connections** | Skip connections add identity path for gradients |
| **Batch normalization** | Normalizes layer inputs, stabilizes gradient flow |
| **LSTM/GRU cells** | Gating mechanisms preserve gradients over sequences |
| **Careful initialization** | Xavier/He init keeps variance stable across layers |
| **Gradient clipping** | Caps gradient magnitude (helps with exploding too) |

### Q3: Explain the Transformer architecture.
**Answer**: The Transformer replaces recurrence with **self-attention**, processing all tokens in parallel.

**Core components**:
```
Input → [Token Embedding + Positional Encoding]
         ↓
    [Multi-Head Self-Attention] ← Q, K, V projections
         ↓
    [Layer Normalization + Residual]
         ↓
    [Feed-Forward Network (2 linear layers + GELU)]
         ↓
    [Layer Normalization + Residual]
         ↓
    × N layers
         ↓
    Output
```

**Self-Attention**: `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`
- **Q** (Query): What am I looking for?
- **K** (Key): What do I contain?
- **V** (Value): What information do I provide?
- **√dₖ**: Scaling factor to prevent softmax saturation

**Multi-Head**: Run attention h times in parallel with different learned projections → captures different relationship types.

### Q4: What is the difference between GPT and BERT?
**Answer**:
| | GPT | BERT |
|--|-----|------|
| **Architecture** | Decoder-only (causal attention) | Encoder-only (bidirectional attention) |
| **Training** | Predict next token (autoregressive) | Mask & predict random tokens (MLM) |
| **Context** | Sees only left context | Sees full bidirectional context |
| **Best for** | Text generation, chat, coding | Classification, NER, Q&A extraction |
| **Inference** | Sequential token generation | Process full input at once |

### Q5: What are LoRA and QLoRA? Why do they matter?
**Answer**: **LoRA** (Low-Rank Adaptation) makes fine-tuning practical:
- Instead of updating all W parameters, decompose update as `ΔW = AB` where A∈ℝᵈˣʳ, B∈ℝʳˣᵈ
- **r << d** (rank 4-64 vs thousands), so trainable params drop by ~99%
- Original weights frozen → no catastrophic forgetting

**QLoRA** adds quantization:
- Base model weights quantized to 4-bit (NF4 format)
- LoRA adapters remain in 16-bit for training
- Can fine-tune 65B parameter models on a single 48GB GPU

## CNNs

### Q6: How do convolutions work in neural networks?
**Answer**: A convolution slides a small filter (3×3, 5×5) across the input, computing dot products to create a feature map.

**Key hyperparameters**:
| Parameter | Effect |
|-----------|--------|
| **Kernel size** | Receptive field size (3×3 most common) |
| **Stride** | Step size (2 = downsample by 2) |
| **Padding** | Border handling (same/valid) |
| **Channels** | Number of filters = number of features learned |

**Why CNNs work**:
- **Translation equivariance**: Detect features regardless of position
- **Parameter sharing**: Same filter applied everywhere → far fewer parameters
- **Hierarchical features**: Early layers = edges, mid = textures, deep = objects
