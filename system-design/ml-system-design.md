# ML System Design — Interview Questions & Answers

## System Design Framework

### Q1: How do you approach an ML system design interview question?
**Answer**: Follow this framework:

```
1. CLARIFY    → Requirements, constraints, metrics, scope
2. DATA       → Sources, features, labeling, storage
3. MODEL      → Algorithm choice, baseline, training pipeline
4. SERVING    → Inference, latency, throughput, A/B testing
5. MONITORING → Metrics, drift detection, retraining triggers
```

### Q2: Design a recommendation system (e.g., Netflix, YouTube).
**Answer**:

**Requirements**: Recommend items to users based on history, preferences, and context.

**Two-Stage Architecture**:
```
Stage 1: Candidate Generation (fast, broad)
    → Collaborative filtering (user-item matrix factorization)
    → Content-based filtering (item features similarity)
    → Popular/trending items (cold start fallback)
    → Output: ~1000 candidates

Stage 2: Ranking (slow, precise)
    → Deep neural network with features:
      - User features (age, history, preferences)
      - Item features (genre, popularity, recency)
      - Context features (time, device, location)
      - Cross features (user-item interactions)
    → Output: Top 20-50 ranked items
```

**Key considerations**:
- **Cold start**: New users → popularity-based; new items → content-based
- **Exploration vs. exploitation**: Epsilon-greedy or Thompson sampling
- **Real-time signals**: Recently watched, session context
- **Diversity**: Avoid filter bubbles with MMR (Maximal Marginal Relevance)
- **Metrics**: CTR, watch time, user satisfaction surveys

### Q3: Design a fraud detection system.
**Answer**:

**Architecture**:
```
Transaction → [Feature Engineering] → [Real-time Scoring] → Accept/Reject/Review
                                            ↓
                                    [Rule Engine] (hard rules)
                                            +
                                    [ML Model] (learned patterns)
```

**Features**:
- Transaction amount, time, location, merchant category
- Velocity features: # transactions in last 1h, 24h, 7d
- Deviation from user patterns (unusual time, amount, location)
- Device fingerprint, IP geolocation
- Graph features (connected fraud networks)

**Model choices**:
- **XGBoost**: Tabular data baseline, handles imbalanced data well
- **Isolation Forest**: Anomaly detection (unsupervised)
- **GNN**: Graph-based fraud ring detection
- **Ensemble**: Combine rule-based + ML for production

**Challenges**:
| Challenge | Solution |
|-----------|----------|
| **Class imbalance** (99.9% legitimate) | SMOTE, cost-sensitive learning, focal loss |
| **Concept drift** | Continuous retraining, monitoring feature distributions |
| **Low latency** (<100ms) | Precompute features, model distillation |
| **Interpretability** | SHAP values for each decision, audit trail |

### Q4: Design an LLM-based RAG system for enterprise.
**Answer**:

```
User Query → [Query Understanding]
                    ↓
             [Retrieval Pipeline]
             ├── Dense retrieval (embeddings + vector DB)
             ├── Sparse retrieval (BM25)
             └── Hybrid (RRF fusion)
                    ↓
             [Reranker] (cross-encoder)
                    ↓
             [Context Assembly] (top-k chunks + metadata)
                    ↓
             [LLM Generation] (with grounding prompt)
                    ↓
             [Response + Citations]
```

**Key decisions**:
| Decision | Options |
|----------|---------|
| **Chunking** | Fixed size (512 tokens), semantic (paragraph), recursive |
| **Embedding model** | OpenAI ada-002, BGE, Cohere, E5 |
| **Vector DB** | Pinecone, Weaviate, Qdrant, FAISS, Chroma |
| **LLM** | GPT-4o, Claude, Llama 3, Mistral |
| **Reranker** | Cohere rerank, ms-marco cross-encoder |

**Evaluation**: Faithfulness, relevance, completeness (using RAGAS or Giskard RAGET).
