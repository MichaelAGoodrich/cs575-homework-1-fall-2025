
### Q: What is the equation if there are no negative edges?

**A:**  
If you compute the **reconstruction loss without using negative samples**, then you're only evaluating how well the model predicts the *positive* edges — i.e., the ones that exist in the graph.

#### 📐 Mathematical Definition (No Negative Sampling)

Given:
- Embeddings: \( z \in \mathbb{R}^{n \times d} \)
- Edge set: \( E \) (from `edge_index`)
- Decoder: \( \hat{A}_{ij} = \sigma(z_i^\top z_j) \)

Then the **loss** is simply:

\[
\mathcal{L}_{\text{recon}} = - \sum_{(i, j) \in E} \log \sigma(z_i^\top z_j)
\]

This is the **positive part** of the full binary cross-entropy loss. You’re only maximizing the likelihood that connected nodes have high similarity (dot product), and not penalizing the model for predicting false positives.

#### 🔍 In words:
- You're encouraging the model to predict high probabilities (close to 1) for **existing** edges.
- But you're not doing anything to discourage it from also predicting high probabilities for **non-edges**, which could lead to overfitting or degenerate embeddings (e.g., all nodes close together in embedding space).

#### 🚨 Why it matters:
- Using **only positive examples** is common in early GAE demos (especially when first illustrating how it works).
- But in practice, **negative sampling** is crucial to learn useful embeddings that generalize.

---

### Q: What is `model.recon_loss` defined mathematically?

**A:**  
In the context of a **Graph Autoencoder (GAE)** in PyTorch Geometric, `model.recon_loss(z, edge_index)` computes the **reconstruction loss** based on how well the model predicts the presence or absence of edges from the embeddings `z`.

#### 🔢 Mathematical Definition of `recon_loss`

Given:
- \( z \in \mathbb{R}^{n \times d} \): matrix of node embeddings
- `edge_index`: observed edges in the graph
- \( \hat{A} = \sigma(z z^\top) \): predicted adjacency probabilities (via inner product decoder)

The **reconstruction loss** is:

\[
\mathcal{L}_{\text{recon}} = - \sum_{(i, j) \in E} \log \sigma(z_i^\top z_j) - \sum_{(i, j) \notin E} \log(1 - \sigma(z_i^\top z_j))
\]

This is the **binary cross-entropy loss** between:
- Positive examples: real edges in the graph
- Negative examples: randomly sampled node pairs with no edge

#### 🧪 In PyG:

Under the hood, PyG's `GAE.recon_loss(z, edge_index)` does something like:

```python
pos_loss = -log(sigmoid(z[i]^T z[j]))  # for (i, j) in edge_index
neg_loss = -log(1 - sigmoid(z[i]^T z[j]))  # for (i, j) in negative samples
total_loss = pos_loss + neg_loss
```

You can optionally control the number of negative samples using:

```python
model.recon_loss(z, pos_edge_index, neg_edge_index)
```
