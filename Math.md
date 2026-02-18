# Mathematics for Machine Learning

---

# 1. Linear Algebra (Most Critical)

## 1.1 Vectors

### What to Learn
- Vector representation → Data points are vectors  
- Vector addition/subtraction → Feature operations  
- Dot product → Similarity, projections  
- Magnitude (norm) → Distance calculations  
- Unit vectors → Normalization  
- Vector spaces → Understanding feature spaces  

### Concepts → Algorithms

| Concept | Used In |
|---------|---------|
| Dot product | Logistic Regression, SVM, Word2Vec |
| Norms (L1, L2) | Regularization, K-Means, DBSCAN |
| Cosine similarity | NLP, Transformers, Embeddings |
| Vector spaces | PCA, t-SNE, UMAP |

---

## 1.2 Matrices

### What to Learn
- Matrix representation → Datasets are matrices  
- Matrix multiplication → Neural network layer computation  
- Transpose → Data reshaping  
- Inverse → Solving linear systems  
- Determinant → Matrix properties  
- Identity matrix → Baseline transformation  
- Eigenvalues & Eigenvectors → PCA, spectral methods  
- Matrix decomposition (SVD) → PCA, recommender systems  
- Rank → Feature independence  
- Positive definite matrices → Covariance, GMM  

### Concepts → Algorithms

| Concept | Used In |
|---------|---------|
| Matrix multiplication | Neural Networks, CNNs, Transformers |
| Eigendecomposition | PCA |
| SVD | Dimensionality reduction |
| Covariance matrix | GMM, PCA |
| Matrix inverse | Linear Regression (Normal Equation) |

---

## 1.3 Algorithms vs Linear Algebra

- **Linear Regression** → Matrices, inverse, transpose  
- **Logistic Regression** → Vectors, dot product, matrix multiplication  
- **Random Forest / Gradient Boosting** → Minimal linear algebra  
- **SVM** → Dot product, kernels, norms  
- **PCA** → Eigenvalues, eigenvectors, SVD, covariance  
- **CNNs / Transformers** → Matrix & tensor operations  
- **K-Means** → Norms, distance metrics  
- **GMM** → Covariance matrices, eigendecomposition  
- **t-SNE / UMAP** → Distance matrices  

---

# 2. Calculus (How Models Learn)

## 2.1 Differential Calculus

### What to Learn
- Derivatives → Rate of change  
  - Power rule: `d/dx (xⁿ) = n xⁿ⁻¹`  
  - Chain rule  
  - Product rule  
- Partial derivatives → Multi-variable change  
- Gradient → Direction of steepest ascent  
- Jacobian → First-order partial matrix (backpropagation)  
- Hessian → Second-order partial matrix (Newton methods)  

### Gradient Descent Example

**Loss function**

```
L(w) = (1/n) Σ (yᵢ − (w xᵢ + b))²
```

**Gradients**

```
∂L/∂w = -(2/n) Σ xᵢ (yᵢ − (w xᵢ + b))
∂L/∂b = -(2/n) Σ (yᵢ − (w xᵢ + b))
```

**Update rule**

```
w_new = w_old − η ∂L/∂w
b_new = b_old − η ∂L/∂b
```

---

## 2.2 Chain Rule (Critical for Deep Learning)

Backpropagation is repeated application of the chain rule:

```
∂Loss/∂w₁ =
∂Loss/∂out × ∂out/∂L3 × ∂L3/∂L2 × ∂L2/∂L1 × ∂L1/∂w₁
```

### Concepts → Algorithms

| Concept | Used In |
|---------|---------|
| Derivatives | All learning algorithms |
| Chain rule | Neural Networks, CNNs, Transformers |
| Gradient | Gradient descent, backpropagation |
| Partial derivatives | Multi-parameter optimization |

---

# 3. Probability & Statistics

## 3.1 Probability Basics

### What to Learn
- Probability rules  
- Conditional probability  
- Independence  
- Bayes’ theorem  
- Random variables (discrete & continuous)  

---

## 3.2 Distributions

- Normal (Gaussian)  
- Bernoulli  
- Binomial  
- Poisson  
- Uniform  
- Multinomial  
- Multivariate Gaussian  

---

## 3.3 Statistics

### Descriptive
- Mean, median, mode  
- Variance, standard deviation  
- Covariance, correlation  
- Percentiles, IQR  

### Inferential
- Hypothesis testing  
- p-values  
- Confidence intervals  
- t-test, chi-square  

### Estimation & Information Theory
- **MLE** → Logistic Regression, GMM, Neural Nets  
- **MAP** → Bayesian regularization  
- **Entropy / Cross-Entropy / KL Divergence / Information Gain**

### Binary Cross-Entropy

```
L = -(1/n) Σ [ yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ) ]
```

---

# 4. Optimization

## Core Concepts
- Gradient Descent (Batch, SGD, Mini-batch)  
- Learning rate  
- Adam, RMSProp, AdaGrad, Momentum  
- Convex vs Non-convex optimization  
- Regularization (L1, L2, Elastic Net)  
- Constrained optimization (Lagrange, KKT)  
- Loss functions (MSE, Cross-Entropy, Hinge, Huber)  

---

# 5. Attention Mechanism (Transformers)

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Where:

- **Q** → Query matrix  
- **K** → Key matrix  
- **V** → Value matrix  
- **dₖ** → Key dimension (scaling factor)  

➡ Entire transformer math = **matrix multiplication + softmax**

---
