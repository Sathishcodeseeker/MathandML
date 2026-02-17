1. Linear Algebra (Most Critical)

What to learn:
├── Vector representation         → Data points are vectors
├── Vector addition/subtraction   → Feature operations
├── Dot product                   → Similarity, projections
├── Magnitude (norm)              → Distance calculations
├── Unit vectors                  → Normalization
└── Vector spaces                 → Understanding feature spaces


Concept                          Algorithm
Dot product                      Logistic Regression, SVM, Word2Vec
Norms (L1, L2)                   Regularization, K-Means, DBSCAN
Cosine similarity                NLP, Transformers, Embeddings
Vector spaces                    PCA, t-SNE, UMAP

Matrices:

What to learn:
├── Matrix representation          → Datasets are matrices
├── Matrix multiplication          → Layer computations in neural nets
├── Transpose                      → Data reshaping
├── Inverse                        → Solving linear systems
├── Determinant                    → Matrix properties
├── Identity matrix                → Baseline transformations
├── Eigenvalues & Eigenvectors     → PCA, spectral methods
├── Matrix decomposition (SVD)     → PCA, recommendations
├── Rank                           → Feature independence
└── Positive definite matrices     → Covariance, GMM

Concept                                Algorithm
Matrix multiplication                  Neural Networks, CNNs, Transformers
Eigendecomposition                     PCA
SVD                                    PCA, Dimensionality Reduction
Covariance matrix                      GMM, PCA
Matrix inverse                         Linear Regression (Normal Equation)

LINEAR REGRESSION:     Matrices, Inverse, Transpose
LOGISTIC REGRESSION:   Vectors, Dot product, Matrix mult
RANDOM FOREST:         Minimal linear algebra
GRADIENT BOOSTING:     Minimal linear algebra
SVM:                   Dot product, Kernel trick, Norms
PCA:                   Eigenvalues, Eigenvectors, SVD, Covariance
CNNs:                  Matrix/Tensor operations, Convolutions
TRANSFORMERS:          Matrix multiplication (Attention = QKᵀ)
K-MEANS:              Norms, Distance metrics
GMM:                   Covariance matrices, Eigendecomposition
t-SNE / UMAP:         Distance matrices, Norms

2. Calculus
   
Why: How models LEARN (optimization)

2.1 Differential Calculus (Derivatives)

What to learn:
├── Derivatives                    → Rate of change
│   ├── Power rule                 → d/dx(xⁿ) = nxⁿ⁻¹
│   ├── Chain rule                 → d/dx[f(g(x))] = f'(g(x))·g'(x)
│   ├── Product rule               → d/dx[f·g] = f'g + fg'
│   └── Quotient rule              → Less common in ML
│
├── Partial derivatives            → Derivatives with multiple variables
│   └── ∂f/∂x₁, ∂f/∂x₂, ...     → How loss changes per parameter
│
├── Gradient                       → Vector of all partial derivatives
│   └── ∇f = [∂f/∂x₁, ∂f/∂x₂]  → Direction of steepest ascent
│
├── Jacobian matrix                → Matrix of first-order partials
│   └── Used in backpropagation
│
└── Hessian matrix                 → Matrix of second-order partials
    └── Used in optimization (Newton's method)

Example — Gradient Descent:
Example
Loss function: L(w) = (1/n) Σ(yᵢ - (wxᵢ + b))²

Gradient:
∂L/∂w = -(2/n) Σ xᵢ(yᵢ - (wxᵢ + b))
∂L/∂b = -(2/n) Σ (yᵢ - (wxᵢ + b))

Update rule:
w_new = w_old - learning_rate × ∂L/∂w
b_new = b_old - learning_rate × ∂L/∂b

# This is how EVERY neural network learns!

2.2 Chain Rule (CRITICAL for Deep Learning)
Example
Backpropagation = Repeated application of chain rule

Neural Network:
Input → Layer1 → Layer2 → Layer3 → Output → Loss

∂Loss/∂w₁ = ∂Loss/∂out × ∂out/∂L3 × ∂L3/∂L2 × ∂L2/∂L1 × ∂L1/∂w₁
              ─────────────────────────────────────────────────────────
                            Chain rule applied repeatedly

Concept                             Algorithm
Derivatives                         ALL learning algorithms
Chain rule                          Neural Networks, CNNs, Transformers
Gradient                            Gradient Descent, Backpropagation
Partial derivatives                 Multi-parameter optimization


3. Probability & Statistics
Why: Understanding data, uncertainty, and model evaluation
3.1 Probability Basics

What to learn:
├── Probability rules
│   ├── P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
│   ├── P(A ∩ B) = P(A) × P(B|A)
│   └── Conditional probability P(A|B)
│
├── Bayes' Theorem                    ← VERY IMPORTANT
│   └── P(A|B) = P(B|A) × P(A) / P(B)
│
├── Independence
│   └── P(A ∩ B) = P(A) × P(B)
│
└── Random variables
    ├── Discrete (classification)
    └── Continuous (regression)

3.2 Distributions

What to learn:
├── Normal (Gaussian) distribution    → Most common assumption
│   └── f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/2σ²)
│
├── Bernoulli distribution            → Binary classification
├── Binomial distribution             → Multiple binary trials
├── Poisson distribution              → Count data
├── Uniform distribution              → Random initialization
├── Multinomial distribution          → Multi-class classification
│
└── Multivariate Gaussian             → GMM, PCA
    └── f(x) = ... involves covariance matrix Σ


3.3 Statistics

What to learn:
├── Descriptive Statistics
│   ├── Mean, Median, Mode
│   ├── Variance, Standard Deviation
│   ├── Covariance & Correlation
│   └── Percentiles, IQR
│
├── Inferential Statistics
│   ├── Hypothesis testing           → A/B testing
│   ├── p-values                     → Feature significance
│   ├── Confidence intervals         → Model uncertainty
│   └── t-test, chi-square test      → Feature selection
│
├── Maximum Likelihood Estimation (MLE)  ← CRITICAL
│   └── Find parameters that maximize P(data|params)
│   └── Used in: Logistic Regression, GMM, Neural Networks
│
├── Maximum A Posteriori (MAP)
│   └── MLE + prior (Bayesian approach)
│   └── Used in: Regularization
│
└── Information Theory
    ├── Entropy: H = -Σ p(x) log p(x)       → Decision Trees
    ├── Cross-entropy                         → Loss function for classification
    ├── KL Divergence                         → Autoencoders (VAE)
    └── Information Gain                      → Random Forests, Gradient Boosting


Example — Cross-Entropy Loss:

Binary Cross-Entropy (Logistic Regression, Neural Nets):

L = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Where:
  yᵢ  = actual label (0 or 1)
  ŷᵢ  = predicted probability

4. Optimization
Why: How models find the best parameters

What to learn:
├── Gradient Descent                      ← MOST IMPORTANT
│   ├── Batch Gradient Descent
│   ├── Stochastic Gradient Descent (SGD)
│   ├── Mini-batch Gradient Descent
│   └── Learning rate concept
│
├── Advanced Optimizers
│   ├── Adam                              → Default for deep learning
│   ├── RMSprop
│   ├── AdaGrad
│   └── Momentum
│
├── Convex vs Non-convex optimization
│   ├── Linear/Logistic Regression → Convex (global minimum)
│   └── Neural Networks → Non-convex (local minima)
│
├── Regularization (Preventing overfitting)
│   ├── L1 (Lasso): adds |w| penalty     → Sparse features
│   ├── L2 (Ridge): adds w² penalty      → Small weights
│   └── Elastic Net: L1 + L2 combined
│
├── Constrained optimization
│   ├── Lagrange multipliers              → SVM
│   └── KKT conditions                    → SVM
│
└── Loss Functions
    ├── MSE (Mean Squared Error)          → Regression
    ├── Cross-Entropy                     → Classification
    ├── Hinge Loss                        → SVM
    └── Huber Loss                        → Robust regression

Attention Mechanism Math (For Transformers

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

Where:
  Q = Query matrix    (What am I looking for?)
  K = Key matrix      (What do I contain?)
  V = Value matrix    (What do I output?)
  dₖ = dimension      (Scaling factor)

This is ALL matrix multiplication + softmax!
