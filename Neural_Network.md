# Deep Learning — From Scratch to Regularisation
> Everything we covered, built step by step.

---

## Table of Contents
1. [Forward Pass](#1-forward-pass)
2. [Loss Function](#2-loss-function)
3. [Why Activation Functions?](#3-why-activation-functions)
4. [ReLU](#4-relu)
5. [Sigmoid](#5-sigmoid)
6. [Softmax](#6-softmax)
7. [Cross Entropy Loss](#7-cross-entropy-loss)
8. [Backpropagation](#8-backpropagation)
9. [Chain Rule — The Engine of Backprop](#9-chain-rule--the-engine-of-backprop)
10. [Adam Optimizer](#10-adam-optimizer)
11. [Overfitting](#11-overfitting)
12. [Dropout](#12-dropout)
13. [L1 and L2 Regularisation](#13-l1-and-l2-regularisation)
14. [Hyperparameters](#14-hyperparameters)
15. [Batch Normalisation](#15-batch-normalisation)
16. [Evaluation Metrics for Multi-Class Classification](#16-evaluation-metrics-for-multi-class-classification)
17. [SHAP Explainability for Neural Networks](#17-shap-explainability-for-neural-networks)
18. [Full TensorFlow Code](#18-full-tensorflow-code)

---

## 1. Forward Pass

Data flows **left to right** through the network. Each weight transforms the value.

```
Input → [w₁] → [w₂] → Prediction
```

### Example

```
Input = 3,  w₁ = 5,  w₂ = 2

Step 1: 3 × 5 = 15        ← input hits w₁
Step 2: 15 × 2 = 30       ← output hits w₂

Prediction = 30
```

### Full Picture

```
Input        w₁           w₂          Prediction
  3    →   [× 5]  →  15  →  [× 2]  →     30
```

> **Rule:** Each layer = multiply by weight. Data flows left to right. Simple.

---

## 2. Loss Function

Loss measures **how wrong the prediction was.**

```
Loss = (Prediction - Actual)²
```

### Example

```
Prediction = 30,  Actual = 25

Loss = (30 - 25)² = 5² = 25
```

The higher the loss → the more wrong the network is.
The goal of training = **drive loss toward 0.**

---

## 3. Why Activation Functions?

Without activation functions, stacking layers is **useless.**

### The Problem

Every layer does `y = w × x` (a straight line).

```
Layer 1: y = 3x
Layer 2: y = 2x
Layer 3: y = 4x

Combined: y = 3 × 2 × 4 × x = 24x  ← still just one straight line!
```

**10 layers = mathematically identical to 1 layer.** No matter how deep you go.

### Why That's Bad

Real problems (recognising cats, detecting fraud, translating languages) are **non-linear.** A straight line can never capture them.

### The Fix

Add an **activation function** after each layer to introduce a bend:

```
Input → [w₁] → 🔀 → [w₂] → 🔀 → Output
                ↑              ↑
           activation     activation
```

Now stacking layers becomes genuinely powerful.

---

## 4. ReLU

**Most popular activation function.** Dead simple.

```
ReLU(x) = x   if x > 0   ← keep it
ReLU(x) = 0   if x ≤ 0   ← kill it
```

### Examples

```
ReLU(8)   = 8    ✅ positive → keep
ReLU(-3)  = 0    ❌ negative → kill
ReLU(0)   = 0    ❌ zero → kill
```

### Visual

```
Output
  |        /
  |       /
  |      /
  |     /
──|────/──────── Input
  |   0
 (flat here)
```

That **bend at zero** is what makes deep networks powerful.

### ⚠️ Dead Neuron Problem

If ReLU kills a neuron's output (outputs 0), everything **downstream also dies:**

```
Layer 1: ReLU(-6) = 0
Layer 2: 0 × w₂  = 0   ← no matter what w₂ is
Layer 3: 0 × w₃  = 0   ← completely dead
```

This is called a **dead neuron** — a real problem in deep networks.

---

## 5. Sigmoid

Used for **yes/no (binary) problems.**

Squishes ANY number into 0 to 1:

```
Sigmoid(x) = 1 / (1 + e^(-x))
```

### What It Does

```
Input        Sigmoid Output
──────────────────────────
-999    →    ~0.00   (almost certainly NO)
-2      →     0.12   (probably no)
 0      →     0.50   (totally unsure)
 2      →     0.88   (probably yes)
 999    →    ~1.00   (almost certainly YES)
```

### Visual

```
1.0  |         _________
     |        /
0.5  |───────/──────────
     |      /
0.0  |_____/
          0
```

### Example

```
Network output = 2.5
Sigmoid(2.5)   = 0.92
→ "92% chance this is spam" ✅
```

---

## 6. Softmax

Used for **multi-class problems** (cat vs dog vs bird).

Converts raw scores into probabilities that **sum to 100%.**

```
Softmax in two steps:
  Step 1: Exponentiate everything  →  makes all values positive
  Step 2: Divide each by total     →  normalises to sum = 1
```

### Example

```
Raw scores (logits):
  Cat  = 3.0
  Dog  = 1.0
  Bird = 0.2

Step 1 — Exponentiate:
  e^3.0 = 20.1
  e^1.0 =  2.7
  e^0.2 =  1.2
  Total = 24.0

Step 2 — Divide by total:
  Cat:  20.1 / 24.0 = 0.84 → 84% 🐱
  Dog:   2.7 / 24.0 = 0.11 → 11% 🐶
  Bird:  1.2 / 24.0 = 0.05 →  5% 🐦
  ─────────────────────────
  Total:              100% ✅
```

### Sigmoid vs Softmax

| | Sigmoid | Softmax |
|---|---|---|
| Problem type | Yes / No | Pick one from many |
| Output | Single number 0–1 | List of probs summing to 1 |
| Example | Is this spam? | Cat, Dog, or Bird? |

---

## 7. Cross Entropy Loss

The proper loss function used with Sigmoid and Softmax.

**Core question:** How confident were you on the correct answer?

```
Loss = -log(probability of correct class)
```

### What log Does to Probabilities

```
-log(0.90) = 0.10   ← small loss    ✅ confident + correct
-log(0.51) = 0.67   ← medium loss   😬 barely confident
-log(0.10) = 2.30   ← huge loss     🚨 confident + WRONG
```

### Visual

```
Loss
 ^
 |  \
 |   \
 |    \
 |     \________
 └─────────────→ Probability of correct class
 0              1
```

As confidence in correct answer grows → loss shrinks toward 0.
As confidence collapses → loss explodes.

### Example

```
Correct answer = Mango 🥭

After Softmax:
  Mango  = 73%
  Apple  = 24%
  Banana =  3%

Cross Entropy only cares about correct class:
  Loss = -log(0.73) = 0.31  ← pretty good!
```

Apple and Banana probabilities are **completely ignored.**

### Why Negative Log?

```
log(0.9) = -0.10   ← wrong direction (high prob = negative number)
log(0.1) = -2.30   ← more negative for low prob

Add negative sign → flips it:
-log(0.9) = 0.10   ← small loss for high confidence ✅
-log(0.1) = 2.30   ← big loss for low confidence   ✅
```

---

## 8. Backpropagation

After the forward pass produces a loss, backprop answers:

> **"Which weights caused the loss — and by how much?"**

### The Factory Analogy

```
Input → Worker A → Worker B → Worker C → Defective Product (loss)
```

Manager walks **backwards** — C first, then B, then A — asking each:
*"How much did YOU contribute to this defect?"*

That's backpropagation.

### Step-by-Step Example

```
Network: Input=2, w₁=3, w₂=0.5, Actual=5

Forward Pass (store every value!):
  z₁ = w₁ × input = 3 × 2   = 6
  z₂ = w₂ × z₁   = 0.5 × 6 = 3
  Loss = (3 - 5)² = 4
```

Now walk backwards:

**Step 1 — Blame the output:**
```
dLoss/dz₂ = 2 × (z₂ - 5) = 2 × (3-5) = -4
```

**Step 2 — Blame w₂:**
```
dLoss/dw₂ = dLoss/dz₂ × z₁ = -4 × 6 = -24
→ Increase w₂ to reduce loss
```

**Step 3 — Pass blame back through w₂:**
```
dLoss/dz₁ = dLoss/dz₂ × w₂ = -4 × 0.5 = -2
```

**Step 4 — Blame w₁:**
```
dLoss/dw₁ = dLoss/dz₁ × input = -2 × 2 = -4
→ Increase w₁ to reduce loss
```

### Result

```
w₂ blame = -24  ← closer to output, more responsible
w₁ blame = -4   ← further back, blame diluted by w₂
```

> **Key insight:** The closer a weight is to the output, the more directly it controls the loss.

---

## 9. Chain Rule — The Engine of Backprop

The chain rule is why blame can travel backwards through layers.

### Plain English

> "If A affects B, and B affects C — then A's effect on C = (A's effect on B) × (B's effect on C)"

### Formula

```
dLoss/dw₁ = dLoss/dz₂ × dz₂/dz₁ × dz₁/dw₁
```

Each layer **multiplies its own contribution** and passes blame further back.

### Visual

```
Loss ← z₂ ← w₂ ← z₁ ← w₁ ← Input

Blame: -4  →  × w₂(0.5)  →  -2  →  × input(2)  →  -4
              (diluted here!)
```

---

## 10. Adam Optimizer

Adam takes the gradients from backprop and uses them **smartly** to update weights.

> Adam = Gradient Descent + per-weight adaptive learning rate

### The Two Problems Adam Solves

**Problem 1 — No memory of direction (plain GD):**
One noisy gradient can send you the wrong way.

**Problem 2 — Same learning rate for all weights:**
Some weights are calm, some are noisy — they shouldn't step at the same speed.

---

### Solution 1 — Momentum (Memory of Direction)

Running average of past gradients:

```
m_t = β₁ × m_(t-1) + (1 - β₁) × g_t

β₁ = 0.9  (remember 90% of past direction)
```

In plain English:
```
new momentum = 90% of old momentum + 10% of today's gradient
```

**Consistent gradients → momentum builds → bold steps:**
```
Gradients: -24, -20, -22, -21, -23

m₁ = 0.9×0    + 0.1×(-24) = -2.4
m₂ = 0.9×-2.4 + 0.1×(-20) = -4.2
m₃ = 0.9×-4.2 + 0.1×(-22) = -6.0
m₄ = 0.9×-6.0 + 0.1×(-21) = -7.5
m₅ = 0.9×-7.5 + 0.1×(-23) = -9.1  ← growing! bold step ✅
```

**Zigzag gradients → momentum cancels → tiny steps:**
```
Gradients: -24, +23, -22, +24

m₁ = -2.4
m₂ = +0.1   ← nearly cancelled!
m₃ = -2.1
m₄ = +0.5   ← nearly cancelled again!
→ momentum stays near 0 → tiny steps ✅
```

---

### Solution 2 — Volatility (Memory of Noise)

Running average of past gradients **squared:**

```
v_t = β₂ × v_(t-1) + (1 - β₂) × g_t²

β₂ = 0.999  (remember 99.9% of past noise)
```

Why squared? Removes the sign — only measures **SIZE of swings:**
```
gradient = -24  →  gradient² = 576  (large swing)
gradient = -2   →  gradient² = 4    (small swing)
gradient = +24  →  gradient² = 576  (same as -24!)
```

**Calm gradients → small volatility → big steps:**
```
Gradients: -2, -2, -2, -2

v₁ = 0.001 × 4  = 0.004
v₂ ≈ 0.008
v₄ ≈ 0.016

√v₄ = 0.13  ← small → big step ✅
```

**Noisy gradients → large volatility → small steps:**
```
Gradients: -24, +23, -22, +24

v₁ = 0.001 × 576 = 0.576
v₂ ≈ 1.104
v₄ ≈ 2.161

√v₄ = 1.47  ← large → small step ✅
```

---

### The Adam Update Formula

```
w_new = w_old - learning_rate × m̂_t / √v̂_t
```

Where `m̂` and `v̂` are bias-corrected versions:
```
m̂_t = m_t / (1 - β₁ᵗ)   ← fixes early steps being too small
v̂_t = v_t / (1 - β₂ᵗ)
```

### The Four Scenarios

| Momentum | Volatility | Result |
|---|---|---|
| High (consistent) | Low (calm) | Bold step 🚀 |
| High (consistent) | High (noisy) | Medium step ⚠️ |
| Low (zigzag) | Low (calm) | Tiny step 🐢 |
| Low (zigzag) | High (noisy) | Barely moves 🛑 |

---

### How Adam Connects to Everything

```
Backprop produces gradients for every weight
         ↓
Adam reads those gradients
         ↓
Computes momentum  (which direction consistently?)
Computes volatility (how noisy has it been?)
         ↓
Updates each weight with a custom smart step
         ↓
Loss drops a little
         ↓
Repeat 100s of times → fully trained network ✅
```

---

## 11. Overfitting

> *"Model memorises training data with no real understanding — fails on new data"*

### The Student Analogy

```
Overfitted model   = Student who memorises past exam answers word for word
Well trained model = Student who understands concepts, solves new problems
```

### How to Detect It

```
Training accuracy = 99%  ← memorised training data
Test accuracy     = 60%  ← fails on new data

Big gap   = OVERFITTING 🚨
Small gap = Generalised well ✅
```

### Why It Happens

```
1. Too little data    → 8 fruits isn't enough to generalise
2. Too many epochs    → 100 chances to memorise every detail
3. Model too complex  → too many weights, memorises noise & outliers
```

### The Full Overfitting Toolkit

```
Fix 1 — Train/Test Split    → detect it early
Fix 2 — More Data           → harder to memorise
Fix 3 — Dropout             → randomly kill neurons
Fix 4 — L2 Regularisation   → shrink all weights
Fix 5 — L1 Regularisation   → kill useless weights
```

---

## 12. Dropout

Randomly **switch off some neurons** every training epoch — forcing every neuron to learn independently.

### The Football Analogy

```
Without Dropout:
  Star player does everything → rest of team never develops
  Star gets injured on match day → team collapses 😬

With Dropout:
  Random players sit out each training session
  Everyone forced to develop → strong team on match day ✅
```

### How It Works

```
Epoch 1: [N1●] [N2 ] [N3●] [N4 ] → Output
Epoch 2: [N1 ] [N2●] [N3 ] [N4●] → Output
Epoch 3: [N1●] [N2●] [N3 ] [N4 ] → Output

● = active,  space = dropped
Every neuron forced to be useful!
```

### Critical Rule

```
Training: Dropout ON  → neurons randomly killed → forces learning
Testing:  Dropout OFF → ALL neurons active      → full power ✅
```

### Choosing Dropout Rate

```
0.2 → drop 20% → light regularisation
0.5 → drop 50% → strong regularisation (most common)
0.8 → drop 80% → too aggressive → underfitting risk
```

### In TensorFlow

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dropout(0.5),   # randomly kill 50% each epoch
    tf.keras.layers.Dense(3, activation='softmax')
])
```

---

## 13. L1 and L2 Regularisation

Both add a **penalty for large weights** to the loss — forcing the network to stay simple.

### Core Idea

```
Normal Loss      = how wrong was the prediction?
Regularised Loss = how wrong was the prediction? + PENALTY for large weights
```

*Like a teacher rewarding students who get the right answer with clean simple working.*

---

### L2 Regularisation — "Shrink Everything"

```
L2 Loss = (pred - actual)² + λ × (w₁² + w₂² + ...)
```

Squaring means bigger weights get punished exponentially more:

```
w = 2  → penalty = 4
w = 10 → penalty = 100  ← 25× more expensive!
```

**Why squaring = percentage shrinking:**
```
L2 gradient = 2λw   ← contains w!
As weight shrinks → chip shrinks → never reaches zero
```

**Salary analogy:** Boss cuts 10% of whatever you earn → cut gets smaller → never broke.

---

### L1 Regularisation — "Kill Useless Weights"

```
L1 Loss = (pred - actual)² + λ × (|w₁| + |w₂| + ...)
```

Fixed chip every epoch regardless of weight size:

```
L1 gradient = λ   ← constant! doesn't depend on w
```

**Salary analogy:** Boss cuts exactly ₹1000 every month → eventually you hit ₹0.

**3 Epoch Example:**
```
w = 0.006, λ = 0.1, lr = 0.01, chip = 0.001 (fixed)

        L1        L2
Start:  0.006     0.006
Ep 1:   0.005     0.005988  ← L2 barely moved!
Ep 2:   0.004     0.005976
Ep 3:   0.003     0.005964

L1 dropped: 0.003
L2 dropped: 0.000036
```

---

### L1 vs L2 Side by Side

| | L1 | L2 |
|---|---|---|
| Penalty | λ × Σ\|wᵢ\| | λ × Σwᵢ² |
| Chip size | Fixed every epoch | Shrinks as weight shrinks |
| Effect | Kills useless weights (→ zero) | Shrinks all weights |
| Best when | Many irrelevant features | Most features useful |
| Analogy | Fixed ₹1000 salary cut | 10% salary cut |

### When to Use Which

```
Most features useful     → L2
Many features irrelevant → L1
Not sure                 → L1 + L2 together (ElasticNet) ← most common
```

### In TensorFlow

```python
# L2
tf.keras.layers.Dense(4, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.01))

# L1
tf.keras.layers.Dense(4, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l1(0.01))

# Both — ElasticNet (most common in practice)
tf.keras.layers.Dense(4, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))
```

---

## 15. Batch Normalisation

> After each layer, rescale all outputs to mean=0, std=1 — so no value dominates the next layer.

### The Problem

Between layers, neuron outputs can be wildly different scales:

```
Neuron 1: 0.001
Neuron 2: 847       ← dominates everything!
Neuron 3: -523      ← dominates everything!
Neuron 4: 0.003
```

The next layer tries to learn from these — but huge values drown out tiny ones.
*Like students answering in cm, km, and m — impossible to compare.*

---

### The Fix — Normalise After Each Layer

```
normalised = (value - mean) / std
```

**Example:**
```
Layer outputs: [0.001, 847, -523, 0.003]

mean = 81
std  = 471

0.001 → (0.001 - 81) / 471 = -0.17
847   → (847   - 81) / 471 = +1.63
-523  → (-523  - 81) / 471 = -1.28
0.003 → (0.003 - 81) / 471 = -0.17

Result: [-0.17, +1.63, -1.28, -0.17]  ← all on same scale ✅
```

---

### Where It Goes in the Network

```
Dense Layer     → raw outputs (wildly different scales)
     ↓
Batch Norm  🔀  → normalise to mean=0, std=1
     ↓
ReLU            → activation
     ↓
Dense Layer     → raw outputs again
     ↓
Batch Norm  🔀  → normalise again
```

Added **after Dense, before activation.**

---

### Three Big Benefits

```
1. Faster training      → gradients flow smoothly backwards
2. More stable          → less sensitive to learning rate
3. Mild regularisation  → slight noise reduces overfitting
```

---

### In TensorFlow

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(2,)),
    tf.keras.layers.BatchNormalization(),   # normalise
    tf.keras.layers.Activation('relu'),     # then activate

    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Dense(3, activation='softmax')
])
```

---

## 16. Evaluation Metrics for Multi-Class Classification

Accuracy has a blind spot — it lies on imbalanced datasets:

```
Dataset: 950 Apples, 30 Mangos, 20 Bananas

Dumb model that always predicts Apple:
Accuracy = 950/1000 = 95%  ← looks great! 🚨 but completely useless
```

Use these instead:

---

### Confusion Matrix — See Every Mistake

```
                Predicted
              Apple  Mango  Banana
Actual Apple  [ 45 ]  [ 3 ]  [ 2 ]   ← 45 correct, 5 wrong
Actual Mango  [  2 ]  [27 ]  [ 1 ]   ← 27 correct, 3 wrong
Actual Banana [  1 ]  [ 2 ]  [17 ]   ← 17 correct, 3 wrong

Diagonal = correct ✅
Off-diagonal = mistakes 🚨
```

Shows WHICH classes confuse your model — not just how many are wrong.

---

### Precision — "When you said Mango, were you right?"

```
Precision = True Positives / (True Positives + False Positives)

Model predicted Mango 30 times:
  27 actually Mango  ← True Positives
   3 actually Apple  ← False Positives

Precision = 27 / 30 = 0.90  → 90% of Mango predictions correct ✅
```

---

### Recall — "Of all actual Mangos, how many did you catch?"

```
Recall = True Positives / (True Positives + False Negatives)

30 actual Mangos in dataset:
  27 correctly predicted  ← True Positives
   3 missed               ← False Negatives

Recall = 27 / 30 = 0.90  → caught 90% of all Mangos ✅
```

---

### Precision vs Recall Trade-off

```
Medical diagnosis (cancer):
  Missing cancer = catastrophic 🚨
  → Maximise RECALL (catch everything, even false alarms)

Spam filter:
  Blocking real email = bad 😬
  → Maximise PRECISION (only flag when very sure)
```

---

### F1 Score — Best of Both Worlds

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Precision = 0.90, Recall = 0.90
F1 = 2 × (0.90 × 0.90) / (0.90 + 0.90) = 0.90

Macro F1 (average across all classes):
  Apple  F1 = 0.91
  Mango  F1 = 0.90
  Banana F1 = 0.86
  ──────────────────
  Macro  F1 = 0.89  ← single overall score
```

---

### ROC-AUC — How Well Does It Separate Classes?

```
AUC = 1.0  → perfect separation ✅
AUC = 0.5  → random guessing   😬
AUC = 0.0  → perfectly wrong   🚨
```

For multi-class → one-vs-rest AUC per class, then average.

---

### Metrics Cheat Sheet

| Metric | Best For | Weakness |
|---|---|---|
| Accuracy | Balanced datasets | Lies on imbalanced data |
| Confusion Matrix | Understanding mistakes | Not a single number |
| Precision | False positives costly | Ignores false negatives |
| Recall | False negatives costly | Ignores false positives |
| F1 Score | Imbalanced datasets | Treats P and R equally |
| ROC-AUC | Ranking confidence | Harder to interpret |

**In practice → always use: Confusion Matrix + Macro F1**

---

### In Code

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_prob = model.predict(X_test)

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Precision, Recall, F1 per class
print(classification_report(y_test, y_pred,
      target_names=['Apple', 'Mango', 'Banana']))

# ROC-AUC
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f"ROC-AUC: {auc:.3f}")
```

---

## 17. SHAP Explainability for Neural Networks

SHAP answers: *"Why did the model predict THIS for THIS input?"*

It assigns each feature a **contribution score** per prediction:

```
Fruit: sweetness=8.0, size=3.5  →  Predicted: Mango (89%)

SHAP explanation:
  Base value (average)    =  0.13
  sweetness=8.0 pushed   +  0.45  ← main reason for Mango
  size=3.5 pushed        +  0.31  ← supporting reason
  ──────────────────────────────
  Final prediction        =  0.89 ✅
```

---

### Two Methods for Neural Networks

```
DeepExplainer   → designed for deep learning (TF/PyTorch)
                  fast, uses backprop-like approach ✅

KernelExplainer → model-agnostic, works on ANY model
                  slower but more flexible
                  use when DeepExplainer fails
```

---

### In Code

```python
import shap

# DeepExplainer (faster, NN-specific)
explainer  = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Summary plot — feature importance across all predictions
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=['sweetness', 'size'],
    class_names=['Apple', 'Mango', 'Banana']
)

# Waterfall plot — explain ONE specific prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[1][0],              # class Mango, first sample
        base_values=explainer.expected_value[1],
        data=X_test[0],
        feature_names=['sweetness', 'size']
    )
)
```

---

### What the Plots Show

```
Summary Plot:
  Feature      Impact on Mango
  sweetness    ████████████  +0.45  ← most important
  size         ███████       +0.31

Waterfall Plot (single fruit):
  Base value        0.13
  + sweetness=8.0  +0.45
  + size=3.5       +0.31
  ──────────────────────
  Mango prediction  0.89 ✅
```

---

### ⚠️ Key Caveat

```
Simple models (linear, trees) → SHAP is EXACT ✅
Neural networks               → SHAP is APPROXIMATE ⚠️

NNs are too complex for exact attribution.
Still very useful — just not mathematically perfect.
```

---

### Best Practice for Production

```
SHAP             → feature contribution per prediction
Confusion Matrix → which classes are confused
Macro F1         → overall model quality
```

---

## 18. Full TensorFlow Code

Everything above — in 40 lines of code.

```python
import tensorflow as tf
import numpy as np

# ─────────────────────────────────────────────
# STEP 1 — THE DATA
# X = inputs [sweetness, size]
# y = correct answers  0=Apple, 1=Mango, 2=Banana
# ─────────────────────────────────────────────
X = tf.constant([
    [2.0, 1.0],   # Apple
    [8.0, 3.0],   # Mango
    [5.0, 2.0],   # Banana
    [3.0, 1.5],   # Apple
    [9.0, 4.0],   # Mango
    [6.0, 2.5],   # Banana
    [1.0, 1.0],   # Apple
    [7.0, 3.5],   # Mango
], dtype=tf.float32)

y = tf.constant([0, 1, 2, 0, 1, 2, 0, 1])

# ─────────────────────────────────────────────
# STEP 2 — BUILD THE NETWORK
# Input(2) → Dense+ReLU → Dense+Softmax → Output(3)
# ─────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # ReLU ✅
    tf.keras.layers.Dense(3, activation='softmax')                  # Softmax ✅
])

# ─────────────────────────────────────────────
# STEP 3 — COMPILE
# Adam optimizer + Cross Entropy loss
# ─────────────────────────────────────────────
model.compile(
    optimizer='adam',                           # Adam ✅
    loss='sparse_categorical_crossentropy',     # Cross Entropy ✅
    metrics=['accuracy']
)

# ─────────────────────────────────────────────
# STEP 4 — TRAIN
# Each epoch: Forward → Loss → Backprop → Adam update
# ─────────────────────────────────────────────
model.fit(X, y, epochs=100, verbose=1)

# ─────────────────────────────────────────────
# STEP 5 — PREDICT on NEW unseen fruits
# ─────────────────────────────────────────────
new_fruits = tf.constant([
    [7.0, 3.0],   # Should be Mango?
    [2.5, 1.0],   # Should be Apple?
    [5.5, 2.0],   # Should be Banana?
], dtype=tf.float32)

predictions = model.predict(new_fruits)
fruit_names = ['Apple 🍎', 'Mango 🥭', 'Banana 🍌']

for i, pred in enumerate(predictions):
    predicted_fruit = fruit_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(f"\nFruit {i+1}:")
    print(f"  Apple:  {pred[0]*100:.1f}%")
    print(f"  Mango:  {pred[1]*100:.1f}%")
    print(f"  Banana: {pred[2]*100:.1f}%")
    print(f"  → {predicted_fruit} ({confidence:.1f}% confident)")
```

---

## The Complete Training Loop — Everything Connected

```
Input Data (X)
     ↓
Forward Pass
  Dense + ReLU       → kills negatives, adds non-linearity
  Dense + Softmax    → raw scores → probabilities
     ↓
Cross Entropy Loss   → how confident on correct answer?
     ↓
Backpropagation      → chain rule assigns blame to each weight
     ↓
Adam Optimizer       → momentum + volatility → smart weight update
     ↓
Repeat × 100 epochs  → network gets smarter each time!
```

---

## 15. Hyperparameters

> Parameters = weights learned automatically. Hyperparameters = settings WE choose before training.

### All Hyperparameters at a Glance

```
1. Learning Rate       → how big each step is
2. Epochs              → how many training loops
3. Batch Size          → how many samples per weight update
4. Network Depth       → how many layers
5. Network Width       → how many neurons per layer
6. Dropout Rate        → how many neurons to kill
7. Lambda (λ)          → L1/L2 penalty strength
8. Optimizer           → Adam, SGD, RMSProp...
9. Activation Function → ReLU, Sigmoid, Softmax
```

---

### Batch Size — The Key One to Understand

```
Batch Size = how many samples the network sees before updating weights
```

Say you have 1 million fruits, batch size = 50,000:

```
Epoch 1:
  Batch 1: 50,000 fruits → loss → update weights
  Batch 2: 50,000 fruits → loss → update weights
  ...
  Batch 20: done! (1M / 50K = 20 updates per epoch)
```

**Three types:**

```
Batch GD   → batch = ALL data    → 1 update/epoch, memory heavy 😬
Mini-Batch → batch = 32–512      → multiple updates/epoch ✅ most common
SGD        → batch = 1           → very noisy, very fast 😵
```

**Trade-off:**
```
Large batch → stable gradient, more memory, risk overfitting
Small batch → noisy (acts like regularisation!), less memory
Default: batch_size = 32
```

---

### How to Choose Each Hyperparameter

**1. Learning Rate — Most Important**
```
Too high  → loss explodes 🚨
Too low   → training crawls 🐢
Default   → 0.001 (Adam) works 90% of the time

Too slow? → try 0.01
Exploding? → try 0.0001
```

**2. Epochs — Use Early Stopping**

Don't guess. Let the network tell you when to stop:

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10        # stop if no improvement for 10 epochs
)
model.fit(X, y, epochs=1000, callbacks=[early_stop])
```

Set epochs high (1000), early stopping decides automatically. ✅

**3. Batch Size**
```
Default:         32   ← start here always
Large dataset:   128 or 256
Memory issues:   16
```

**4. Depth & Width — Rules of Thumb**
```
Simple problem   → 1-2 layers, 4-64 neurons
Medium problem   → 2-4 layers, 64-256 neurons
Complex problem  → 5+ layers, 256-1024 neurons
```

Start small. Add complexity only if underfitting.

**5. Grid Search — When Unsure**

Try combinations systematically:
```
Learning rates: [0.1, 0.01, 0.001]
Batch sizes:    [32, 64, 128]
Dropout:        [0.2, 0.5]
→ pick best test accuracy combination
```

---

### Quick Decision Guide

```
Hyperparameter    Start With              Tune If
──────────────────────────────────────────────────────
Learning Rate     0.001                  loss explodes/crawls
Epochs            1000 + EarlyStopping   gap between train/test
Batch Size        32                     memory issues
Depth             2 hidden layers        underfitting
Width             64 neurons             underfitting
Dropout           0.5                    overfitting
Lambda            0.01                   overfitting
Optimizer         Adam                   rarely change
```

> **Golden Rule:** You don't calculate hyperparameters — you experiment.
> Start with defaults → watch train vs test gap → adjust accordingly.

---

## 🚀 What's Next

```
✅ Completed:
   Forward Pass → Loss → ReLU → Sigmoid → Softmax
   → Cross Entropy → Backprop → Chain Rule → Adam
   → Overfitting → Dropout → L1 → L2
   → Hyperparameters → Batch Normalisation
   → Evaluation Metrics → SHAP Explainability

📌 Next topics:
   1. Convolutional Neural Networks (CNNs)
      (how networks understand images)
   2. Transformers
      (the architecture behind ChatGPT and LLMs)
   3. Transfer Learning
      (reuse pretrained models instead of training from scratch)
```
