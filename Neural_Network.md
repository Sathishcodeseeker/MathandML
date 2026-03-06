# Deep Learning — From Scratch to Adam Optimizer
> Everything we covered today, built step by step.

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
11. [Full TensorFlow Code](#11-full-tensorflow-code)

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

## 11. Full TensorFlow Code

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

## 🚀 What's Next

```
✅ Completed today:
   Forward Pass → Loss → ReLU → Sigmoid → Softmax
   → Cross Entropy → Backprop → Chain Rule → Adam

📌 Next topics:
   1. Overfitting & Regularisation
      (what if the network memorises instead of learns?)
   2. Train/Test Split
      (how do we know if the model truly generalised?)
   3. Dropout
      (randomly killing neurons to force generalisation)
```
