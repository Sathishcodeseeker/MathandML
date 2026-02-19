# Evaluation Metrics Reference Guide

A comprehensive reference for evaluation metrics across common ML tasks and algorithms.

---

## 1. Linear Regression

Linear regression models are evaluated based on how well predictions match continuous target values.

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** (Mean Absolute Error) | `mean(|y - ŷ|)` | Average absolute deviation; same unit as target |
| **MSE** (Mean Squared Error) | `mean((y - ŷ)²)` | Penalizes large errors more; sensitive to outliers |
| **RMSE** (Root Mean Squared Error) | `√MSE` | Same unit as target; most widely reported |
| **R² (R-Squared)** | `1 - SS_res/SS_tot` | Proportion of variance explained; 1.0 = perfect |
| **Adjusted R²** | `1 - (1-R²)(n-1)/(n-p-1)` | Penalizes for extra features; use with multiple regression |
| **MAPE** | `mean(|y - ŷ|/|y|) × 100` | Percentage error; avoid when y has zeros |

### Diagnostic Checks

- **Residual Plots** — Residuals should be randomly scattered (no patterns)
- **Homoscedasticity** — Constant variance of residuals across predictions
- **Durbin-Watson Statistic** — Tests for autocorrelation in residuals (ideal: 1.5–2.5)
- **VIF (Variance Inflation Factor)** — Detects multicollinearity; VIF > 10 is problematic

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)
```

---

## 2. Logistic Regression (Classification)

Logistic regression outputs probabilities for discrete classes, requiring classification metrics.

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` | Misleading with class imbalance |
| **Precision** | `TP/(TP+FP)` | How many predicted positives are correct |
| **Recall (Sensitivity)** | `TP/(TP+FN)` | How many actual positives are captured |
| **F1-Score** | `2 × (P × R)/(P + R)` | Harmonic mean of Precision & Recall |
| **AUC-ROC** | Area under ROC curve | Discrimination ability; 0.5 = random, 1.0 = perfect |
| **Log Loss** | `-mean(y·log(ŷ)+(1-y)·log(1-ŷ))` | Penalizes confident wrong predictions |
| **MCC** | Balanced metric even for imbalanced classes | Range: -1 to +1 |

### Additional Diagnostics

- **Confusion Matrix** — Full breakdown of TP, TN, FP, FN
- **Calibration Curve** — Checks if predicted probabilities match actual frequencies
- **Precision-Recall AUC** — Preferred over ROC-AUC for heavily imbalanced datasets
- **Brier Score** — Mean squared error of probability estimates (lower = better)

```python
from sklearn.metrics import (classification_report, roc_auc_score,
                             log_loss, confusion_matrix)

print(classification_report(y_true, y_pred))
auc      = roc_auc_score(y_true, y_prob)
logloss  = log_loss(y_true, y_prob)
cm       = confusion_matrix(y_true, y_pred)
```

---

## 3. Dimensionality Reduction — UMAP

UMAP (Uniform Manifold Approximation and Projection) is a non-linear dimensionality reduction technique. Evaluation is indirect since there are no ground-truth labels for the embedding itself.

### Intrinsic Metrics (No Labels Required)

| Metric | Description |
|--------|-------------|
| **Trustworthiness** | Measures how well local neighborhoods are preserved (0–1, higher = better) |
| **Continuity** | Measures whether nearby points in high-dim remain nearby in low-dim |
| **Neighborhood Preservation** | % of k-nearest neighbors retained after projection |
| **Reconstruction Error** | Distance mismatch between original and projected pairwise distances |

### Extrinsic Metrics (Labels Required)

| Metric | Description |
|--------|-------------|
| **KNN Classifier Accuracy** | Train KNN on embedding; high accuracy = structure preserved |
| **Silhouette Score on Embedding** | Checks class separability in reduced space |
| **Linear Separability** | Logistic regression accuracy on embedding as proxy |

### Qualitative Checks

- Visual cluster cohesion and separation in 2D/3D scatter plots
- Stability across multiple random seeds
- Sensitivity analysis on `n_neighbors` and `min_dist` hyperparameters

```python
from sklearn.manifold import trustworthiness

trust = trustworthiness(X_original, X_embedded, n_neighbors=10)

# KNN proxy evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X_embedded, y_labels, cv=5)
```

---

## 4. Clustering — HDBSCAN

HDBSCAN (Hierarchical Density-Based Spatial Clustering) identifies clusters of varying density and labels noise points as `-1`.

### Intrinsic Metrics (No Ground Truth)

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | -1 to 1 | Measures intra-cluster cohesion vs inter-cluster separation; higher = better |
| **Davies-Bouldin Index** | ≥ 0 | Average ratio of within/between cluster distances; lower = better |
| **Calinski-Harabasz Index** | ≥ 0 | Ratio of between/within cluster dispersion; higher = better |
| **DBCV** (Density-Based Cluster Validation) | -1 to 1 | Designed specifically for density-based clusters; native to HDBSCAN |

### Extrinsic Metrics (Ground Truth Available)

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **ARI** (Adjusted Rand Index) | -1 to 1 | Corrected-for-chance cluster agreement |
| **NMI** (Normalized Mutual Info) | 0 to 1 | Information shared between clusters and labels |
| **Homogeneity** | 0 to 1 | Each cluster contains only one class |
| **Completeness** | 0 to 1 | All members of a class are in the same cluster |
| **V-Measure** | 0 to 1 | Harmonic mean of Homogeneity & Completeness |

### HDBSCAN-Specific Diagnostics

- **Noise Ratio** — Proportion of points labeled as noise (`-1`); high ratio signals poor `min_cluster_size`
- **Number of Clusters** — Compare to domain expectations
- **Cluster Persistence** — HDBSCAN's internal measure of cluster stability (higher = more stable)
- **Condensed Tree Visualization** — Inspect cluster hierarchy for meaningful structure

```python
import hdbscan
from sklearn.metrics import silhouette_score, adjusted_rand_score

clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
labels = clusterer.fit_predict(X)

mask = labels != -1  # exclude noise
sil  = silhouette_score(X[mask], labels[mask])
ari  = adjusted_rand_score(y_true, labels)

noise_ratio = (labels == -1).sum() / len(labels)
print(f"Noise: {noise_ratio:.2%} | Clusters: {labels.max()+1} | Silhouette: {sil:.3f}")
```

---

## 5. Anomaly Detection — Isolation Forest

Isolation Forest assigns an anomaly score to each sample based on how quickly it is isolated in random trees. Lower scores indicate anomalies.

### Core Metrics

| Metric | Description |
|--------|-------------|
| **Anomaly Score** | Raw output from `decision_function()`; more negative = more anomalous |
| **Contamination** | Assumed fraction of anomalies; tunes the decision threshold |
| **Precision @ K** | Of top-K anomaly-scored samples, how many are true anomalies |
| **Recall @ K** | Of all true anomalies, how many appear in top-K |
| **AUC-ROC** | If ground truth exists; discriminates anomalies from normals |
| **Average Precision (AP)** | Area under Precision-Recall curve; better for imbalanced anomaly data |

### Threshold Tuning Metrics

- **F1-Score at various thresholds** — Optimize threshold on validation set
- **False Positive Rate (FPR)** — Critical in operational settings (alert fatigue)
- **False Negative Rate (FNR)** — Missing real anomalies; costly in safety-critical domains

### Without Ground Truth

- **Score Distribution Analysis** — Visualize score histogram; look for bimodal separation
- **Consistency Check** — Anomalies should be domain-verifiable on manual inspection
- **Cross-validation of contamination** — Try multiple `contamination` values and compare cluster overlap with known rare events

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)

scores = iso.decision_function(X_test)   # higher = more normal
preds  = iso.predict(X_test)             # 1 = normal, -1 = anomaly

# If ground truth available (1 = normal, 0 = anomaly)
auc = roc_auc_score(y_true, scores)
ap  = average_precision_score(y_true, scores)
```

---

## 6. Predictions — XGBoost

XGBoost is a gradient-boosted tree ensemble applicable to regression, classification, and ranking tasks. Metrics depend on the objective.

### Classification Metrics

| Metric | Use Case |
|--------|----------|
| **Accuracy, Precision, Recall, F1** | Balanced or imbalanced classification |
| **AUC-ROC** | Probability ranking; threshold-independent |
| **AUC-PR** | Preferred when positive class is rare |
| **Log Loss** | Probability calibration |
| **MCC** | Single balanced metric for any class distribution |

### Regression Metrics

| Metric | Use Case |
|--------|----------|
| **RMSE / MAE** | Standard error metrics (XGBoost default: RMSE) |
| **RMSLE** | When target spans several orders of magnitude |
| **Huber Loss** | Robust to outliers |
| **R²** | Variance explained |

### Model Diagnostics

| Tool | Description |
|------|-------------|
| **Feature Importance** (`gain`, `weight`, `cover`) | Which features drive predictions |
| **SHAP Values** | Additive, per-prediction feature attribution |
| **Learning Curves** | Train vs. eval metric over boosting rounds (detect overfitting) |
| **Early Stopping Rounds** | Optimal number of trees based on validation metric |

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import shap

model = xgb.XGBClassifier(
    n_estimators=500,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_prob = model.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

# SHAP explanations
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## Quick Reference Summary

| Task | Primary Metric | Secondary Metrics |
|------|---------------|-------------------|
| Linear Regression | RMSE, R² | MAE, Adjusted R², Residual plots |
| Logistic Regression | AUC-ROC, F1 | Log Loss, Precision-Recall AUC, MCC |
| UMAP | Trustworthiness | Neighborhood preservation, KNN accuracy |
| HDBSCAN | Silhouette, DBCV | ARI (if labels), Noise ratio, Persistence |
| Isolation Forest | AUC-ROC, AUC-PR | Precision@K, FPR, Score distribution |
| XGBoost | Task-dependent | SHAP values, Learning curves, Feature importance |

---

Here's the ordered list:

1. Mean, Median, Variance, Standard Deviation
2. Probability Theory & Conditional Probability
3. Bayes' Theorem
4. Probability Distributions (Bernoulli, Gaussian, Chi-squared, F-distribution)
5. Dot Products & Vector Spaces
6. Distance Metrics (Euclidean, Cosine, Manhattan)
7. Eigenvalues & Eigenvectors
8. Entropy & Information Theory
9. KL Divergence & Cross-Entropy
10. Probability Density Functions (PDF)
11. Kernel Density Estimation
12. Hypothesis Testing & p-values
13. Confidence Intervals
14. Bias-Variance Tradeoff
15. Sensitivity & Specificity
16. Threshold-based Decision Making
17. Ranking & Order Statistics
18. Concordance & Mann-Whitney U

---
