# Univariate, Multivariate, EDA Flow, and Data Leakage — Complete Notes

---

# 1. Univariate Analysis

## Meaning

**Univariate analysis** studies **one variable at a time** to understand its:

* Distribution
* Central value
* Spread
* Outliers
* Shape

---

## What We Measure

### Central Tendency

* Mean
* Median
* Mode

### Spread

* Range
* Variance
* Standard deviation
* Interquartile Range (IQR)

### Distribution Shape

* Skewness (left/right tail)
* Kurtosis (extreme values vs flatness)

### Outlier Detection

* Z‑score method (|z| > 3)
* IQR rule:

  * Lower = Q1 − 1.5 × IQR
  * Upper = Q3 + 1.5 × IQR

---

## Visualizations

* Histogram
* Box plot
* Density plot
* Bar chart (categorical)

---

## Role in Data Science

Univariate analysis is the **first step of EDA** before:

* Bivariate analysis
* Multivariate analysis

---

# 2. Multivariate Analysis

## Meaning

Studies **three or more variables together** to understand:

* Relationships
* Interactions
* Hidden structure
* Prediction capability

---

## Major Technique Groups

### Dimensionality Reduction

* PCA
* t‑SNE
* UMAP
* Autoencoders

**Purpose:** reduce many features → few informative components.

---

### Regression Methods

* Multiple Linear Regression
* Polynomial Regression
* Ridge / Lasso / Elastic Net

**Purpose:** predict numeric target.

---

### Classification Methods

* Logistic Regression
* Decision Trees / Random Forest
* SVM
* Neural Networks
* Gradient Boosting (XGBoost, LightGBM, CatBoost)

**Purpose:** predict categories.

---

### Clustering Methods

* K‑Means
* Hierarchical clustering
* DBSCAN
* Gaussian Mixture Models
* Spectral clustering

**Purpose:** discover natural groups.

---

### Statistical Multivariate Methods

* MANOVA
* Factor Analysis
* Canonical Correlation
* Discriminant Analysis

---

# 3. Multivariate Analysis Inside EDA

## Core Activities

### Relationship Analysis

* Correlation matrix
* Multicollinearity detection
* Heatmaps and pair plots

**Outcome:** identify useful vs redundant features.

---

### Feature Interaction Exploration

* 3D scatter plots
* Grouped box/violin plots
* Facet plots

**Outcome:** discover nonlinear or joint effects.

---

### Dimensionality Understanding

* PCA variance explained
* t‑SNE / UMAP visualization

**Outcome:** detect structure vs randomness.

---

### Cluster Detection

* K‑Means, DBSCAN, hierarchical clustering
* PCA visualization of clusters

**Outcome:** understand separability before modeling.

---

### Multidimensional Outlier Detection

* Mahalanobis distance
* Isolation Forest
* DBSCAN noise points

**Outcome:** remove harmful anomalies.

---

### Feature Selection Insight

* Correlation filtering
* PCA loadings
* Mutual information

**Outcome:** cleaner dataset and better ML performance.

---

# 4. What Happens After EDA

## Standard ML Workflow

1. Data preprocessing and cleaning
2. Feature engineering and selection
3. Train/validation/test split
4. Model selection
5. Training and hyperparameter tuning
6. Model evaluation
7. Model interpretation
8. Deployment and monitoring

---

## Full Real‑World Lifecycle

* Problem framing and success metrics
* Iterative data preparation
* Modeling and validation
* Business interpretation
* Productionization (MLOps)
* Monitoring, retraining, drift handling
* Governance, ethics, compliance

---

# 5. Data Leakage

## Definition

**Data leakage** occurs when the model uses **information unavailable at real prediction time**, causing:

* Unrealistically high test accuracy
* Failure in production

In simple terms: **the model is cheating**.

---

## Common Types

### Train–Test Leakage

* Preprocessing before split
* Feature selection on full data
* Duplicate rows across splits

---

### Target Leakage

* Feature directly reveals label

Examples:

* Loan approved amount → predicting approval
* Post‑treatment result → predicting disease

---

### Time Leakage

* Random split in time‑series
* Using future values to predict past

---

### Preprocessing Leakage

* Scaling, imputation, PCA fitted on full dataset instead of training set only

---

## How to Check for Leakage

1. Confirm each feature exists at prediction time
2. Split data **before** preprocessing
3. Question suspiciously high accuracy
4. Use time‑aware validation for time series
5. Check extreme correlation with target
6. Remove duplicate or ID‑based leakage

---

# Final Summary

## EDA Flow

**Univariate → Bivariate → Multivariate → Modeling → Deployment**

## Critical Principle

Without **leakage checks**, even highly accurate models may be **useless in reality**.

---

**End of Notes**
