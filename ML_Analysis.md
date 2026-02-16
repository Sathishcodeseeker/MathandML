# Complete Algorithm Reference

## Tier 1: Production Workhorses (80% of ML models)

### Supervised Learning – Tabular Data

**Gradient Boosting (XGBoost / LightGBM / CatBoost) 🏆**  
- **Usage:** 70–80% of tabular ML problems  
- **Where:** Finance, e-commerce, insurance, healthcare  
- **Why dominant:** Best accuracy on structured data, fast, handles missing values  
- **Learn priority:** ⭐⭐⭐⭐⭐ (HIGHEST)

**Logistic Regression**  
- **Usage:** Extremely high  
- **Where:** A/B testing, CTR prediction, baseline models  
- **Why:** Interpretable, fast, reliable, regulation-friendly  
- **Learn priority:** ⭐⭐⭐⭐⭐

**Linear Regression**  
- **Usage:** Very high  
- **Where:** Pricing, forecasting, demand prediction  
- **Why:** Simple, interpretable, good baseline  
- **Learn priority:** ⭐⭐⭐⭐⭐

**Random Forests**  
- **Usage:** High (declining vs. boosting)  
- **Where:** Classification, feature importance, ensembles  
- **Why:** Robust, less overfitting than single trees  
- **Learn priority:** ⭐⭐⭐⭐

**Support Vector Machines (SVM)**  
- **Usage:** Medium (declining)  
- **Where:** Small datasets, text classification (legacy)  
- **Status:** Being replaced by neural networks and boosting  
- **Learn priority:** ⭐⭐ (Optional)

---

### Deep Learning – Text / NLP

**Transformers (BERT, GPT, T5, etc.) 🚀**  
- **Usage:** Exploding growth (30%+ of NLP, up from 5% in 2022)  
- **Where:** Chatbots, search, translation, code generation  
- **Approach:** Use pre-trained models via APIs (OpenAI, Anthropic)  
- **Learn priority:** ⭐⭐⭐⭐⭐

**Word Embeddings (Word2Vec, GloVe)**  
- **Usage:** Declining (replaced by transformers)  
- **Where:** Legacy systems, resource-constrained apps  
- **Learn priority:** ⭐⭐ (Historical knowledge)

---

### Deep Learning – Computer Vision

**Convolutional Neural Networks (CNNs)**  
- **Usage:** Very high in vision tasks  
- **Where:** Manufacturing defects, medical imaging, autonomous vehicles  
- **Approach:** Transfer learning from pre-trained models (ResNet, EfficientNet)  
- **Learn priority:** ⭐⭐⭐⭐

**YOLO / Object Detection**  
- **Usage:** High in specific domains  
- **Where:** Real-time detection, surveillance, autonomous driving  
- **Learn priority:** ⭐⭐⭐ (If doing computer vision)

**U-Net / Segmentation Models**  
- **Usage:** Medium–high in specialized domains  
- **Where:** Medical imaging, satellite imagery  
- **Learn priority:** ⭐⭐ (Specialized)

---

## Tier 2: Important but Specialized

### Clustering & Unsupervised Learning

**K-Means Clustering**  
- **Usage:** High for clustering  
- **Where:** Customer segmentation, image compression  
- **Why popular:** Simple, fast, interpretable  
- **Learn priority:** ⭐⭐⭐⭐

**DBSCAN (Density-Based Clustering)**  
- **Usage:** Medium–high  
- **Where:** Anomaly detection, geographic data, fraud detection  
- **Advantage:** Finds arbitrary shapes, identifies outliers automatically  
- **Learn priority:** ⭐⭐⭐

**Hierarchical Clustering**  
- **Usage:** Medium  
- **Where:** Taxonomies, dendrograms, biological data  
- **Learn priority:** ⭐⭐

**HDBSCAN**  
- **Usage:** Growing  
- **Where:** Better version of DBSCAN for varying densities  
- **Learn priority:** ⭐⭐ (Advanced)

**Gaussian Mixture Models (GMM)**  
- **Usage:** Medium  
- **Where:** Soft clustering, density estimation  
- **Learn priority:** ⭐⭐

---

### Dimensionality Reduction

**Principal Component Analysis (PCA)**  
- **Usage:** Very high  
- **Where:** Feature reduction, visualization, noise reduction  
- **Learn priority:** ⭐⭐⭐⭐

**t-SNE**  
- **Usage:** High for visualization only  
- **Where:** Data visualization, exploratory analysis  
- **Limitation:** Slow, not for feature extraction  
- **Learn priority:** ⭐⭐⭐

**UMAP**  
- **Usage:** Growing (replacing t-SNE)  
- **Where:** Visualization, feature extraction  
- **Advantage:** Faster than t-SNE, preserves global structure  
- **Learn priority:** ⭐⭐⭐

**Autoencoders**  
- **Usage:** Medium  
- **Where:** Dimensionality reduction, anomaly detection, denoising  
- **Learn priority:** ⭐⭐⭐

---

## Metric Selection Guide

**Is it Classification?**  
- **Yes → Is data balanced?**  
  - **Yes →** Accuracy, F1-Score, ROC-AUC  
  - **No →** F1/F2-Score, PR-AUC, MCC  

**No → Is it Regression?**  
- **Yes → Are outliers important?**  
  - **Yes →** RMSE  
  - **No →** MAE  

**No → Is it Ranking?**  
- **Yes →** NDCG, MAP
