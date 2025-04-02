# 🩺 Heart Disease Classification: Cost-Sensitive & Imbalance Learning

This project explores **cost-sensitive learning** and **class imbalance techniques** to improve model performance on the **Statlog (Heart)** dataset. We experiment with methods like rebalancing, weighting, SMOTE, and EasyEnsemble using **RandomForest, SVM, and Naive Bayes** models.

## 📁 Dataset
**Statlog (Heart)** dataset from UCI Machine Learning Repository:  
🔗 [Dataset Link](https://archive.ics.uci.edu/dataset/145/statlog+heart)  
- Predicts presence of heart disease (binary classification).
- Contains 270 samples with 13 features (clinical & demographic).

## 🧪 Part 1: Cost-Sensitive Learning

### ⚖️ Rebalancing (Over/Under-Sampling)
- Adjust class distribution based on misclassification costs to prioritize costly classes.

### 🏋️ Class Weighting
- Apply `class_weight` in models (e.g., RandomForest) to penalize costly class errors.

### 🎯 Minimizing Expected Cost
- **`CalibratedClassifierCV`**:
  - **Sigmoid/Isotonic** calibration to align predicted probabilities with true frequencies.
  - Optimizes decision thresholds for cost-sensitive predictions.

---

## 🧬 Part 2: Class Imbalance Techniques

### 🌟 SMOTE (Synthetic Oversampling)
- Generates synthetic samples for minority classes using k-NN.

### ✂️ NearMiss (Under-Sampling)
- Reduces majority class samples to balance distribution.

### 📚 EasyEnsemble
- **What it does**:  
  Uses ensemble learning with multiple balanced sub-samples (via under-sampling) and aggregates predictions to improve robustness.

---
