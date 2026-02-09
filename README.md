# 🔐 Anomaly Detection for Network Intrusion Detection  
**Gaussian Mixture Models, PCA, and Classical Machine Learning**

This repository contains a **notebook-based implementation** of anomaly detection techniques applied to **network intrusion detection systems (NIDS)**.  
The project focuses on modeling **normal network behavior** and detecting anomalous traffic using **probabilistic models and classical machine learning algorithms**.

---

## Overview

With the rapid growth of networked systems and IoT devices, detecting cyberattacks has become increasingly complex.  
Traditional rule-based security mechanisms struggle to detect **unknown or novel attacks**, motivating the use of **anomaly detection techniques**.

This project explores:
- probabilistic modeling of normal network traffic,
- feature transformation and dimensionality analysis,
- comparison of multiple unsupervised and supervised **classical machine learning models** for intrusion detection.

---

## Project Scope

The work is structured around **two complementary notebooks**:

1. **Probabilistic Anomaly Detection**
   - Gaussian Mixture Models (GMM)
   - Feature-wise probability modeling
   - Voting-based anomaly decision strategy

2. **Classical Machine Learning for IDS**
   - K-Nearest Neighbors (KNN)
   - Decision Trees
   - Support Vector Machines (SVM)
   - Random Forest
   - Logistic Regression
   - Model comparison across attack categories

Both notebooks follow a **CRISP-DM methodology**:  
Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.

---

## What’s Inside the Notebooks

### 1) Data Understanding & Exploration
- Analysis of network traffic features
- Study of quantitative and categorical variables
- Visualization of feature distributions
- Analysis of class imbalance across attack categories

### 2) Data Preparation
- Removal of redundant and non-relevant features
- Handling categorical variables (encoding)
- Feature scaling and normalization
- Outlier analysis
- Train/test dataset splitting
- Feature selection using statistical techniques

### 3) Dimensionality & Feature Transformation
- **Principal Component Analysis (PCA)**
  - Used to obtain uncorrelated features
  - Preserves most of the original variance
- Construction of multiple dataset variants:
  - raw
  - normalized
  - PCA-transformed
  - probability-transformed

### 4) Anomaly Detection with GMM
- Feature-wise Gaussian Mixture Models
- Estimation of probability distributions for normal behavior
- Aggregation of feature probabilities using a **voting scheme**
- Fully unsupervised training on normal traffic only

### 5) Classical Machine Learning Models
- K-Means clustering
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Tree
- Random Forest
- Logistic Regression
- Multilayer Perceptron (MLP)

### 6) Evaluation
- Confusion matrices
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Sensitivity
  - Intrusion Detection Capacity
- Comparative analysis across models and datasets

---

## Dataset

The experiments are conducted using the **NSL-KDD dataset**, a refined version of the KDD Cup 99 dataset designed for intrusion detection research.

**Dataset characteristics:**
- Network connection records
- 41 original features (numeric and categorical)
- Reduced redundancy compared to KDD’99
- Separate training and testing sets with unseen attacks

**Attack categories:**
- DoS (Denial of Service)
- Probe
- R2L (Remote to Local)
- U2R (User to Root)
- Normal traffic

**Why NSL-KDD:**
- Eliminates duplicate records
- Provides more reliable evaluation
- Widely used benchmark for IDS research


---

## Technologies & Libraries

### Core / Scientific Stack
- Python
- NumPy
- Pandas

### Machine Learning
- scikit-learn
- Gaussian Mixture Models (GMM)
- KNN
- Decision Trees
- SVM
- Random Forest
- Logistic Regression
- Multilayer Perceptron (MLP)

### Visualization
- Matplotlib
- Seaborn

### Dimensionality Reduction
- PCA (Principal Component Analysis)

### Environment
- Developed using **Jupyter Notebook**
- Compatible with **Google Colab** and local execution

---

## Repository Structure
├── anomaly_detection_gmm.ipynb # GMM-based anomaly detection & voting strategy
├── intrusion_detection_classical_ml.ipynb # Classical ML models for intrusion detection
└── README.md
