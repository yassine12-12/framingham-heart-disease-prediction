# Framingham Heart Disease Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning pipeline for predicting 10-year coronary heart disease (CHD) risk using the Framingham Heart Study dataset. This project implements multiple classification algorithms with advanced preprocessing, class imbalance handling, and fairness evaluation.

## 🎯 Project Overview

This project develops and evaluates machine learning models to predict the 10-year risk of coronary heart disease based on patient demographics, lifestyle factors, and clinical measurements. The analysis focuses on achieving high recall (sensitivity) to minimize false negatives in medical predictions.

### Key Features

- **Comprehensive EDA**: Missing data analysis, correlation studies, and outlier detection
- **Advanced Preprocessing**: Median imputation, winsorization, and robust scaling
- **Class Imbalance Handling**: SMOTEENN oversampling technique
- **Multiple ML Algorithms**: Logistic Regression, XGBoost, and SVM
- **Threshold Optimization**: Maximizing recall for medical diagnosis
- **Fairness Audit**: Gender-based bias evaluation
- **Robust Evaluation**: 5-fold stratified cross-validation

## 📊 Dataset

**Source**: [Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)

- **Total Samples**: 4,240 patients
- **Features**: 15 predictors (after preprocessing)
- **Target**: Binary classification (10-year CHD risk)
- **Class Distribution**: ~15% positive cases (imbalanced)

### Features Include:
- **Demographics**: Age, gender
- **Lifestyle**: Smoking status, cigarettes per day
- **Clinical Measurements**: Blood pressure, cholesterol, glucose
- **Medical History**: Hypertension, diabetes, stroke history

## 🛠️ Methodology

### 1. Data Preprocessing
```
📋 Raw Data (4,240 × 16)
    ↓
🗑️  Feature Removal (education column)
    ↓
📈 Exploratory Data Analysis
    ↓
🔧 Preprocessing Pipeline:
    • Median imputation for missing values
    • Winsorization (1%-99% percentiles)
    • Robust scaling
    ↓
⚖️  SMOTEENN Oversampling
    ↓
🤖 Model Training & Evaluation
```

### 2. Models Evaluated
- **Logistic Regression**: Linear baseline with L1/L2 regularization
- **XGBoost**: Gradient boosting with tree-based learning
- **Support Vector Machine**: Non-linear classification with RBF/linear kernels

### 3. Hyperparameter Tuning
- **Grid Search**: Exhaustive parameter optimization
- **Cross-Validation**: 5-fold stratified validation
- **Scoring Metric**: ROC-AUC for model selection

### 4. Threshold Optimization
- **Medical Focus**: Prioritizing high recall (sensitivity)
- **Optimal Threshold**: 0.10 (vs. default 0.50)
- **Trade-off Analysis**: Precision vs. recall evaluation

## 📈 Results

### Model Performance (Optimized Thresholds)

| Model | AUC | Accuracy | Precision | Recall | F1-Score |
|-------|-----|----------|-----------|--------|----------|
| **SVM** | **0.698** | **0.671** | **0.265** | **1.000** | **0.420** |
| XGBoost | 0.672 | 0.663 | 0.259 | 1.000 | 0.412 |
| Logistic Regression | 0.669 | 0.659 | 0.256 | 1.000 | 0.408 |

### Key Findings

✅ **Best Model**: SVM achieves the highest overall performance  
✅ **Perfect Recall**: All models achieve 100% recall after threshold tuning  
✅ **Medical Priority**: Successfully minimizes false negatives  
✅ **Fairness**: No significant gender bias detected  
⚠️ **Trade-off**: High recall comes with lower precision (more false positives)

## 🏥 Clinical Implications

### Threshold Selection (0.10 vs 0.50)
- **Conservative Approach**: 10% probability threshold for CHD prediction
- **Medical Benefit**: Captures all potential CHD cases
- **Clinical Impact**: Early intervention for more patients
- **Cost Consideration**: Increased follow-up testing for false positives

### Model Deployment Considerations
- **High Sensitivity**: Excellent for screening applications
- **Risk Stratification**: Helps prioritize patient care
- **Clinical Integration**: Supports physician decision-making
- **Population Health**: Suitable for preventive care programs

## 🔍 Fairness Analysis

The project includes a comprehensive fairness audit examining gender-based disparities in model predictions:

- **Male Recall**: Consistently high across all models
- **Female Recall**: Comparable performance, no significant bias
- **Conclusion**: Models demonstrate fair performance across gender groups


## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/framingham-heart-disease-prediction.git
cd framingham-heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the analysis
jupyter notebook framingham_pipeline.ipynb
```

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
missingno>=0.5.0
scipy>=1.7.0
```

## 📊 Visualizations

The project generates comprehensive visualizations including:

- **Data Quality**: Missing data patterns and distributions
- **Feature Analysis**: Correlation heatmaps and outlier detection  
- **Model Performance**: ROC curves, precision-recall curves
- **Evaluation Metrics**: Confusion matrices and threshold analysis
- **Fairness Assessment**: Gender-based performance comparison


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Framingham Heart Study**: For providing the foundational dataset
- **Kaggle Community**: For dataset accessibility and documentation
- **Scikit-learn**: For comprehensive machine learning tools
- **Medical Research Community**: For establishing CHD risk factors
