# ML Project Analysis - Kaggle Competitions

This repository contains complete analysis, model explanations, and code for two Kaggle competitions from a Machine Learning course (Part-1 only).

## Competitions Covered

1. **domestic-flight-price-prediction-challenge** - Regression: predict flight price in INR
2. **vehicle-type-classification** - Classification: predict vehicle class (Sedan/SUV/Sports Car)

## Allowed Models (per competition rules)

### Flight Price Competition
- **Ridge Regression (L2)** - Best linear model, fixes singular one-hot matrix from collinear features
- **Lasso Regression (L1)** - Best sparse linear model, performs feature selection by zeroing useless coefficients
- **Regression Tree (DecisionTreeRegressor)** - Best nonlinear model, captures feature interactions

### Vehicle Type Classification
- **LDA** (Linear Discriminant Analysis) - Tied best at ~0.8928 accuracy, uses shared covariance
- **Logistic/Softmax Regression with L2** - Tied best at ~0.8922 accuracy, discriminative model with regularization

> **Note**: Ensembles (RF, XGBoost, etc.), polynomial features, SVM, Neural Networks, and other models are **explicitly banned** by the competition rules for Part-1.

## Repository Structure

```
ML-Project-Analysis/
├── README.md                          # This file
├── flight-models/                     # Flight price prediction models
│   ├── ridge_lasso_comparison.py      # Ridge vs Lasso comparison code
│   ├── decision_tree_model.py         # Regression tree model code
│   └── models_explained.md           # Why these models are chosen
├── vehicle-models/                    # Vehicle type classification models
│   ├──lda_logistic_comparison.py      # LDA vs Logistic comparison code
│   ├── model_details.md               # Why these models are chosen
├── comparisons/                       # Cross-competition comparison
    └── model_comparison.md           # Summary of which model is better per competition
```

## How to Use

1. Clone this repository to upload to GitHub
2. Each model notebook/code is self-contained with explanations
3. Run the Python scripts to reproduce results or adapt for your own data
4. Refer to `models_explained.md` and `model_details.md` for theoretical justifications