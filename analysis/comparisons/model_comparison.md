# Cross-Competition Model Comparison

This document summarizes which models are best for each Kaggle competition and why.

## Competition 1: domestic-flight-price-prediction-challenge (Regression)

| Model | Type | Key Strength | Expected Performance |
|---|---|---|---|
| **Ridge Regression** | Linear (L2) | Fixes singular one-hot matrix; stable with collinear features | RMSE ~18313 (reference); alpha 1-10 optimal |
| **Lasso Regression** | Linear (L1) | Same stability as Ridge + feature selection (zeros out weak features) | Similar RMSE to Ridge; fewer nonzero coefs |
| **Regression Tree** | Nonlinear | Captures feature interactions (e.g., class × duration); only allowed nonlinear model | Typically higher RMSE than linear models if overfitting; prefer Ridge if large train/val gap |

**Winner**: Ridge Regression is the preferred model. It achieves the best reference RMSE (~18313) and is most robust. Lasso is a close second with added feature selection. Tree is third unless it has minimal train/val gap.

**Rule**: Only these 3 models allowed; ensembles, polynomial, SVM, NN are banned.

---

## Competition 2: vehicle-type-classification (Classification)

| Model | Type | Key Strength | Expected Performance |
|---|---|---|---|
| **LDA** | Linear discriminant | Shared covariance matrix; fewest params; handles 0.95 correlations naturally | Accuracy ~0.8928 (reference best) |
| **Logistic/Softmax (L2)** | Discriminative regression | Discriminative boundary; L2 regularization resists correlated features; matches LDA at C=100 | Accuracy ~0.8922 (reference); improves with tuning to C=100/1000 |

**Winner**: **Tied** — both LDA and Logistic achieve nearly identical accuracy (~0.892+). 
- Choose **LDA** if you want no hyperparameter tuning and maximum stability
- Choose **Logistic** if you want to tune C and potentially match/beat LDA with proper tuning

**Rule**: Only these 2 models allowed; Naive Bayes, QDA, KNN, Trees, SVM, NN are banned.

---

## Summary Comparison Across Competitions

| Aspect | Flight Price (Regression) | Vehicle Type (Classification) |
|---|---|---|
| **Number of allowed models** | 3 | 2 |
| **Best model type** | Ridge (L2 linear) | LDA & Logistic (tied linear) |
| **Key challenge** | Singular one-hot matrix, 20% missing, skewed target | Extreme multicollinearity (0.95 correlations), balanced classes, duplicate rows |
| **Regularization need** | Critical (Ridge/Lasso fix singularity) | Critical (L2 handles 0.95 correlations) |
| **Tuning required** | Yes (alpha for Ridge/Lasso; depth for Tree) | Yes (C for Logistic; optional shrinkage for LDA) |
| **Theory basis** | MLE, penalty methods | Bayes discriminant functions, maximum entropy |

## How to Use This Repository

1. **Clone to GitHub**: `git init`, add files, commit, push
2. **Run individual models**: Each Python script in `flight-models/` and `vehicle-models/` is self-contained
3. **Review model explanations**: `models_explained.md` and `model_details.md` contain the "why" behind each choice
4. **Compare results**: Use the comparison tables above to decide which model to submit for each competition
5. **Generate submissions**: Follow each notebook's Sec.7-12 workflow to refit winner on full train and create `submission.csv`