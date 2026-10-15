# Kaggle Competitions - ML Project Analysis

This repository contains two Kaggle competition kernels and their associated analysis from a Machine Learning course (Part-1 only).

## Folder Structure

```
Kaggle-Competitions/
├── flight_price_model.ipynb     # Domestic flight price prediction notebook
├── vehicle_type_model.ipynb     # Vehicle type classification notebook
├── train.csv                    # Flight training data (15589x12)
├── test.csv                     # Flight test data (3898x11)
├── sample_submission.csv        # Flight submission format
├── vehicle_train_10000.csv      # Vehicle training data (10000 rows)
├── vehicle_test.csv             # Vehicle test data
├── vehicle_sample_submission.csv # Vehicle submission format
├── domestic-flight-price-prediction-challenge.zip # zipped competition data
└── vehicle-type-classification.zip      # zipped competition data
└── analysis/                    # Model explanation and code analysis
   ├── README.md
   ├── flight-models/
   │   ├── models_explained.md
   │   ├── ridge_lasso_comparison.py
   │   └── decision_tree_model.py
   ├── vehicle-models/
   │   ├── model_details.md
   │   └── lda_logistic_comparison.py
   └── comparisons/
       └── model_comparison.md
```

## Competitions Included

### 1. domestic-flight-price-prediction-challenge
- **Type**: Regression - predict flight price in INR
- **Target**: `price`
- **Training data**: 15,589 rows, 12 columns
- **Test data**: 3,898 rows, 11 columns
- **Allowed models ONLY (per competition rules)**:
  - **Ridge Regression (L2)** - Best linear model, fixes singular one-hot matrix
  - **Lasso Regression (L1)** - Best sparse linear model, performs feature selection
  - **Regression Tree (DecisionTreeRegressor)** - Best nonlinear model, captures interactions
- **Banned models**: OLS, Polynomial, SVM, NN, Random Forest, Boosting, Ensembles

### 2. vehicle-type-classification
- **Type**: Classification - predict vehicle class (0=Sedan, 1=SUV, 2=Sports Car)
- **Target**: `vehicle_class`
- **Training data**: 10,000 rows, 21 columns (20 features + target)
- **Test data**: 2,500 rows, 20 columns
- **Allowed models ONLY (per competition rules)**:
  - **LDA** (Linear Discriminant Analysis) - Tied best at ~0.8928 accuracy
  - **Logistic/Softmax Regression with L2** - Tied best at ~0.8922 accuracy
- **Banned models**: Naive Bayes, QDA, KNN, Trees, SVM, NN, Ensembles

## Model Explanations

Each competition has strict model restrictions for Part-1:

### Flight Price - Why These 3 Models?
1. **Ridge (L2)**: Fixes the singular X'X matrix problem from 40+ one-hot encoded categorical features. Regularization parameter `lambda` ensures stable solution. Reference RMSE ~18313 with alpha=1-10.
2. **Lasso (L1)**: Same stability as Ridge + automatic feature selection by zeroing useless coefficients (e.g., rare airlines). Pairs with Ridge for variety requirement.
3. **Regression Tree**: Only allowed nonlinear model. Captures feature interactions (e.g., flight_class × duration). Tunable via max_depth and min_samples_leaf.

### Vehicle Type - Why These 2 Models? (Tied Best)
1. **LDA**: Gaussian assumption with shared covariance matrix. Fewest parameters (one 20x20 matrix). Naturally handles 0.95-correlated features (horsepower↔torque 0.959, displacement↔torque 0.953). Reference accuracy 0.8928.
2. **Logistic L2**: Discriminative model with L2 regularization. Handles correlated features and resists noisy hidden test. Matches LDA at C=100 (~0.892 accuracy). Tunable via C parameter.

## How to Use

1. **Clone to GitHub**: Initialize git repo and push
2. **Run notebooks**: Open `.ipynb` files in Jupyter/Lab or VS Code
3. **Review model explanations**: See `analysis/flight-models/models_explained.md` and `analysis/vehicle-models/model_details.md` for theoretical justifications
4. **Run Python scripts**: Use `analysis/flight-models/ridge_lasso_comparison.py` etc. for quick model runs
5. **Generate submissions**: Follow notebook Sec.7-12 workflow to refit winner and create `submission.csv`

## Data Files

- **train.csv**: 15589 rows × 12 columns (flight price competition)
- **test.csv**: 3898 rows × 11 columns (flight price competition, no `price` column)
- **vehicle_train_10000.csv**: 10000 rows × 21 columns (vehicle competition)
- **vehicle_test.csv**: 2500 rows × 20 columns (vehicle competition, no target)

## Competition Rules Summary

**Flight Price**: Only Ridge, Lasso, and Regression Tree allowed. Ensembles, polynomial features, SVM, NN explicitly banned.

**Vehicle Type**: Only LDA and Logistic L2 allowed. Naive Bayes, QDA, KNN, Trees, SVM, NN explicitly banned.

Both competitions emphasize regularized linear models due to the data characteristics: multicollinearity, missing values, and the need for generalization to hidden test sets.