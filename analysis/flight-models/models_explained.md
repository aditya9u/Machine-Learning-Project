# Flight Price Prediction - Model Explanations

## Why These 3 Models Are Applied

Per the competition rules (`regression_models.md`), only 3 model families are allowed for the **domestic-flight-price-prediction-challenge**:

### 1. Ridge Regression (L2 Regularization) — **BEST LINEAR MODEL**
- **Objective**: `min ||y - Xw||^2 + lambda||w||^2`, solution: `w = (X'X + lambda*I)^-1 X'y`
- **Why chosen**: 
  - Fixes the **singular X'X matrix** problem caused by the one-hot encoding of 40+ dummy columns from categorical features (airline, city, flight_class, etc.)
  - The raw data has extreme multicollinearity (many near-constant one-hot columns), making Ordinary Least Squares (OLS) unstable — it throws `LinAlgError: Singular matrix`
  - Ridge adds `lambda*I` to regularize, guaranteeing a unique solution
  - Per numpy 3-fold CV reference: alpha=1–10 gives best mean RMSE ~18313; larger alphas degrade performance
- **Notebook source**: `flight_price_model.ipynb` §5b-5c, §8a-8c
- **Tuned parameter**: `alpha` (tested: 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0)

### 2. Lasso Regression (L1 Regularization) — **BEST SPARSE LINEAR MODEL**
- **Objective**: `lambda||w||_1`, no closed form, uses coordinate descent
- **Why chosen**:
  - Same stability as Ridge for the singular one-hot matrix problem
  - **Additional benefit**: Automatically zeroes out coefficients of useless features (e.g., rare airlines), performing built-in feature selection over the 40+ one-hot columns
  - Pairs with Ridge for the "variety of models" report requirement
  - Tuned alpha grid: 0.01, 0.1, 1.0, 10.0, 100.0
- **Notebook source**: `flight_price_model.ipynb` §5b, §8b
- **Sparsity report**: Number of nonzero coefficients out of total 40+ one-hot features

### 3. Regression Tree (DecisionTreeRegressor) — **BEST NONLINEAR MODEL**
- **Mechanism**: Leaf nodes contain mean target value; greedy MSE-based splits; supports pre/post pruning
- **Why chosen**:
  - **Only allowed nonlinear regressor** in Part-1 rules
  - Captures **feature interactions** that linear models miss (e.g., `flight_class` × `duration`, `airline` × `days_left`)
  - Robust to the 20% missing values after imputation
  - Tunable via `max_depth` (tested: 4, 6, 8, 12, None) and `min_samples_leaf` (tested: 1, 5, 20)
  - Cost-complexity pruning path (`ccp_alpha`) available as alternative tuning approach
- **Notebook source**: `flight_price_model.ipynb` §5c-5d, §8c-8d
- **Key diagnostic**: Check train/val gap — large gap indicates overfitting; prefer simpler model (Ridge) in that case

## Models Explicitly Removed (Not Best/Allowed)

| Removed Model | Reason |
|---|---|
| Univariate/Multiple OLS | Subsumed by Ridge; singular on one-hot matrix |
| Polynomial/basis expansion | Explodes to 800+ features on 40+ one-hots; severe overfitting |
| Random Forest, Boosting, SVM, NN | NOT taught in Part 1; NOT allowed by competition rules |
| log-target Ridge | Worse RMSE (~19.4k vs ~18.3k on raw target) — confirmed worse |

## How to Run the Models

Each model follows the same workflow:
1. Load train.csv and test.csv
2. Preprocess: median impute `duration`, one-hot encode categoricals, scale numeric features
3. Split 80/20 train/validation (random_state=42)
4. Train model with GridSearchCV over alpha/depth parameters
5. Evaluate using RMSE on validation set
6. Refine winner on full training set, generate `submission.csv`

See the individual Python scripts in this folder for complete, runnable code.