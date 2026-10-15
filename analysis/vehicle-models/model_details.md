# Vehicle Type Classification - Model Explanations

## Why These 2 Models Are Applied

Per the competition rules (`classification_models.md`), only 2 model families are allowed for the **vehicle-type-classification** competition:

### 1. LDA (Linear Discriminant Analysis) — **TIED BEST at ~0.8928 accuracy**
- **Theory**: Assumes `x|y=k ~ N(mu_k, Sigma)` (Gaussian per class, shared covariance Sigma across all classes). 
  Parameters learned by MLE: priors, class means, and pooled covariance.
- **Why chosen**:
  - **Fewest parameters**: Only one 20x20 pooled covariance matrix (vs. 3 separate matrices for QDA), making it stable
  - **Handles multicollinearity**: The pooled covariance naturally deals with the extreme feature correlations (mean |corr| = 0.46, top pairs at 0.95: horsepower↔torque 0.959, displacement↔torque 0.953, wheelbase↔width 0.952)
  - **Resists measurement noise**: Host warns hidden test has extra measurement noise → simple regularized models generalize best
  - **Top reference score**: numpy-only stratified 3-fold CV achieves LDA 0.8928 > Logistic(C=100) 0.8922
  - **No tuning needed** (optional shrinkage variant available), making it practical for Part-1
- **Shrinkage variant**: `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` — stabilizes covariance estimate; expected to tie or slightly beat plain LDA
- **Notebook source**: `vehicle_type_model.ipynb` §5a, §5b, §7-8, §12

### 2. Logistic / Softmax Regression with L2 — **TIED BEST at ~0.8922 accuracy**
- **Theory**: `P(y=k|x) = exp(w_k'x) / sum_j(exp(w_j'x))`, optimized via categorical cross-entropy with L2 regularization. 
  Tunable parameter: `C` (inverse regularization strength).
- **Why chosen**:
  - **Discriminative model**: No `P(x|y)` assumption like LDA; directly optimizes the decision boundary
  - **L2 regularization handles correlations**: The L2 penalty (`||w||^2`) resists the 0.95-correlated features and protects against noisy hidden test set
  - **Matches LDA accuracy**: With C=100, Logistic matches LDA's ~0.892 accuracy
  - **Improves with tuning**: Grid over C = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] can potentially exceed LDA
  - **Robust**: L2 penalty prevents coefficient explosion from correlated features
- **Notebook source**: `vehicle_type_model.ipynb` §5b, §5c-5d, §7-8, §12

## Models Explicitly Removed (Not Best/Allowed)

| Removed Model | Accuracy (ref) | Reason |
|---|---|---|
| QDA | 0.8877 | 3 per-class 20x20 covariances overfit noise/duplicates; loses to LDA |
| Gaussian Naive Bayes | 0.8443 | Independence assumption violated by 0.95 correlations (horsepower/torque/displacement triple-count evidence) |
| KNN (k=5) | 0.8752 | Noise + duplicates + 20-dim distances hurt; needs perfect scaling |
| KNN (k=15) | 0.8816 | Same issues as KNN5, slightly better but still trails linear models |
| Decision Tree | N/A | Axis-aligned thresholds brittle under measurement noise; unstable on duplicated rows |
| Random Forest, Boosting, SVM, NN | NOT taught in Part 1; NOT allowed by competition rules |

## How to Run the Models

Each model follows the same workflow:
1. Load vehicle_train_10000.csv and vehicle_test.csv
2. Preprocess: median impute + StandardScaler on 20 numeric features; drop `vehicle_id`
3. Optional: drop one feature from each >0.9 correlated pair (experiment in Sec.9)
4. Split stratified 80/20 train/validation (preserves 3-class ratios, random_state=42)
5. Train model:
   - **LDA**: No hyperparameter tuning needed (optional: shrinkage='auto')
   - **Logistic**: GridSearchCV over C = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] with multinomial + lbfgs
6. Evaluate using accuracy (competition metric) on validation set
7. Refine winner on full training set, generate `submission.csv` with predictions {0, 1, 2}

See the individual Python scripts in this folder for complete, runnable code.