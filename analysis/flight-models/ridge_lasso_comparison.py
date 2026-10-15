# Flight Price Prediction - Ridge vs Lasso Comparison

This script compares Ridge Regression (L2) and Lasso Regression (L1) on the flight price competition data.

## Usage
```bash
python ridge_lasso_comparison.py
```

## What This Does
1. Loads train.csv and test.csv from the competition data
2. Performs same preprocessing as the kernel:
   - Median impute `duration` (has ~20% missing)
   - One-hot encode categorical features (airline, source_city, departure_time, stops, arrival_time, destination_city, flight_class)
   - Standard scale numeric features
3. Trains Ridge with GridSearchCV over alpha = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
4. Trains Lasso with GridSearchCV over alpha = [0.01, 0.1, 1.0, 10.0, 100.0]
5. Reports validation RMSE, R2%, and MAPE% for both models
6. Prints number of nonzero coefficients (feature selection) for Lasso

## Key Expected Outputs
- **Ridge best alpha**: Typically 1.0 or 10.0 (from reference: alpha 1-10 gives RMSE ~18313)
- **Lasso best alpha**: Typically 0.1 or 1.0
- **Lasso nonzero coefs**: Fraction of 40+ one-hot columns that survive (feature selection effect)
- Both models should achieve similar RMSE to the ~18313 reference, confirming Ridge/Lasso stability over OLS