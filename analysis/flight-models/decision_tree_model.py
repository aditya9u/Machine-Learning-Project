# Flight Price Prediction - Regression Tree Model

This script trains a DecisionTreeRegressor on the flight price competition data.

## Usage
```bash
python decision_tree_model.py
```

## What This Does
1. Loads train.csv and test.csv from the competition data
2. Performs same preprocessing as the kernel:
   - Median impute `duration` (has ~20% missing)
   - One-hot encode categorical features (airline, source_city, departure_time, stops, arrival_time, destination_city, flight_class)
   - Standard scale numeric features (duration, days_left)
3. Trains DecisionTreeRegressor with GridSearchCV over:
   - `max_depth`: [4, 6, 8, 12, None]
   - `min_samples_leaf`: [1, 5, 20]
4. Uses 5-fold CV with neg_root_mean_squared_error scoring
5. Reports validation RMSE for best model
6. Optionally outputs the pruning path (ccp_alpha) as secondary analysis

## Key Expected Outputs
- **Best max_depth**: Typically 6 or 8 (U-shaped RMSE curve — falls then rises)
- **Best min_samples_leaf**: Typically 5 or 20 (prevents leaf overfitting)
- **Validation RMSE**: Should be compared against Ridge (~18313) and Lasso to determine winner
- **Overfitting diagnostic**: Check train/val gap — if tree has much lower train RMSE than val, it's overfitting and Ridge may generalize better on private test set

## Post-Analysis
After running, compare the val RMSE against:
- Ridge best RMSE (from ridge_lasso_comparison.py)
- Lasso best RMSE (from ridge_lasso_comparison.py)
- The competition winner is chosen based on which has lowest val RMSE AND smallest train/val gap