# Vehicle Type Classification - LDA vs Logistic Comparison

This script compares LDA and L2 Logistic/Softmax Regression on the vehicle type competition data.

## Usage
```bash
python lda_logistic_comparison.py
```

## What This Does
1. Loads vehicle_train_10000.csv and vehicle_test.csv from the competition data
2. Performs same preprocessing as the kernel:
   - All 20 numeric features
   - Median impute + StandardScaler pipeline
   - Drop `vehicle_id` identifier
3. Trains LDA (plain and shrinkage variants) and Logistic Regression with grid over C
4. Reports validation accuracy for all models
5. Prints per-class accuracy, confusion matrix insights, and feature importance

## Expected Workflow
1. **Sec.5a**: Compare LDA (plain) vs LDA (shrinkage=auto) — expected tied or shrinkage slightly better
2. **Sec.5b**: Train Logistic with C grid — expected peak at C=100 (~0.892), tied with LDA
3. **Sec.6**: Generate comparison table with accuracy for LDA, LDA-shrink, Logistic
4. **Sec.12**: 5-fold CV on full train with mean +- std for each model
5. **Submit**: Refit best model on full train, predict test set, upload submission.csv manually

## Key Expected Outputs
- **LDA val accuracy**: ~0.892–0.893 (depends on shrinkage)
- **LDA-shrink val accuracy**: ~0.8928 (reference) or tied with plain LDA
- **Logistic best C**: Typically 100.0 (matches LDA ~0.892); may reach 1000.0 if curve keeps climbing
- **Ranking**: LDA and Logistic should be very close (~0.892+); the "tied best" status means either can be chosen
- **Decision factor**: Choose LDA for simplicity (no tuning) or Logistic if you want C tuning flexibility