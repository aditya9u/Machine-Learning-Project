# Vehicle Type Classification (VehicleVision) - Submission Guidelines

Competition: vehicle-type-classification
URL: https://www.kaggle.com/competitions/vehicle-type-classification
Deadline: 2026-10-15 18:30:00

## Files Downloaded
- vehicle_train_10000.csv (~10000 rows, 20 numeric features + id + target)
- vehicle_test.csv (~2500 rows, same features without target)
- vehicle_sample_submission.csv (2501 x 2 incl header)
- Full zip retained: vehicle-type-classification.zip

Note: Competition docs refer to vehicle_train.csv / vehicle_public_test.csv,
actual downloadable files are named vehicle_train_10000.csv / vehicle_test.csv.

Features (20 numeric):
engine_displacement_l, horsepower_hp, vehicle_weight_kg, wheelbase_mm,
vehicle_length_mm, vehicle_width_mm, vehicle_height_mm, fuel_tank_capacity_l,
city_fuel_efficiency_kmpl, highway_fuel_efficiency_kmpl, engine_cylinders,
torque_nm, ground_clearance_mm, turning_radius_m, cargo_capacity_l,
acceleration_0_100_kmh_s, braking_distance_m, tire_width_mm,
drag_coefficient, battery_capacity_kwh

ID column: vehicle_id (do NOT use as predictive feature)
Target: vehicle_class (0=Sedan, 1=SUV, 2=Sports Car)

## Submission Format (from page: "rules" / abstract)
Your submission should contain:
```
vehicle_id,vehicle_class
```

For every vehicle in the test set, provide one predicted class (0, 1 or 2).

Example from vehicle_sample_submission.csv:
```
vehicle_id,vehicle_class
10001,0
10002,0
...
```

## Evaluation (from rules page)
Submissions are evaluated using classification accuracy.
Accuracy = fraction of predictions that exactly match hidden vehicle classes.

## Important Notes from Host
- Training data may contain data-quality issues such as duplicate vehicle measurements.
- vehicle_id identifies a record, should not be treated as predictive characteristic.
- Hidden evaluation data introduces measurement uncertainty / random measurement error
  (different measurement environment). Model that is best on train is not necessarily
  best on final evaluation -> prefer robust/generalizable models.
- This is a synthetic educational dataset, not real vehicles.
- Perform EDA + appropriate preprocessing before training.

## How to Submit (for YOU to do manually)
1. Go to competition page -> Submit Predictions
2. Upload your CSV (vehicle_id,vehicle_class)
3. No submission has been made by the assistant.

---
Full page contents fetched via Kaggle API `competitions pages list --content`.
Pages available: data-description (=NA), rules, abstract, foundational-rules.
Foundational rules also apply (see Kaggle site).
