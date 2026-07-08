# AER ML Pipeline

This repository contains a small machine-learning pipeline for predicting Air Exchange Rate (AER) from barn ventilation data.

I built this project to make the workflow easier to rerun and inspect: checking the raw CSV file, cleaning columns, rebuilding angle features, comparing feature sets, training regression models, and saving the results.

The dataset is related to my research context on naturally ventilated livestock buildings. The main comparison is between a full feature set and a smaller practical feature set using only temperature, inlet angle, and window opening state.

## What the pipeline does

- checks the raw CSV for missing values, duplicate rows, constant columns, and invalid categories
- removes constant or redundant columns (`BOX`, `box_ver`, and the old `inlet_angle_sine`)
- recalculates angle features correctly using degrees:
  - `sin(deg2rad(inlet_angle))`
  - `cos(deg2rad(inlet_angle))`
- compares three feature scenarios to test the feature trade-off
- trains Linear Regression, Random Forest, and Gradient Boosting models on each scenario
- saves metrics, plots, and the best model

## Dataset

486 samples from barn ventilation experiments. Features include temperature, air velocity, inlet angle, window state, thermal and geometric parameters. Target: AER (continuous, 18.93–197.90).

## Key Decisions & Trade-Offs

- Feature scenarios test whether a minimal set (4 inputs) can match the full model (10 inputs)
- Three algorithms: Linear Regression (baseline), Random Forest and Gradient Boosting (ensemble)
- Fixed random seeds ensure reproducibility
- Trigonometric features recalculated correctly: `sin(deg2rad(inlet_angle))`, `cos(deg2rad(inlet_angle))`

## Reproducibility

- Modular code: validation → cleaning → engineering → training → evaluation
- Fixed random seeds (42) for all splits and model initialization
- Automated metrics and figure generation
- Ready for version control

## Potential Extensions

- K-Fold and GroupKFold cross-validation
- Hyperparameter tuning
- Feature interactions
- Model explainability (SHAP)

## How to Run

```bash
pip install -r requirements.txt
python src/pipeline.py
pytest tests/
```

## Results

Random 80/20 split, best model per scenario:

- Full Model (Random Forest): R² = 0.9957, MAE = 1.34, RMSE = 2.08
- Practical Minimal (Gradient Boosting): R² = 0.9884, MAE = 2.71, RMSE = 3.41
- Weak Baseline (no window): R² = 0.5307

## Notes

The current results are based on a random 80/20 train-test split. I treat them as an initial benchmark. A stronger version of the project would add KFold and GroupKFold validation.
