# AER ML Pipeline

This repository contains a small machine-learning pipeline for AER (Air Exchange Rate) prediction from barn ventilation data.

I built this project to organize and test a reproducible workflow: validating the raw data, cleaning columns, fixing incorrect angle features, comparing different feature sets, and training models. The main question I wanted to explore was whether a small set of practical inputs—temperature, inlet angle, and window state—could capture most of the variation in AER.

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

**Input**: `data/raw/aer_raw.csv` (486 samples)

**Target**: `AER` (Air Exchange Rate, continuous, 18.93–197.90)

**Key Features**:
- **Temperature** (temp): -2 to 32°C
- **Air Velocity** (vel): 0.25–3.20 m/s
- **Inlet Angle** (inlet_angle): 0°, 45°, 90°
- **Window State** (window): Open, OpenDown, OpenUp
- **Richardson Number** (richardson_num): Thermal stratification measure
- **Richness Number** (rich_num): Categorical (ORI, URI, ZRI)
- **Length-Width Ratio** (lwratio): 2, 3, or 4
- **Box Configuration** (box_num): 10 or 40
- **Building Type** (old_new): new or old

## Pipeline Overview

```
Raw CSV Data
    ↓
Data Validation (reports issues)
    ↓
Data Cleaning (standardization, type conversion)
    ↓
Feature Engineering (correct trigonometric features)
    ↓
Preprocessing (scaling, encoding)
    ↓
Model Training (3 scenarios × 3 algorithms)
    ↓
Model Evaluation & Visualization
    ↓
Reports & Metrics
```

## Feature Scenarios

### Scenario 1: Full Model
Uses all meaningful features to capture maximum predictive information:
```
temp, vel, inlet_angle_sine, inlet_angle_cosine, window, 
rich_num, richardson_num, lwratio, box_num, old_new
```

### Scenario 2: Practical Minimal Model ⭐
Focuses on the most actionable features—temperature, angle, and window state:
```
temp, inlet_angle_sine, inlet_angle_cosine, window
```
This scenario demonstrates that careful feature selection can achieve near-full-model performance with far fewer inputs, important for real-world deployment and interpretability.

### Scenario 3: Weak Baseline Model
Excludes practical ventilation descriptors (window) to show their importance:
```
temp, inlet_angle_sine, inlet_angle_cosine
```

## Algorithms

1. **Linear Regression**: Establishes interpretable baseline
2. **Random Forest**: Captures non-linearities and interactions
3. **Gradient Boosting**: Sequential ensemble refinement

## Evaluation Metrics

- **R²**: Coefficient of determination (variance explained)
- **MAE**: Mean absolute error (average prediction error)
- **RMSE**: Root mean squared error (penalizes large errors)

## Results

Using a random 80/20 train-test split, the full model with Random Forest achieved the best overall performance:

- **R² = 0.9957** | MAE = 1.34 | RMSE = 2.08

The practical minimal model used only four inputs: `temp`, `inlet_angle_sine`, `inlet_angle_cosine`, and `window`. With Gradient Boosting, it still reached:

- **R² = 0.9884** | MAE = 2.71 | RMSE = 3.41

The weak baseline without `window` dropped to **R² = 0.5307**. This suggests that the window opening state carries important information for AER prediction in this dataset.

**Note**: These results are based on a random split of 98 test samples. I treat these as an initial benchmark rather than final validation.

## How to Run

```bash
pip install -r requirements.txt
python src/pipeline.py
pytest tests/
```

All pipeline stages are orchestrated in `src/pipeline.py`. Output files (cleaned data, metrics, plots, trained model) are saved automatically to `data/processed/`, `reports/`, and `models/`.

## Project Structure

```
aer-ml-pipeline/
├── data/
│   ├── raw/aer_raw.csv
│   └── processed/aer_cleaned.csv
├── src/
│   ├── pipeline.py
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── reports/
│   ├── metrics/
│   └── figures/
├── models/best_model.pkl
├── tests/
├── requirements.txt
└── README.md
```

## Key Decisions & Trade-Offs

1. **Feature Scenarios**: Demonstrates that a carefully chosen minimal feature set can be competitive with the full model, balancing performance vs. interpretability.

2. **Three Algorithms**: Linear Regression for interpretability, Random Forest and Gradient Boosting for capturing non-linearities.

3. **Train-Test Split (80-20)**: Standard practice for regression; enables honest evaluation without data leakage.

4. **Standardization + OneHotEncoding**: Ensures fair feature scaling and proper categorical handling via ColumnTransformer.

5. **Feature Engineering Correctness**: Recalculates trigonometric features using `np.sin(np.deg2rad(...))` to avoid prior errors.

## Reproducibility

- **Fixed Random Seeds**: All splits and model initializations use `random_state=42`
- **Modular Code**: Each stage (validation, cleaning, engineering, training) is a separate module
- **Configuration in Code**: No hard-coded paths or magic numbers; all settings are explicit
- **Automated Reporting**: Metrics, plots, and model are saved automatically
- **Version Control Ready**: Includes .gitignore; ready to push to GitHub

## Potential Extensions

- K-Fold and GroupKFold cross-validation for robust validation
- Hyperparameter tuning (GridSearchCV)
- Feature interactions and polynomial features
- Model explainability (SHAP values)
- REST API deployment
