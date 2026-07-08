# AER ML Data Pipeline

## Project Objective

This project demonstrates a **reproducible data pipeline for scientific regression modeling**, using Air Exchange Rate (AER) prediction as the applied engineering case study.

The goal is not only to train a machine-learning model, but to showcase the engineering practices, data workflows, and reproducibility standards essential for professional data science and ML work.

## Context

The dataset contains measurements from barn ventilation experiments / CFD simulations, with environmental and structural parameters. The task is to predict Air Exchange Rate from ventilation characteristics, inlet geometry, and thermal conditions.

## Skills Demonstrated

This project illustrates proficiency in:

- **Data Validation**: Comprehensive checks for data quality, missing values, duplicates, and categorical validity
- **Data Cleaning**: Standardization, type conversion, removal of invalid/constant/redundant features
- **Feature Engineering**: Correct recalculation of derived features (trigonometric transformations)
- **Machine-Learning Pipeline Design**: Modular, reproducible workflow using scikit-learn
- **Model Comparison**: Multiple algorithms and feature scenarios for performance analysis
- **Evaluation & Visualization**: Residual plots, predictions vs actual, scenario/model comparisons
- **Automated Reporting**: Metrics export, figure generation, model persistence
- **GitHub-Ready Project Structure**: Professional organization with documentation, tests, and version control

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

## Results (Random 80/20 Train-Test Split)

### Model Performance by Scenario

| Scenario | Best Model | R² | MAE | RMSE | Observations |
|----------|-----------|-----|-------|--------|------------|
| **Full Model** (10 features) | Random Forest | 0.9957 | 1.34 | 2.08 | Highest accuracy; includes all features |
| **Practical Minimal** (4 features) | Gradient Boosting | 0.9884 | 2.71 | 3.41 | **Comparable performance** with only 4 features |
| **Weak Baseline** (3 features) | Gradient Boosting | 0.5307 | 16.36 | 21.71 | Without window information, performance drops significantly |

### Key Findings

Although the full model achieved the highest accuracy, **the practical minimal model performed nearly as well using only temperature, inlet angle, and window opening state**. This suggests that a small set of practical ventilation descriptors can capture most of the predictive information for AER in this dataset.

The dramatic performance drop in the weak baseline model (R² = 0.53 without window state) highlights the **critical importance of window opening state** for predicting AER, validating the real-world significance of this feature.

**Note**: These results are based on a random 80/20 train-test split (test set: 98 samples). For production models, cross-validation and hold-out validation on truly independent data would be recommended.

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execute Full Pipeline
```bash
cd src
python pipeline.py
```

This single command runs:
1. Data validation and reporting
2. Data cleaning and preprocessing
3. Feature engineering
4. Training of all 3 scenarios × 3 models
5. Model evaluation and figure generation
6. Metrics and model persistence

### Expected Output

```
Pipeline results saved to:
- data/processed/aer_cleaned.csv       (cleaned dataset)
- reports/metrics/validation_report.json (data quality report)
- reports/metrics/model_metrics.csv     (all model results)
- reports/metrics/model_metrics.json    (metrics as JSON)
- models/best_model.pkl                 (best trained model)
- reports/figures/                      (visualizations)
```

### Run Tests
```bash
pytest tests/
```

## Project Structure

```
aer-ml-data-pipeline/
│
├── data/
│   ├── raw/
│   │   └── aer_raw.csv                 (input: raw dataset)
│   └── processed/
│       └── aer_cleaned.csv             (output: cleaned dataset)
│
├── src/
│   ├── data_validation.py              (validates raw data)
│   ├── data_cleaning.py                (cleans and standardizes)
│   ├── feature_engineering.py          (creates derived features)
│   ├── train_model.py                  (trains 3 scenarios, 3 models each)
│   ├── evaluate_model.py               (generates plots and metrics)
│   └── pipeline.py                     (orchestrates full workflow)
│
├── reports/
│   ├── metrics/
│   │   ├── validation_report.json      (data quality checks)
│   │   ├── model_metrics.csv           (model performance table)
│   │   └── model_metrics.json          (metrics as JSON)
│   └── figures/
│       ├── predicted_vs_actual.png     (actual vs predicted scatter)
│       ├── residual_plot.png           (residuals analysis)
│       ├── scenario_comparison.png     (metrics by scenario)
│       ├── model_comparison.png        (metrics by algorithm)
│       └── feature_importance.png      (top features, if available)
│
├── models/
│   └── best_model.pkl                  (serialized best model)
│
├── tests/
│   └── test_data_cleaning.py           (unit tests)
│
├── notebooks/
│   └── 01_exploration.ipynb            (optional: exploratory analysis)
│
├── requirements.txt                     (Python dependencies)
├── .gitignore                          (Git exclusions)
├── README.md                           (this file)
└── LICENSE                             (optional)
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

- **Cross-Validation**: K-Fold and GroupKFold for robust performance estimation
- **Hyperparameter Tuning**: GridSearchCV, RandomizedSearchCV for model optimization
- **Feature Interactions**: Polynomial features, interaction terms (e.g., window × inlet_angle)
- **Explainability**: SHAP values, LIME for model interpretation
- **Time-Series Analysis**: If temporal structure exists in the data
- **Model Deployment**: Flask, FastAPI for REST API
- **Automated ML**: AutoML frameworks (AutoSklearn, H2O AutoML)

## Author

Built with attention to professional standards and best practices in data science and ML engineering.

---

**For questions or improvements, refer to the modular structure and comments in each source file.**
