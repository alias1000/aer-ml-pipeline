# AER ML Pipeline

This repository contains a small machine-learning pipeline for predicting Air Exchange Rate (AER) from barn ventilation data.

I built it to make the workflow easier to rerun and inspect. The pipeline checks the raw CSV file, cleans the columns, rebuilds angle-based features, compares different feature sets, trains regression models, and saves the outputs.

The dataset is related to my research context on naturally ventilated livestock buildings. The main comparison is between a full feature set and a smaller practical feature set using only temperature, inlet angle, and window opening state.

## What the pipeline does

- checks missing values, duplicate rows, constant columns, and invalid categories
- removes constant or redundant columns such as `BOX`, `box_ver`, and the old `inlet_angle_sine`
- recalculates angle features using:
  - `sin(deg2rad(inlet_angle))`
  - `cos(deg2rad(inlet_angle))`
- compares three feature scenarios
- trains Linear Regression, Random Forest, and Gradient Boosting models
- saves metrics, plots, and the best model

## Feature scenarios

### Full model

Uses the available ventilation and geometry-related variables:

`temp`, `vel`, `inlet_angle_sine`, `inlet_angle_cosine`, `window`, `rich_num`, `richardson_num`, `lwratio`, `box_num`, `old_new`

### Practical minimal model

Uses only four practical inputs:

`temp`, `inlet_angle_sine`, `inlet_angle_cosine`, `window`

### Weak baseline

Uses the same variables as the minimal model, but without `window`:

`temp`, `inlet_angle_sine`, `inlet_angle_cosine`

## Results

Using a random 80/20 train-test split, the full model with Random Forest gave the best overall result:

- R² = 0.9957
- MAE = 1.34
- RMSE = 2.08

The practical minimal model with Gradient Boosting reached:

- R² = 0.9884
- MAE = 2.71
- RMSE = 3.41

The weak baseline without `window` dropped to R² = 0.5307. This suggests that window opening state carries important information for AER prediction in this dataset.

These results are an initial benchmark based on a random split. A stronger validation step would be to add KFold and GroupKFold cross-validation.

## How to run

```bash
pip install -r requirements.txt
cd src
python pipeline.py
pytest tests/
```

All outputs are saved to `data/processed/`, `reports/`, and `models/`.

## Project structure

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
