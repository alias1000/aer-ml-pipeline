"""
Model training module for AER dataset.

Trains three feature scenarios:
1. Full model: all features
2. Practical minimal model: essential features only
3. Weak baseline model: temperature and angle only

Compares Linear Regression, Random Forest, and Gradient Boosting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from pathlib import Path


def prepare_features_and_target(df, feature_list):
    """
    Prepare feature matrix and target vector.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned and feature-engineered data
    feature_list : list
        List of feature column names
    
    Returns:
    --------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target (AER)
    """
    X = df[feature_list].copy()
    y = df['aer'].copy()
    return X, y


def identify_feature_types(X):
    """Identify numerical and categorical features."""
    numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = X.select_dtypes(include=['category', 'object']).columns.tolist()
    return numerical, categorical


def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """Create ColumnTransformer for preprocessing."""
    transformers = []
    
    if numerical_cols:
        transformers.append(
            ('num', StandardScaler(), numerical_cols)
        )
    
    if categorical_cols:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        )
    
    return ColumnTransformer(transformers=transformers)


def train_model(X_train, y_train, X_test, y_test, model_type='linear'):
    """
    Train a single model.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    model_type : str
        'linear', 'rf', or 'gb'
    
    Returns:
    --------
    model : Trained model
    metrics : dict with r2, mae, rmse
    """
    numerical, categorical = identify_feature_types(X_train)
    preprocessor = create_preprocessing_pipeline(numerical, categorical)
    
    if model_type == 'linear':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
    elif model_type == 'rf':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
    elif model_type == 'gb':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse)
    }
    
    return model, metrics, y_pred


def train_models_for_scenario(df, scenario_name, feature_list, models=['linear', 'rf', 'gb']):
    """
    Train multiple models for a single feature scenario.
    
    Returns:
    --------
    results : list of dicts with model metrics
    best_model : tuple (model, model_name, scenario, metrics, predictions)
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    print(f"Features: {feature_list}")
    print(f"Number of features: {len(feature_list)}\n")
    
    X, y = prepare_features_and_target(df, feature_list)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    results = []
    best_model_info = None
    best_r2 = -np.inf
    
    for model_type in models:
        print(f"Training {model_type.upper()}...", end=" ")
        model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test, model_type)
        print(f"[OK]")
        
        result = {
            'scenario': scenario_name,
            'model': model_type,
            'features': len(feature_list),
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse']
        }
        results.append(result)
        
        print(f"  R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_info = (model, model_type, scenario_name, metrics, y_pred, y_test)
    
    return results, best_model_info


def train_all_scenarios(df):
    """
    Train all three feature scenarios.
    
    Returns:
    --------
    all_results : list of all model results
    overall_best : best model across all scenarios
    """
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    # Define scenarios
    scenarios = {
        'Full Model': [
            'temp', 'vel', 'inlet_angle_sine', 'inlet_angle_cosine',
            'window', 'rich_num', 'richardson_num', 'lwratio', 'box_num', 'old_new'
        ],
        'Practical Minimal Model': [
            'temp', 'inlet_angle_sine', 'inlet_angle_cosine', 'window'
        ],
        'Weak Baseline Model': [
            'temp', 'inlet_angle_sine', 'inlet_angle_cosine'
        ]
    }
    
    all_results = []
    best_models = {}
    overall_best = None
    overall_best_r2 = -np.inf
    
    for scenario_name, feature_list in scenarios.items():
        results, best_model_info = train_models_for_scenario(
            df, scenario_name, feature_list
        )
        all_results.extend(results)
        best_models[scenario_name] = best_model_info
        
        if best_model_info[3]['r2'] > overall_best_r2:
            overall_best_r2 = best_model_info[3]['r2']
            overall_best = best_model_info
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for scenario_name, best_model_info in best_models.items():
        model, model_type, _, metrics, _, _ = best_model_info
        print(f"\n{scenario_name}:")
        print(f"  Best model: {model_type.upper()}")
        print(f"  R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL BEST:")
    model, model_type, scenario, metrics, _, _ = overall_best
    print(f"  Scenario: {scenario}")
    print(f"  Model: {model_type.upper()}")
    print(f"  R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    print(f"{'='*80}\n")
    
    return all_results, overall_best


def save_metrics(all_results, csv_path, json_path):
    """Save metrics to CSV and JSON."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df_metrics = pd.DataFrame(all_results)
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV to {csv_path}")
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved metrics JSON to {json_path}")
