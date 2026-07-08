"""
Main pipeline orchestration script.

Runs the complete workflow:
1. Data validation
2. Data cleaning
3. Feature engineering
4. Model training
5. Model evaluation
6. Metrics and figure saving
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

from data_validation import validate_data
from data_cleaning import clean_data
from feature_engineering import create_features
from train_model import train_all_scenarios, save_metrics
from evaluate_model import generate_evaluation_plots


def run_pipeline():
    """Execute the complete data pipeline."""
    print("\n" + "="*80)
    print("AER ML DATA PIPELINE")
    print("="*80 + "\n")
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    raw_data_path = base_dir / 'data' / 'raw' / 'aer_raw.csv'
    cleaned_data_path = base_dir / 'data' / 'processed' / 'aer_cleaned.csv'
    validation_report_path = base_dir / 'reports' / 'metrics' / 'validation_report.json'
    metrics_csv_path = base_dir / 'reports' / 'metrics' / 'model_metrics.csv'
    metrics_json_path = base_dir / 'reports' / 'metrics' / 'model_metrics.json'
    best_model_path = base_dir / 'models' / 'best_model.pkl'
    figures_dir = base_dir / 'reports' / 'figures'
    
    # === STEP 1: DATA VALIDATION ===
    print("STEP 1: DATA VALIDATION")
    print("-" * 80)
    validate_data(str(raw_data_path), str(validation_report_path))
    
    # === STEP 2: DATA CLEANING ===
    print("\nSTEP 2: DATA CLEANING")
    print("-" * 80)
    df_cleaned = clean_data(str(raw_data_path), str(cleaned_data_path))
    
    # === STEP 3: FEATURE ENGINEERING ===
    print("\nSTEP 3: FEATURE ENGINEERING")
    print("-" * 80)
    df_features = create_features(df_cleaned)
    
    # === STEP 4: MODEL TRAINING ===
    print("\nSTEP 4: MODEL TRAINING")
    print("-" * 80)
    all_results, overall_best = train_all_scenarios(df_features)
    
    # === STEP 5: SAVE METRICS ===
    print("\nSTEP 5: SAVING METRICS")
    print("-" * 80)
    save_metrics(all_results, str(metrics_csv_path), str(metrics_json_path))
    
    # === STEP 6: SAVE BEST MODEL ===
    print("\nSTEP 6: SAVING BEST MODEL")
    print("-" * 80)
    best_model, best_model_type, best_scenario, best_metrics, y_pred, y_test = overall_best
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, str(best_model_path))
    print(f"Saved best model to {best_model_path}")
    
    # === STEP 7: GENERATE EVALUATION PLOTS ===
    print("\nSTEP 7: GENERATING EVALUATION PLOTS")
    print("-" * 80)
    metrics_df = pd.read_csv(str(metrics_csv_path))
    generate_evaluation_plots(metrics_df, y_test, y_pred, best_model, best_model_type, str(figures_dir))
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nBest scenario: {best_scenario}")
    print(f"Best model: {best_model_type.upper()}")
    print(f"Best R²: {best_metrics['r2']:.4f}")
    print(f"Best MAE: {best_metrics['mae']:.4f}")
    print(f"Best RMSE: {best_metrics['rmse']:.4f}")
    print(f"\nOutput files saved to:")
    print(f"  - Data: {cleaned_data_path}")
    print(f"  - Metrics: {metrics_csv_path}")
    print(f"  - Model: {best_model_path}")
    print(f"  - Figures: {figures_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
