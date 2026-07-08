"""
Feature engineering module for AER dataset.

Creates derived features:
- Recalculates inlet angle trigonometric features correctly
- Encodes categorical variables
- Prepares features for modeling
"""

import pandas as pd
import numpy as np


def create_features(df):
    """
    Create engineered features from raw data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
    
    Returns:
    --------
    df : pd.DataFrame
        Dataset with engineered features
    """
    df = df.copy()
    
    print("Creating engineered features...")
    
    # === Trigonometric features from inlet_angle ===
    # Recalculate correctly using degrees
    if 'inlet_angle' in df.columns:
        df['inlet_angle_sine'] = np.sin(np.deg2rad(df['inlet_angle']))
        df['inlet_angle_cosine'] = np.cos(np.deg2rad(df['inlet_angle']))
        print("  [OK] Created inlet_angle_sine (recalculated correctly)")
        print("  [OK] Created inlet_angle_cosine")
    
    print("\nFeature engineering complete.")
    print(f"Final columns: {list(df.columns)}")
    
    return df
