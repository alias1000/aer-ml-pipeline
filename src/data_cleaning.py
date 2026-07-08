"""
Data cleaning module for AER dataset.

Performs:
- Column name standardization
- Duplicate row removal
- Constant column removal
- Data type conversion
- Missing value handling
"""

import pandas as pd
from pathlib import Path


def load_data(input_path):
    """Load raw CSV data."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows × {df.shape[1]} columns")
    return df


def clean_column_names(df):
    """Standardize column names to lowercase with underscores."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    return df


def remove_constant_columns(df):
    """Remove columns with only one unique value."""
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
    return df


def remove_redundant_columns(df):
    """Remove columns that are redundant (box_ver duplicates box_num info)."""
    cols_to_drop = []
    
    # Drop box_ver (perfect collinearity with box_num)
    if 'box_ver' in df.columns:
        cols_to_drop.append('box_ver')
    
    # Drop old inlet_angle_sine (will recalculate correctly)
    if 'inlet_angle_sine' in df.columns:
        cols_to_drop.append('inlet_angle_sine')
    
    if cols_to_drop:
        print(f"Removing redundant columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


def convert_data_types(df):
    """Convert columns to appropriate data types."""
    # Categorical columns
    categorical_cols = []
    if 'window' in df.columns:
        df['window'] = df['window'].astype('category')
        categorical_cols.append('window')
    if 'rich_num' in df.columns:
        df['rich_num'] = df['rich_num'].astype('category')
        categorical_cols.append('rich_num')
    if 'old_new' in df.columns:
        df['old_new'] = df['old_new'].astype('category')
        categorical_cols.append('old_new')
    
    # Ensure numerical columns are numeric
    numerical_cols = ['AER', 'temp', 'vel', 'inlet_angle', 'richardson_num', 'Lwratio']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if categorical_cols:
        print(f"Converted to categorical: {categorical_cols}")
    
    return df


def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    print(f"Final dataset: {len(df)} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")


def clean_data(input_path, output_path):
    """Main cleaning pipeline."""
    print("="*80)
    print("DATA CLEANING")
    print("="*80 + "\n")
    
    df = load_data(input_path)
    print(f"\nInitial shape: {df.shape}")
    
    df = clean_column_names(df)
    df = remove_duplicates(df)
    df = remove_constant_columns(df)
    df = remove_redundant_columns(df)
    df = convert_data_types(df)
    
    print(f"\nFinal shape: {df.shape}")
    save_cleaned_data(df, output_path)
    
    print("\n" + "="*80 + "\n")
    
    return df
